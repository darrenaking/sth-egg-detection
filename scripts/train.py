"""
Train Faster R-CNN for parasite egg detection.
Pipeline step 3 of 3: download_data.py → process_annotations.py → train.py
Usage: python scripts/train.py --config configs/faster_rcnn_baseline.yaml
See docs/train_design_decisions.txt for rationale behind training choices.
"""
import argparse
import random
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from src.data.augmentations import build_transforms
from src.data.dataset import EggDetectionDataset
from src.models.faster_rcnn import build_model


# ─── Utilities ──────────────────────────────────────────────────────────


def collate_fn(batch):
    # Detection models expect a list of images and a list of target dicts,
    # not a stacked batch (each image has a different number of boxes).
    return tuple(zip(*batch))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WorkerInitFn:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed(self.seed + worker_id)
        random.seed(self.seed + worker_id)
        torch.manual_seed(self.seed + worker_id)


def auto_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ─── Data Splitting ────────────────────────────────────────────────────


def split_image_ids(config):
    """Return (train_ids, val_ids) using stratified split on filename_class."""
    df = pd.read_csv(config["data"]["annotations_csv"])
    # One row per image for stratification
    image_df = df.drop_duplicates(subset="image_id")[["image_id", "filename_class"]]

    strategy = config["validation"]["strategy"]

    if strategy == "split":
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config["validation"]["val_split"],
            random_state=config["training"]["seed"],
        )
        train_idx, val_idx = next(
            splitter.split(image_df["image_id"], image_df["filename_class"])
        )
        train_ids = image_df["image_id"].iloc[train_idx].values
        val_ids = image_df["image_id"].iloc[val_idx].values

    elif strategy == "kfold":
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=config["validation"]["num_folds"],
            shuffle=True,
            random_state=config["training"]["seed"],
        )
        fold = config["validation"]["fold"]  # 1-indexed
        num_folds = config["validation"]["num_folds"]
        if not (1 <= fold <= num_folds):
            raise ValueError(f"fold must be between 1 and {num_folds}, got {fold}")
        splits = list(
            skf.split(image_df["image_id"], image_df["filename_class"])
        )
        train_idx, val_idx = splits[fold - 1]
        train_ids = image_df["image_id"].iloc[train_idx].values
        val_ids = image_df["image_id"].iloc[val_idx].values

    else:
        raise ValueError(f"Unknown validation strategy: {strategy}")

    return train_ids, val_ids


# ─── Freeze / Unfreeze ─────────────────────────────────────────────────


def get_frozen_stages(config):
    """Return dict of {stage_name: unfreeze_after_epoch} from config."""
    schedule = config.get("freeze_schedule", {}).get("temporary", {})
    return {name: info["unfreeze_after_epoch"] for name, info in schedule.items()}


def freeze_stage(model, stage_name):
    """Freeze a backbone stage and set its BatchNorm to eval mode."""
    stage = getattr(model.backbone.body, stage_name)
    for param in stage.parameters():
        param.requires_grad = False
    stage.eval()


def set_frozen_bn_eval(model, frozen_stages):
    """Re-freeze BatchNorm for frozen stages after model.train().

    model.train() sets all modules to train mode, which would cause frozen
    BatchNorm layers to use batch stats instead of running stats. This
    must be called after model.train() each epoch.
    """
    for stage_name in frozen_stages:
        stage = getattr(model.backbone.body, stage_name)
        stage.eval()


# ─── Parameter Groups ──────────────────────────────────────────────────


def build_param_groups(model, config):
    """Build optimizer parameter groups with differential learning rates.

    All backbone stages get their own param group (even frozen ones),
    so the scheduler tracks them from the start. PyTorch 2.x optimizers
    skip params with grad=None, so frozen params are not updated.
    """
    base_lr = config["optimizer"]["lr"]
    multipliers = config["optimizer"]["lr_multipliers"]

    # Per-stage backbone groups (iteration order matches module registration)
    stage_params = {}
    for name, param in model.backbone.body.named_parameters():
        stage = name.split(".")[0]
        stage_params.setdefault(stage, []).append(param)

    groups = []
    for stage, params in stage_params.items():
        groups.append({
            "params": params,
            "lr": base_lr * multipliers["backbone"],
            "name": f"backbone_{stage}",
        })

    # FPN
    fpn_params = list(model.backbone.fpn.parameters())
    groups.append({
        "params": fpn_params,
        "lr": base_lr * multipliers["fpn"],
        "name": "fpn",
    })

    # Heads (RPN + ROI heads)
    head_params = list(model.rpn.parameters()) + list(model.roi_heads.parameters())
    groups.append({
        "params": head_params,
        "lr": base_lr * multipliers["heads"],
        "name": "heads",
    })

    return groups


def unfreeze_stage(model, stage_name):
    """Unfreeze a backbone stage."""
    stage = getattr(model.backbone.body, stage_name)
    for param in stage.parameters():
        param.requires_grad = True
    stage.train()


# ─── LR Scheduler ──────────────────────────────────────────────────────


def build_scheduler(optimizer, config):
    """Build LR scheduler with optional linear warmup."""
    sched_cfg = config["scheduler"]
    epochs = config["training"]["epochs"]
    warmup_epochs = sched_cfg.get("warmup_epochs", 0)
    sched_type = sched_cfg["type"]

    # T_max / milestones account for warmup: the main scheduler only runs
    # for (epochs - warmup_epochs) steps since SequentialLR handles the split.
    remaining = epochs - warmup_epochs

    if sched_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=sched_cfg["eta_min"]
        )
    elif sched_type == "step":
        # step_size is a relative interval — works inside SequentialLR
        # without adjustment (unlike milestones, which are absolute)
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg["step_size"],
            gamma=sched_cfg["gamma"],
        )
    elif sched_type == "multistep":
        # Shift milestones back by warmup_epochs so they refer to overall epoch numbers
        adjusted = [m - warmup_epochs for m in sched_cfg["milestones"]]
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=adjusted,
            gamma=sched_cfg["gamma"],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=sched_cfg["warmup_start_factor"],
            total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

    return main_scheduler


# ─── Gradient Norms ─────────────────────────────────────────────────────


def compute_gradient_norms(optimizer):
    """Compute L2 gradient norm per named parameter group."""
    norms = {}
    for group in optimizer.param_groups:
        name = group.get("name", "unknown")
        total_norm = 0.0
        for p in group["params"]:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        norms[name] = total_norm ** 0.5
    return norms


# ─── Validation ─────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, val_loader, device, class_names=None):
    """Run validation and return mAP metrics (aggregate + per-class)."""
    model.eval()
    metric = MeanAveragePrecision(class_metrics=True)

    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        predictions = model(images)

        preds = []
        gts = []
        for pred, tgt in zip(predictions, targets):
            preds.append({
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
                "labels": pred["labels"].cpu(),
            })
            gts.append({
                "boxes": tgt["boxes"].cpu(),
                "labels": tgt["labels"].cpu(),
            })

        metric.update(preds, gts)

    results = metric.compute()

    metrics = {
        "mAP@0.5": results["map_50"].item(),
        "mAP@0.5:0.95": results["map"].item(),
    }

    # Per-class AP@0.5:0.95 (already computed, zero extra cost)
    if "map_per_class" in results and class_names:
        per_class_ap = results["map_per_class"]
        class_ids = results.get("classes", None)
        if class_ids is not None:
            for i, cid in enumerate(class_ids.tolist()):
                cid = int(cid)
                name = class_names.get(cid, f"class_{cid}")
                metrics[f"AP/{name}"] = per_class_ap[i].item()

    return metrics


# ─── Training Loop ──────────────────────────────────────────────────────


def train_one_epoch(model, train_loader, optimizer, device, scaler, config, wandb_run,
                    global_step, frozen_stage_names):
    """Train for one epoch. Returns average loss and updated global_step."""
    model.train()
    set_frozen_bn_eval(model, frozen_stage_names)

    log_every = config["wandb"].get("log_every_n_steps", 10)
    log_gradients = config["wandb"].get("log_gradient_norms", False)
    use_amp = config["training"].get("amp", False)
    clip_norm = config["optimizer"].get("clip_grad_norm", 0)

    running_loss = 0.0
    num_batches = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        running_loss += total_loss.item()
        num_batches += 1
        global_step += 1

        # Step-level W&B logging
        if wandb_run and global_step % log_every == 0:
            step_log = {
                f"train/{k}": v.item() for k, v in loss_dict.items()
            }
            step_log["train/total_loss"] = total_loss.item()
            for group in optimizer.param_groups:
                name = group.get("name", "unknown")
                step_log[f"lr/{name}"] = group["lr"]

            if log_gradients:
                grad_norms = compute_gradient_norms(optimizer)
                for name, norm in grad_norms.items():
                    step_log[f"grad_norm/{name}"] = norm

            wandb_run.log(step_log, step=global_step)

    avg_loss = running_loss / num_batches
    return avg_loss, global_step


# ─── Checkpointing ─────────────────────────────────────────────────────


def save_checkpoint(path, *, epoch, global_step, model, optimizer, scheduler,
                    scaler, best_metric, best_epoch, frozen_stage_names,
                    wandb_run_id, config):
    """Save a full resumable checkpoint with atomic write."""
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "frozen_stage_names": list(frozen_stage_names),
        "wandb_run_id": wandb_run_id,
        "config": config,
    }
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path):
    """Load a checkpoint and return the state dict.

    Raises ValueError if the file is a model-only checkpoint.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "epoch" not in ckpt:
        raise ValueError(
            f"{path} appears to be a model-only checkpoint, not a resumable "
            f"checkpoint. Use a last.pt or epoch_N.pt file from a run that "
            f"saved full state."
        )
    return ckpt


# ─── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train STH egg detector")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to train on (default: auto-detect cuda > mps > cpu)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a resumable checkpoint (e.g. last.pt) to continue training"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Seed
    set_seed(config["training"]["seed"])

    # Device
    device = torch.device(args.device) if args.device else auto_device()
    print(f"Device: {device}")

    # Load resume checkpoint (if any) before building other components
    resuming = args.resume is not None
    ckpt = None
    if resuming:
        ckpt = load_checkpoint(Path(args.resume))
        # Warn on config mismatches
        saved_config = ckpt.get("config", {})
        for key in ("model", "optimizer", "scheduler", "data", "validation"):
            if key in saved_config and saved_config[key] != config.get(key):
                warnings.warn(
                    f"Config section '{key}' differs from checkpoint. "
                    f"Using the provided --config value."
                )

    # W&B
    wandb_run = None
    if not args.no_wandb:
        import wandb
        if resuming and ckpt.get("wandb_run_id"):
            wandb_run = wandb.init(
                project=config["wandb"]["project"],
                tags=config["wandb"].get("tags", []),
                config=config,
                id=ckpt["wandb_run_id"],
                resume="must",
            )
        else:
            wandb_run = wandb.init(
                project=config["wandb"]["project"],
                tags=config["wandb"].get("tags", []),
                config=config,
            )

    # Data split
    train_ids, val_ids = split_image_ids(config)
    print(f"Train images: {len(train_ids)} | Val images: {len(val_ids)}")

    # Subsample for debugging (shuffle so subset isn't biased by CSV row order)
    max_images = config["data"].get("max_images") or None
    if max_images:
        rng = np.random.default_rng(config["training"]["seed"])
        rng.shuffle(train_ids)
        rng.shuffle(val_ids)
        train_ids = train_ids[:max_images]
        val_ids = val_ids[:max_images]
        print(f"  (max_images={max_images}: using {len(train_ids)} train, {len(val_ids)} val)")

    # Augmentations (training only)
    train_transforms = build_transforms(config)

    # Datasets
    annotations_csv = config["data"]["annotations_csv"]
    image_dir = config["data"]["image_dir"]
    train_dataset = EggDetectionDataset(
        annotations_csv, image_dir, image_ids=train_ids, transforms=train_transforms,
    )
    val_dataset = EggDetectionDataset(
        annotations_csv, image_dir, image_ids=val_ids,
    )
    class_names = train_dataset.class_names

    if len(train_dataset) == 0:
        raise ValueError(
            f"Training dataset is empty. Check annotations_csv "
            f"({annotations_csv}) and image_dir ({image_dir})."
        )

    # DataLoaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    seed = config["training"]["seed"]
    pin_memory = device.type == "cuda"
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        worker_init_fn=WorkerInitFn(seed), pin_memory=pin_memory,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    # Model (build on CPU, load checkpoint if resuming, then move to device)
    model = build_model(config)

    # Freeze schedule
    frozen_stages = get_frozen_stages(config)
    if resuming:
        # Restore freeze state from checkpoint (single source of truth)
        frozen_stage_names = set(ckpt["frozen_stage_names"])
        for stage_name in frozen_stage_names:
            freeze_stage(model, stage_name)
            print(f"Frozen (restored): {stage_name}")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed model from {args.resume} (epoch {ckpt['epoch']})")
    else:
        # Fresh run: freeze all stages from config
        frozen_stage_names = set(frozen_stages.keys())
        for stage_name in sorted(frozen_stage_names):
            freeze_stage(model, stage_name)
            print(f"Frozen: {stage_name} (unfreeze after epoch {frozen_stages[stage_name]})")

    model.to(device)

    # Optimizer (built after model is on device)
    param_groups = build_param_groups(model, config)
    opt_type = config["optimizer"]["type"]
    if opt_type == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=config["optimizer"]["momentum"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config["optimizer"]["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    # Scheduler
    scheduler = build_scheduler(optimizer, config)

    # AMP scaler
    use_amp = config["training"].get("amp", False)
    if use_amp and device.type == "mps":
        warnings.warn("AMP is not reliably supported on MPS devices. Disabling.")
        use_amp = False
        config["training"]["amp"] = False
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # Restore optimizer/scheduler/scaler state on resume
    if resuming:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Move optimizer state tensors to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        saved_scaler = ckpt.get("scaler_state_dict")
        if scaler is not None and saved_scaler is not None:
            scaler.load_state_dict(saved_scaler)
        elif (scaler is None) != (saved_scaler is None):
            warnings.warn(
                "AMP setting changed since checkpoint was saved. "
                "Scaler state will not be restored."
            )

    # Run directory
    if resuming:
        ckpt_dir = Path(args.resume).parent
    else:
        config_name = Path(args.config).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config_name}_{timestamp}"
        ckpt_dir = Path(config["checkpoint"]["dir"]) / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, ckpt_dir / "config.yaml")
    print(f"Run directory: {ckpt_dir}")

    # Training state
    if resuming:
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_metric = ckpt["best_metric"]
        best_epoch = ckpt["best_epoch"]
    else:
        start_epoch = 1
        global_step = 0
        best_metric = -1.0
        best_epoch = 0

    monitor = config["checkpoint"]["monitor"]
    if monitor not in ("mAP@0.5", "mAP@0.5:0.95"):
        raise ValueError(f"Unknown monitor metric: {monitor}")

    epochs = config["training"]["epochs"]
    patience = config["checkpoint"].get("early_stopping_patience", 0)
    wandb_run_id = wandb_run.id if wandb_run else None

    if resuming:
        print(f"\nResuming training from epoch {start_epoch} to {epochs}\n")
    else:
        print(f"\nStarting training for {epochs} epochs\n")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Check for unfreezes
        stages_to_unfreeze = [
            name for name, after_epoch in frozen_stages.items()
            if epoch == after_epoch + 1 and name in frozen_stage_names
        ]
        for stage_name in stages_to_unfreeze:
            unfreeze_stage(model, stage_name)
            frozen_stage_names.discard(stage_name)
            lr = config["optimizer"]["lr"] * config["optimizer"]["lr_multipliers"]["backbone"]
            print(f"→ Unfreezing {stage_name} (base_lr={lr})")

        # Train
        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, scaler, config,
            wandb_run, global_step, frozen_stage_names,
        )

        # Validate
        val_metrics = validate(model, val_loader, device, class_names)

        # Scheduler step
        scheduler.step()

        # Epoch summary
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss: {avg_loss:.4f} | "
            f"mAP@0.5: {val_metrics['mAP@0.5']:.4f} | "
            f"mAP@0.5:0.95: {val_metrics['mAP@0.5:0.95']:.4f} | "
            f"lr: {current_lr:.6f} | "
            f"{format_time(elapsed)}"
        )

        # Epoch-level W&B logging
        if wandb_run:
            epoch_log = {
                "epoch": epoch,
                "train/avg_loss": avg_loss,
                "val/mAP@0.5": val_metrics["mAP@0.5"],
                "val/mAP@0.5:0.95": val_metrics["mAP@0.5:0.95"],
            }
            for key, value in val_metrics.items():
                if key.startswith("AP/"):
                    epoch_log[f"val/per_class/{key}"] = value
            wandb_run.log(epoch_log, step=global_step)

        # Checkpointing
        current_metric = val_metrics[monitor]

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            if config["checkpoint"]["save_best"]:
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
                print(f"→ Saved best model ({monitor}: {best_metric:.4f})")

        save_every = config["checkpoint"]["save_every_n_epochs"]
        if save_every > 0 and epoch % save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")
            print(f"→ Saved checkpoint: epoch_{epoch}.pt")

        # Always save last.pt (full resumable state, atomic write)
        save_checkpoint(
            ckpt_dir / "last.pt",
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_metric=best_metric,
            best_epoch=best_epoch,
            frozen_stage_names=frozen_stage_names,
            wandb_run_id=wandb_run_id,
            config=config,
        )

        # Early stopping
        if patience > 0 and (epoch - best_epoch) >= patience:
            print(f"→ Early stopping: no improvement for {patience} epochs")
            break

    print(f"\nTraining complete. Best {monitor}: {best_metric:.4f}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
