import argparse
import random
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from src.data.augmentations import build_transforms
from src.data.dataset import EggDetectionDataset
from src.models.faster_rcnn import build_model


# ─── Utilities ──────────────────────────────────────────────────────────


def collate_fn(batch):
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
    """After model.train(), reset frozen stages to eval mode for BatchNorm."""
    for stage_name in frozen_stages:
        stage = getattr(model.backbone.body, stage_name)
        stage.eval()


# ─── Parameter Groups ──────────────────────────────────────────────────


def build_param_groups(model, config, frozen_stage_names):
    """Build optimizer parameter groups with differential learning rates.

    Frozen stages are excluded — they get added via add_param_group() on unfreeze.
    """
    base_lr = config["optimizer"]["lr"]
    multipliers = config["optimizer"]["lr_multipliers"]

    backbone_params = []
    fpn_params = []
    head_params = []

    # Backbone body (excluding frozen stages)
    for name, param in model.backbone.body.named_parameters():
        stage = name.split(".")[0]
        if stage in frozen_stage_names:
            continue
        if param.requires_grad:
            backbone_params.append(param)

    # FPN
    for param in model.backbone.fpn.parameters():
        if param.requires_grad:
            fpn_params.append(param)

    # Heads (RPN + ROI heads)
    for param in model.rpn.parameters():
        if param.requires_grad:
            head_params.append(param)
    for param in model.roi_heads.parameters():
        if param.requires_grad:
            head_params.append(param)

    groups = []
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": base_lr * multipliers["backbone"],
            "name": "backbone",
        })
    groups.append({
        "params": fpn_params,
        "lr": base_lr * multipliers["fpn"],
        "name": "fpn",
    })
    groups.append({
        "params": head_params,
        "lr": base_lr * multipliers["heads"],
        "name": "heads",
    })

    return groups


def unfreeze_stage(model, optimizer, stage_name, config):
    """Unfreeze a backbone stage and add its parameters to the optimizer."""
    stage = getattr(model.backbone.body, stage_name)
    for param in stage.parameters():
        param.requires_grad = True

    base_lr = config["optimizer"]["lr"]
    multiplier = config["optimizer"]["lr_multipliers"]["backbone"]

    optimizer.add_param_group({
        "params": list(stage.parameters()),
        "lr": base_lr * multiplier,
        "name": f"backbone_{stage_name}",
    })


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
def validate(model, val_loader, device):
    """Run validation and return mAP metrics."""
    model.eval()
    metric = MeanAveragePrecision()

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

    return {
        "mAP@0.5": results["map_50"].item(),
        "mAP@0.5:0.95": results["map"].item(),
    }


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
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Seed
    set_seed(config["training"]["seed"])

    # Device
    device = torch.device(args.device) if args.device else auto_device()
    print(f"Device: {device}")

    # W&B
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            tags=config["wandb"].get("tags", []),
            config=config,
        )

    # Data split
    train_ids, val_ids = split_image_ids(config)
    print(f"Train images: {len(train_ids)} | Val images: {len(val_ids)}")

    # Augmentations (training only)
    train_transforms = build_transforms(config)

    # Datasets
    annotations_csv = config["data"]["annotations_csv"]
    image_dir = config["data"]["image_dir"]
    max_images = config["data"].get("max_images") or None
    train_dataset = EggDetectionDataset(
        annotations_csv, image_dir, image_ids=train_ids, transforms=train_transforms,
        max_images=max_images,
    )
    val_dataset = EggDetectionDataset(
        annotations_csv, image_dir, image_ids=val_ids,
        max_images=max_images,
    )

    # DataLoaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    seed = config["training"]["seed"]
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        worker_init_fn=WorkerInitFn(seed), pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    # Model
    model = build_model(config)
    model.to(device)

    # Freeze schedule
    frozen_stages = get_frozen_stages(config)
    frozen_stage_names = set(frozen_stages.keys())
    for stage_name in sorted(frozen_stage_names):
        freeze_stage(model, stage_name)
        print(f"Frozen: {stage_name} (unfreeze after epoch {frozen_stages[stage_name]})")

    # Optimizer
    param_groups = build_param_groups(model, config, frozen_stage_names)
    opt_type = config["optimizer"]["type"]
    if opt_type == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=config["optimizer"]["momentum"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=config["optimizer"]["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    # Scheduler
    scheduler = build_scheduler(optimizer, config)

    # AMP scaler
    use_amp = config["training"].get("amp", False)
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # Checkpointing — per-run directory
    config_name = Path(args.config).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config_name}_{timestamp}"
    ckpt_dir = Path(config["checkpoint"]["dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, ckpt_dir / "config.yaml")
    print(f"Run directory: {ckpt_dir}")
    best_metric = -1.0
    best_epoch = 0
    monitor = config["checkpoint"]["monitor"]
    if monitor not in ("mAP@0.5", "mAP@0.5:0.95"):
        raise ValueError(f"Unknown monitor metric: {monitor}")

    # Training
    epochs = config["training"]["epochs"]
    global_step = 0
    patience = config["checkpoint"].get("early_stopping_patience", 0)

    print(f"\nStarting training for {epochs} epochs\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Check for unfreezes
        stages_to_unfreeze = [
            name for name, after_epoch in frozen_stages.items()
            if epoch == after_epoch + 1 and name in frozen_stage_names
        ]
        for stage_name in stages_to_unfreeze:
            unfreeze_stage(model, optimizer, stage_name, config)
            frozen_stage_names.discard(stage_name)
            lr = config["optimizer"]["lr"] * config["optimizer"]["lr_multipliers"]["backbone"]
            print(f"→ Unfreezing {stage_name} (lr={lr})")

        # Train
        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, scaler, config,
            wandb_run, global_step, frozen_stage_names,
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

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
            wandb_run.log(epoch_log, step=global_step)

        # Checkpointing
        current_metric = val_metrics[monitor]

        if config["checkpoint"]["save_best"] and current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"→ Saved best model ({monitor}: {best_metric:.4f})")

        save_every = config["checkpoint"]["save_every_n_epochs"]
        if save_every > 0 and epoch % save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")
            print(f"→ Saved checkpoint: epoch_{epoch}.pt")

        # Early stopping
        if patience > 0 and (epoch - best_epoch) >= patience:
            print(f"→ Early stopping: no improvement for {patience} epochs")
            break

    print(f"\nTraining complete. Best {monitor}: {best_metric:.4f}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
