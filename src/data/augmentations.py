import albumentations as A


def build_transforms(config):
    """Build an Albumentations Compose pipeline from the config's augmentation flags."""
    aug_cfg = config["data"]["augmentations"]
    min_visibility = aug_cfg.get("min_visibility", 0.3)

    transforms_list = []

    if aug_cfg.get("horizontal_flip", False):
        transforms_list.append(A.HorizontalFlip(p=0.5))

    if aug_cfg.get("vertical_flip", False):
        transforms_list.append(A.VerticalFlip(p=0.5))

    rotate_cfg = aug_cfg.get("rotate", {})
    if rotate_cfg.get("enabled", False):
        transforms_list.append(A.Rotate(limit=rotate_cfg.get("limit", 30), p=0.5, border_mode=0))

    cj_cfg = aug_cfg.get("color_jitter", {})
    if cj_cfg.get("enabled", False):
        transforms_list.append(
            A.ColorJitter(
                brightness=cj_cfg.get("brightness", 0.2),
                contrast=cj_cfg.get("contrast", 0.2),
                saturation=cj_cfg.get("saturation", 0.2),
                hue=cj_cfg.get("hue", 0.1),
                p=0.5,
            )
        )

    if not transforms_list:
        return None

    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=min_visibility,
        ),
    )
