import logging
import numpy as np
import pandas as pd
import torch
from torchvision.transforms.functional import to_tensor
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EggDetectionDataset(Dataset):
    def __init__(self, annotations_csv, image_dir, image_ids=None, transforms=None):
        self.image_dir = Path(image_dir)
        self.transforms = transforms

        df = pd.read_csv(annotations_csv)
        if image_ids is not None:
            df = df[df["image_id"].isin(image_ids)]
        self.image_ids = df["image_id"].unique()
        self.annotations = df.groupby("image_id")

        # Class name mapping: model label (category_id + 1) -> name
        id_to_name = df.drop_duplicates("category_id").set_index("category_id")["category_name"]
        self.class_names = {cid + 1: name for cid, name in id_to_name.items()}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.annotations.get_group(image_id)

        file_name = records.iloc[0]["file_name"]
        image = Image.open(self.image_dir / file_name).convert("RGB")

        # Bbox format: COCO xywh (CSV) → Pascal VOC xyxy (here) → stays xyxy
        # through augmentations (format="pascal_voc") and into the model.
        boxes = records[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].values.copy()
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # Clip boxes to image boundaries (defensive — data should already be clean)
        w, h = image.size
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

        # Shift category IDs by +1 (torchvision reserves 0 for background)
        labels = records["category_id"].values + 1

        if self.transforms:
            image = np.array(image)
            # Augmentations with min_visibility can drop all boxes (e.g., rotation
            # pushing them out of frame). Retry up to 10 times; fall back to
            # un-augmented image if all attempts fail. Faster R-CNN requires at
            # least one box per image. With current augmentations (horizontal flip
            # only) this retry effectively never triggers.
            augmented = False
            for _ in range(10):
                transformed = self.transforms(
                    image=image, bboxes=boxes.tolist(), labels=labels.tolist()
                )
                if len(transformed["bboxes"]) > 0:
                    augmented = True
                    break
            if augmented:
                image = to_tensor(transformed["image"])
                boxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
                labels = np.array(transformed["labels"], dtype=np.int64)
            else:
                logger.warning(
                    "Augmentation dropped all boxes for image_id=%d after 10 retries. "
                    "Using un-augmented image.", image_id
                )
                image = to_tensor(image)
        else:
            image = to_tensor(image)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([image_id]),
            "area": torch.as_tensor(
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros(len(labels), dtype=torch.uint8),
        }

        return image, target
