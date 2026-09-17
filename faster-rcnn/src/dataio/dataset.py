"""YOLO-format dataset adapter for torchvision detection models.

Reads the same ``images/<split>`` + ``labels/<split>`` layout that Ultralytics
uses and yields ``(image_tensor, target_dict)`` pairs in torchvision's
detection format.
"""

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

try:
    from configs.config import IMG_EXTS
except ImportError:
    from config import IMG_EXTS


# ---------------------------------------------------------------------------
# data.yaml handling
# ---------------------------------------------------------------------------
def load_data_config(data_yaml):
    """Resolve a YOLO ``data.yaml`` into (dataset_root, split_dirs, class_names)."""
    data_yaml = Path(data_yaml).resolve()

    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("path"):
        dataset_root = Path(cfg["path"])
        if not dataset_root.is_absolute():
            dataset_root = (data_yaml.parent / dataset_root).resolve()
    else:
        dataset_root = data_yaml.parent.resolve()

    split_dirs = {}
    for split in ("train", "val", "test"):
        entry = cfg.get(split)
        if entry is None:
            continue

        entry_path = Path(entry)
        if not entry_path.is_absolute():
            entry_path = (dataset_root / entry_path).resolve()

        split_dirs[split] = entry_path

    names = cfg.get("names", ["object"])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    return dataset_root, split_dirs, list(names)


def labels_dir_for(images_dir):
    """Map ``.../images/train`` to ``.../labels/train``."""
    images_dir = Path(images_dir)
    parts = list(images_dir.parts)

    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)

    raise ValueError(f"Could not derive a labels directory from {images_dir}")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def _hflip(img, boxes):
    width = img.shape[2]
    img = torch.flip(img, dims=[2])

    if len(boxes):
        x1 = boxes[:, 0].clone()
        boxes[:, 0] = width - boxes[:, 2]
        boxes[:, 2] = width - x1

    return img, boxes


def _vflip(img, boxes):
    height = img.shape[1]
    img = torch.flip(img, dims=[1])

    if len(boxes):
        y1 = boxes[:, 1].clone()
        boxes[:, 1] = height - boxes[:, 3]
        boxes[:, 3] = height - y1

    return img, boxes


def _scale_translate(img, boxes, labels, scale_jitter, translate_frac, min_visibility=0.3):
    """Random zoom + shift on a canvas of the original size.

    Mirrors the YOLO run's ``scale=0.25`` / ``translate=0.05``. Boxes that end
    up with less than ``min_visibility`` of their original area inside the
    canvas are dropped.
    """
    _, height, width = img.shape

    scale = random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))

    resized = torch.nn.functional.interpolate(
        img.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Center the resized image, then jitter it.
    off_y = (height - new_h) // 2 + int(round(random.uniform(-translate_frac, translate_frac) * height))
    off_x = (width - new_w) // 2 + int(round(random.uniform(-translate_frac, translate_frac) * width))

    canvas = torch.zeros_like(img)

    dst_y1 = max(0, off_y)
    dst_x1 = max(0, off_x)
    dst_y2 = min(height, off_y + new_h)
    dst_x2 = min(width, off_x + new_w)

    if dst_y2 <= dst_y1 or dst_x2 <= dst_x1:
        return img, boxes, labels  # Degenerate shift; skip the augmentation.

    src_y1 = dst_y1 - off_y
    src_x1 = dst_x1 - off_x
    canvas[:, dst_y1:dst_y2, dst_x1:dst_x2] = resized[
        :,
        src_y1 : src_y1 + (dst_y2 - dst_y1),
        src_x1 : src_x1 + (dst_x2 - dst_x1),
    ]

    if len(boxes):
        original_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        moved = boxes * scale
        moved[:, [0, 2]] += off_x
        moved[:, [1, 3]] += off_y

        moved[:, [0, 2]] = moved[:, [0, 2]].clamp(0, width)
        moved[:, [1, 3]] = moved[:, [1, 3]].clamp(0, height)

        new_area = (moved[:, 2] - moved[:, 0]).clamp(min=0) * (
            moved[:, 3] - moved[:, 1]
        ).clamp(min=0)

        keep = new_area >= (min_visibility * original_area * scale * scale)
        keep &= (moved[:, 2] - moved[:, 0]) > 2
        keep &= (moved[:, 3] - moved[:, 1]) > 2

        boxes = moved[keep]
        labels = labels[keep]

    return canvas, boxes, labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class YoloDetectionDataset(Dataset):
    """Ultralytics-style directory of images + normalized ``cls cx cy w h`` labels."""

    def __init__(
        self,
        images_dir,
        train=False,
        fliplr=0.5,
        flipud=0.5,
        scale=0.25,
        translate=0.05,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = labels_dir_for(self.images_dir)

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")

        self.image_paths = sorted(
            p
            for p in self.images_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
        )

        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.images_dir}")

        self.train = train
        self.fliplr = fliplr
        self.flipud = flipud
        self.scale = scale
        self.translate = translate

    def __len__(self):
        return len(self.image_paths)

    def label_path(self, image_path):
        return self.labels_dir / (Path(image_path).stem + ".txt")

    def read_boxes(self, image_path, width, height):
        """Return (boxes_xyxy_pixels, class_ids) for one image."""
        label_path = self.label_path(image_path)

        boxes = []
        classes = []

        if label_path.is_file():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue

                cls = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])

                x1 = (cx - w / 2.0) * width
                y1 = (cy - h / 2.0) * height
                x2 = (cx + w / 2.0) * width
                y2 = (cy + h / 2.0) * height

                x1 = max(0.0, min(x1, width))
                x2 = max(0.0, min(x2, width))
                y1 = max(0.0, min(y1, height))
                y2 = max(0.0, min(y2, height))

                if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
                    continue

                boxes.append([x1, y1, x2, y2])
                # torchvision reserves label 0 for background.
                classes.append(cls + 1)

        if boxes:
            return (
                torch.as_tensor(boxes, dtype=torch.float32),
                torch.as_tensor(classes, dtype=torch.int64),
            )

        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
            (0,), dtype=torch.int64
        )

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        with Image.open(image_path) as im:
            im = im.convert("RGB")
            width, height = im.size
            arr = np.array(im, dtype=np.uint8)

        img = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        boxes, labels = self.read_boxes(image_path, width, height)

        if self.train:
            if self.fliplr > 0 and random.random() < self.fliplr:
                img, boxes = _hflip(img, boxes)

            if self.flipud > 0 and random.random() < self.flipud:
                img, boxes = _vflip(img, boxes)

            if self.scale > 0 or self.translate > 0:
                img, boxes, labels = _scale_translate(
                    img, boxes, labels, self.scale, self.translate
                )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if len(boxes)
            else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return img, target

    def image_name(self, idx):
        return self.image_paths[idx].name


def collate_fn(batch):
    return tuple(zip(*batch))


def build_dataset(data_yaml, split, train=False, **aug_kwargs):
    _, split_dirs, names = load_data_config(data_yaml)

    if split not in split_dirs:
        raise KeyError(f"Split '{split}' is not defined in {data_yaml}")

    dataset = YoloDetectionDataset(split_dirs[split], train=train, **aug_kwargs)

    return dataset, names
