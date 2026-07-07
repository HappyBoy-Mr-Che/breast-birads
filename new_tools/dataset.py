"""
Dataset for breast ultrasound BIRADS classification + feature detection.

Merges two data sources:
  1. future/train/  — images with 4 feature labels
  2. classfy/train/ — images with BIRADS class labels
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    AUG_BRIGHTNESS,
    AUG_CUTMIX_ALPHA,
    AUG_ELASTIC_ALPHA,
    AUG_ELASTIC_PROB,
    AUG_HFLIP_PROB,
    AUG_MIXUP_ALPHA,
    AUG_ROTATION_DEG,
    AUG_SCALE_RANGE,
    BIRADS_CLASSES,
    FEATURE_TYPES,
    IMAGE_SIZE,
    RANDOM_SEED,
    TRAIN_VAL_SPLIT,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def imread_unicode(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _parse_yolo_label(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            return cls, bbox
    return None


def _find_file(stem, directory, exts=(".jpg", ".png")):
    for ext in exts:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class Transforms:
    """Augmentation pipeline for ultrasound images (config-driven intensity)."""

    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    def __call__(self, image: np.ndarray, bboxes: dict) -> tuple:
        h, w = image.shape[:2]

        if self.is_train:
            image, bboxes = self._random_affine(image, bboxes)
            image, bboxes = self._random_horizontal_flip(image, bboxes)
            image = self._random_brightness_contrast(image)
            image = self._elastic_transform(image)

        image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)

        image = (image.astype(np.float32) / 255.0).astype(np.float32)
        image = ((image - np.array([0.485, 0.456, 0.406], dtype=np.float32))
                 / np.array([0.229, 0.224, 0.225], dtype=np.float32))
        image = torch.from_numpy(image).permute(2, 0, 1)

        targets = {}
        for feat_name, label in bboxes.items():
            targets[feat_name] = torch.tensor(label, dtype=torch.float32)

        return image, targets

    # --- private augmentations ---

    def _random_affine(self, img, bboxes):
        """Random rotation + scale + translation from config."""
        deg = AUG_ROTATION_DEG
        angle = random.uniform(-deg, deg)
        scale = random.uniform(*AUG_SCALE_RANGE)
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        tx = random.uniform(-0.05, 0.05) * w
        ty = random.uniform(-0.05, 0.05) * h

        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        new_bboxes = {}
        for k, v in bboxes.items():
            cls, xc, yc, bw, bh = v
            px, py = xc * w, yc * h
            px2 = matrix[0, 0] * px + matrix[0, 1] * py + matrix[0, 2]
            py2 = matrix[1, 0] * px + matrix[1, 1] * py + matrix[1, 2]
            bw2 = bw * scale
            bh2 = bh * scale
            new_bboxes[k] = [cls, px2 / w, py2 / h, bw2, bh2]
        return img, new_bboxes

    def _random_horizontal_flip(self, img, bboxes):
        if random.random() < AUG_HFLIP_PROB:
            img = cv2.flip(img, 1)
            new_bboxes = {}
            for k, v in bboxes.items():
                cls, xc, yc, bw, bh = v
                new_bboxes[k] = [cls, 1.0 - xc, yc, bw, bh]
            return img, new_bboxes
        return img, bboxes

    def _random_brightness_contrast(self, img):
        b = AUG_BRIGHTNESS
        alpha = 1.0 + random.uniform(-b, b)
        beta = random.uniform(-30, 30)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if random.random() < 0.3:
            gamma = random.uniform(0.7, 1.3)
            table = (255.0 * (np.linspace(0, 1, 256) ** gamma)).astype(np.uint8)
            img = cv2.LUT(img, table)
        return img

    def _elastic_transform(self, img, sigma=4):
        """Mild elastic deformation for ultrasound texture variation."""
        if random.random() < AUG_ELASTIC_PROB:
            alpha = AUG_ELASTIC_ALPHA
            h, w = img.shape[:2]
            dx = cv2.GaussianBlur(
                (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                (0, 0), sigma
            ) * alpha
            dy = cv2.GaussianBlur(
                (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                (0, 0), sigma
            ) * alpha
            map_x = (np.arange(w) + dx).astype(np.float32)
            map_y = (np.arange(h)[:, None] + dy).astype(np.float32)
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return img


# ---------------------------------------------------------------------------
# MixUp / CutMix helpers
# ---------------------------------------------------------------------------

def _mixup_data(x, y_birads, y_features, alpha=0.2):
    """MixUp: blend two images and their labels."""
    if alpha <= 0:
        return x, y_birads, y_features
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * y_features["birads"].new_empty(0)  # placeholder
    mixed_x = lam * x + (1 - lam) * x[index]

    # BIRADS: treat as soft label
    y_birads_mixed = y_birads.clone()
    y_birads_a = y_birads.clone()
    y_birads_b = y_birads[index].clone()
    # Mark mixed samples with -2 so loss can handle them (use CE with soft targets)
    y_birads_mixed = (y_birads_a, y_birads_b, lam)

    y_features_mixed = {}
    for k, v in y_features.items():
        y_features_mixed[k] = (v, v[index], lam)

    return mixed_x, y_birads_mixed, y_features_mixed


def _cutmix_data(x, y_birads, y_features, alpha=0.2):
    """CutMix: replace a rectangular region of an image with another."""
    if alpha <= 0:
        return x, y_birads, y_features
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.shape
    index = torch.randperm(batch_size, device=x.device)

    # Generate random box
    cx = np.random.uniform(0, W)
    cy = np.random.uniform(0, H)
    cut_w = W * np.sqrt(1 - lam)
    cut_h = H * np.sqrt(1 - lam)
    x1 = int(np.clip(cx - cut_w / 2, 0, W))
    y1 = int(np.clip(cy - cut_h / 2, 0, H))
    x2 = int(np.clip(cx + cut_w / 2, 0, W))
    y2 = int(np.clip(cy + cut_h / 2, 0, H))

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    y_birads_mixed = (y_birads, y_birads[index], lam)
    y_features_mixed = {}
    for k, v in y_features.items():
        y_features_mixed[k] = (v, v[index], lam)

    return mixed_x, y_birads_mixed, y_features_mixed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BreastUltrasoundDataset(Dataset):
    """Merged dataset: future (feature) + classfy (BIRADS) sources."""

    def __init__(
        self,
        future_images_dir: str,
        future_labels_dir: dict,
        classfy_dir: str,
        is_train: bool = True,
        split_ratio: float = TRAIN_VAL_SPLIT,
    ):
        self.future_images_dir = Path(future_images_dir)
        self.future_labels_dir = {k: Path(v) for k, v in future_labels_dir.items()}
        self.classfy_dir = Path(classfy_dir)
        self.is_train = is_train
        self.transform = Transforms(is_train=is_train)

        future_stems = {
            p.stem for p in self.future_images_dir.glob("*.jpg")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        } | {
            p.stem for p in self.future_images_dir.glob("*.png")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        }

        classify_stems = set()
        for cls_name in BIRADS_CLASSES:
            img_dir = self.classfy_dir / cls_name / "images"
            if img_dir.exists():
                classify_stems |= {p.stem for p in img_dir.glob("*")}

        self.all_stems = sorted(future_stems | classify_stems)

        rng = np.random.RandomState(RANDOM_SEED)
        indices = rng.permutation(len(self.all_stems))
        split_n = int(len(self.all_stems) * split_ratio)
        chosen = indices[:split_n] if is_train else indices[split_n:]
        self.stems = [self.all_stems[i] for i in chosen]

        self.stem_to_birads = {}
        for idx, cls_name in enumerate(BIRADS_CLASSES):
            img_dir = self.classfy_dir / cls_name / "images"
            if not img_dir.exists():
                continue
            for p in img_dir.glob("*"):
                self.stem_to_birads[p.stem] = idx

        self._cls_stem_to_path = {}
        for cls_name in BIRADS_CLASSES:
            img_dir = self.classfy_dir / cls_name / "images"
            if img_dir.exists():
                for p in img_dir.glob("*"):
                    self._cls_stem_to_path[p.stem] = p

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        stem = self.stems[index]

        img_path = _find_file(stem, self.future_images_dir) or self._cls_stem_to_path.get(stem)
        if img_path is None:
            raise FileNotFoundError(f"No image for stem: {stem}")
        image = imread_unicode(str(img_path))

        feature_bboxes = {}
        for feat_name in FEATURE_TYPES:
            label_path = self.future_labels_dir[feat_name] / f"{stem}.txt"
            label = [0, 0.0, 0.0, 0.0, 0.0]
            if label_path.exists():
                parsed = _parse_yolo_label(str(label_path))
                if parsed is not None:
                    cls, bbox = parsed
                    label = [cls] + bbox
            feature_bboxes[feat_name] = label

        birads_idx = self.stem_to_birads.get(stem, -1)

        image, targets = self.transform(image, feature_bboxes)
        targets["birads"] = torch.tensor(birads_idx, dtype=torch.long)

        return image, targets, stem


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = {}
    for key in batch[0][1].keys():
        targets[key] = torch.stack([item[1][key] for item in batch])
    stems = [item[2] for item in batch]
    return images, targets, stems
