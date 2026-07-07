"""
Inference script for test data — produces BIRADS classification + feature detection results.

Usage:
    python infer.py --ckpt outputs/models/best.pt

Outputs:
    outputs/submission/
        class_result.json    — BIRADS classification results
        future_result.json   — Feature detection results
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def imread_unicode(path):
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BIRADS_CLASSES,
    DEVICE,
    FEATURE_TYPES,
    FEATURE_CLASS_NAMES,
    IMAGE_SIZE,
    MODEL_DIR,
    OUTPUT_DIR,
    TEST_CLASS,
    TEST_FUTURE,
)
from models.multitask import create_model
from utils import load_checkpoint


# ---------------------------------------------------------------------------
# Test datasets
# ---------------------------------------------------------------------------

class TestClassDataset(Dataset):
    """Test set for BIRADS classification."""

    def __init__(self, root_dir, image_size=IMAGE_SIZE):
        self.root = Path(root_dir)
        self.image_size = image_size
        self.samples = []

        for cls_name in BIRADS_CLASSES:
            cls_dir = self.root / cls_name / "images"
            if cls_dir.exists():
                for p in cls_dir.glob("*"):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        self.samples.append((p, cls_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_name = self.samples[idx]
        image = imread_unicode(str(img_path))
        h, w = image.shape[:2]

        # Resize
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        image = (image.astype(np.float32) / 255.0).astype(np.float32)
        image = ((image - np.array([0.485, 0.456, 0.406], dtype=np.float32))
                 / np.array([0.229, 0.224, 0.225], dtype=np.float32))
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, img_path.stem, cls_name, (h, w)


class TestFutureDataset(Dataset):
    """Test set for feature detection."""

    def __init__(self, root_dir, image_size=IMAGE_SIZE):
        self.root = Path(root_dir)
        self.image_size = image_size
        self.images_dir = self.root / "images"
        self.samples = []

        for p in self.images_dir.glob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                self.samples.append(p)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = imread_unicode(str(img_path))
        h, w = image.shape[:2]

        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        image = (image.astype(np.float32) / 255.0).astype(np.float32)
        image = ((image - np.array([0.485, 0.456, 0.406], dtype=np.float32))
                 / np.array([0.229, 0.224, 0.225], dtype=np.float32))
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, img_path.stem, (h, w)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _denormalise_bbox(bbox, orig_h, orig_w):
    """Convert normalised [xc, yc, w, h] back to unnormalised pixel coords."""
    xc, yc, w, h = bbox
    return [xc * orig_w, yc * orig_h, w * orig_w, h * orig_h]


@torch.no_grad()
def run_classification(model, data_loader):
    """Run BIRADS classification on test set."""
    model.eval()
    results = []

    for images, stems, cls_names, orig_sizes in tqdm(data_loader, desc="Classify"):
        images = images.to(DEVICE)
        birads_logits, _ = model(images)
        preds = birads_logits.argmax(dim=1).cpu().numpy()

        for stem, gt_cls, pred_idx in zip(stems, cls_names, preds):
            results.append({
                "image_id": stem,
                "ground_truth": gt_cls,
                "predicted_birads": BIRADS_CLASSES[pred_idx],
            })

    return results


@torch.no_grad()
def run_feature_detection(model, data_loader):
    """Run feature detection on test set."""
    model.eval()
    results = []

    for images, stems, orig_sizes in tqdm(data_loader, desc="Detect features"):
        images = images.to(DEVICE)
        _, feature_outputs = model(images)

        for i, stem in enumerate(stems):
            orig_h, orig_w = orig_sizes[0][i].item(), orig_sizes[1][i].item()
            entry = {"image_id": stem}

            for feat_name in FEATURE_TYPES:
                cls_logits, bbox_pred = feature_outputs[feat_name]
                cls_idx = cls_logits[i].argmax().item()
                bbox = bbox_pred[i].cpu().numpy().tolist()
                bbox_denorm = _denormalise_bbox(bbox, orig_h, orig_w)

                names = FEATURE_CLASS_NAMES[feat_name]
                cls_name = names[cls_idx] if cls_idx < len(names) else str(cls_idx)
                entry[feat_name] = {
                    "class": cls_name,
                    "bbox": bbox_denorm,
                }

            results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference for breast ultrasound analysis")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Resolve checkpoint
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = MODEL_DIR / "best.pt"

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Train a model first, or specify --ckpt path/to/model.pt")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / "submission"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    model = create_model(backbone_name=args.backbone).to(DEVICE)
    epoch, best_metric = load_checkpoint(ckpt_path, model)
    print(f"Loaded checkpoint from epoch {epoch}, best metric: {best_metric:.4f}")

    # ---- Classification ----
    class_ds = TestClassDataset(TEST_CLASS)
    class_loader = DataLoader(class_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    class_results = run_classification(model, class_loader)

    with open(output_dir / "class_result.json", "w", encoding="utf-8") as f:
        json.dump(class_results, f, ensure_ascii=False, indent=2)
    print(f"Classification results saved to {output_dir / 'class_result.json'} ({len(class_results)} images)")

    # ---- Feature detection ----
    future_ds = TestFutureDataset(TEST_FUTURE)
    future_loader = DataLoader(future_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    future_results = run_feature_detection(model, future_loader)

    with open(output_dir / "future_result.json", "w", encoding="utf-8") as f:
        json.dump(future_results, f, ensure_ascii=False, indent=2)
    print(f"Feature detection results saved to {output_dir / 'future_result.json'} ({len(future_results)} images)")


if __name__ == "__main__":
    main()
