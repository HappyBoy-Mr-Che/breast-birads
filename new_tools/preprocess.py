"""
Data preprocessing and analysis utilities.

Usage:
    python preprocess.py           # Analyse dataset statistics
    python preprocess.py --clean   # Clean and validate dataset
    python preprocess.py --split   # Create train/val split files
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

def imread_unicode(path):
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


from config import (
    BIRADS_CLASSES,
    CLASSFY_TRAIN,
    FEATURE_TYPES,
    FUTURE_TRAIN_IMAGES,
    FUTURE_TRAIN_LABELS,
    RANDOM_SEED,
    TRAIN_VAL_SPLIT,
)


def analyse_dataset():
    """Print dataset statistics."""
    print("=" * 60)
    print("Dataset Analysis")
    print("=" * 60)

    # --- Feature detection data ---
    print("\n[Feature Detection Data]")
    img_dir = Path(FUTURE_TRAIN_IMAGES)
    stem_set = {p.stem for p in img_dir.glob("*.jpg")} | {p.stem for p in img_dir.glob("*.png")}
    all_stems = sorted(stem_set)
    print(f"  Total images: {len(all_stems)}")

    for feat_name in FEATURE_TYPES:
        label_dir = Path(FUTURE_TRAIN_LABELS[feat_name])
        label_stems = {p.stem for p in label_dir.glob("*.txt")}
        labeled = len(stem_set & label_stems)
        class_counts = Counter()
        for stem in sorted(stem_set & label_stems):
            with open(label_dir / f"{stem}.txt") as f:
                line = f.readline().strip()
                if not line:
                    continue
                cls = int(line.split()[0])
                class_counts[cls] += 1

        print(f"  {feat_name}: {labeled} labeled images")
        for cls, count in sorted(class_counts.items()):
            name = ["neg", "pos"][cls] if cls < 2 else f"class_{cls}"
            print(f"    class {cls} ({name}): {count} ({100*count/labeled:.1f}%)")

    # --- Classification data ---
    print("\n[BIRADS Classification Data]")
    classfy_root = Path(CLASSFY_TRAIN)
    birads_counts = {}
    for cls_name in BIRADS_CLASSES:
        img_dir = classfy_root / cls_name / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            birads_counts[cls_name] = count
            print(f"  {cls_name}: {count} images")

    total_cls = sum(birads_counts.values())
    print(f"  Total: {total_cls}")

    # --- Overlap ---
    print("\n[Overlap Analysis]")
    cls_stems = set()
    for cls_name in BIRADS_CLASSES:
        img_dir = classfy_root / cls_name / "images"
        if img_dir.exists():
            cls_stems |= {p.stem for p in img_dir.glob("*")}
    overlap = stem_set & cls_stems
    print(f"  Images in both datasets: {len(overlap)}")
    print(f"  Feature-only images: {len(stem_set - cls_stems)}")
    print(f"  Classify-only images: {len(cls_stems - stem_set)}")


def validate_dataset():
    """Check for corrupted images and inconsistent labels."""
    print("Validating dataset...")
    issues = []

    img_dir = Path(FUTURE_TRAIN_IMAGES)
    all_stems = sorted({p.stem for p in img_dir.glob("*.jpg")} | {p.stem for p in img_dir.glob("*.png")})

    for stem in tqdm(all_stems, desc="Checking images"):
        # Check image integrity
        img_path = None
        for ext in (".jpg", ".png"):
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            issues.append(f"Missing image: {stem}")
            continue

        img = imread_unicode(str(img_path))
        if img is None:
            issues.append(f"Corrupted image: {img_path}")
            continue

        # Check labels
        for feat_name in FEATURE_TYPES:
            label_path = Path(FUTURE_TRAIN_LABELS[feat_name]) / f"{stem}.txt"
            if label_path.exists():
                try:
                    with open(label_path) as f:
                        line = f.readline().strip()
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"Bad label format ({feat_name}): {label_path}")
                        continue
                    cls = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    if cls < 0:
                        issues.append(f"Negative class ({feat_name}): {label_path}")
                    if any(not (0 <= x <= 1) for x in bbox):
                        issues.append(f"Bbox out of range ({feat_name}): {label_path}")
                except Exception as e:
                    issues.append(f"Error reading ({feat_name}): {label_path} - {e}")

    if issues:
        print(f"\nFound {len(issues)} issues:")
        for issue in issues[:50]:
            print(f"  {issue}")
        if len(issues) > 50:
            print(f"  ... and {len(issues)-50} more")
    else:
        print("No issues found!")


def create_split_files():
    """Create JSON files listing train/val image stems for reproducibility."""
    img_dir = Path(FUTURE_TRAIN_IMAGES)
    all_stems = sorted({p.stem for p in img_dir.glob("*.jpg")} | {p.stem for p in img_dir.glob("*.png")})

    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.permutation(len(all_stems))
    split_n = int(len(all_stems) * TRAIN_VAL_SPLIT)

    train_stems = [all_stems[i] for i in indices[:split_n]]
    val_stems = [all_stems[i] for i in indices[split_n:]]

    split_dir = Path(__file__).resolve().parent / "outputs"
    split_dir.mkdir(parents=True, exist_ok=True)

    with open(split_dir / "train_stems.json", "w") as f:
        json.dump(train_stems, f)
    with open(split_dir / "val_stems.json", "w") as f:
        json.dump(val_stems, f)

    print(f"Split files created: {len(train_stems)} train / {len(val_stems)} val")


def main():
    parser = argparse.ArgumentParser(description="Preprocessing utilities")
    parser.add_argument("--clean", action="store_true", help="Validate dataset")
    parser.add_argument("--split", action="store_true", help="Create train/val split files")
    args = parser.parse_args()

    analyse_dataset()

    if args.clean:
        validate_dataset()

    if args.split:
        create_split_files()


if __name__ == "__main__":
    main()
