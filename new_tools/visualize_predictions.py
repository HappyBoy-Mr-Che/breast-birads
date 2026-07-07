"""
Visualize model predictions on test images.

Draws BI-RADS classification text on classification test images,
and bounding boxes on feature detection test images.

Usage:
    python new_tools/visualize_predictions.py --ckpt new_tools/outputs/models/best.pt

Output:
    new_tools/outputs/visualized/
        classification/       — images with predicted BI-RADS class overlay
        feature_detection/    — images with 4 feature bounding boxes
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — resolve from script location, NOT from config.py (Linux paths)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    BIRADS_CLASSES,
    DEVICE,
    FEATURE_TYPES,
    FEATURE_CLASS_NAMES,
    IMAGE_SIZE,
)

# Local Windows paths (override config.py's hardcoded /root/autodl-tmp paths)
TEST_CLASS = PROJECT_ROOT / "test_A" / "class_test" / "A"
TEST_FUTURE = PROJECT_ROOT / "test_A" / "future_test" / "A"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

from models.multitask import create_model
from utils import load_checkpoint

# Optional PIL import for Chinese text rendering
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Colours for the 4 feature types (BGR order for cv2)
# ---------------------------------------------------------------------------
FEATURE_COLORS = {
    "boundary":      (0, 255, 0),    # green
    "calcification": (255, 0, 0),    # blue
    "shape":         (0, 165, 255),  # orange
    "direction":     (0, 0, 255),    # red
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def imread_unicode(path):
    """Read image with Unicode path support. Returns BGR numpy array."""
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img  # BGR


def _get_chinese_font(size):
    """Return a PIL ImageFont that supports Chinese characters, or default."""
    if not HAS_PIL:
        return None
    # Common Windows Chinese font paths
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",    # Microsoft YaHei
        "C:/Windows/Fonts/simhei.ttf",  # SimHei
        "C:/Windows/Fonts/simsun.ttc",  # SimSun
        "C:/Windows/Fonts/msyhbd.ttc",  # Microsoft YaHei Bold
    ]
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------

def draw_classification(img_bgr, predicted_class, ground_truth):
    """
    Draw predicted BI-RADS class banner on a classification image.
    Green banner = correct, red banner = wrong.
    Uses PIL for Chinese text; falls back to English abbreviation if PIL unavailable.
    """
    h, w = img_bgr.shape[:2]
    is_correct = (predicted_class == ground_truth)
    banner_color = (0, 180, 0) if is_correct else (0, 0, 200)  # BGR

    # --- Semi-transparent banner at top ---
    banner_h = max(50, int(h * 0.08))
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
    img_bgr = cv2.addWeighted(img_bgr, 0.55, overlay, 0.45, 0)

    # --- Text via PIL (Chinese-capable) ---
    font_size = max(16, int(min(w, h) / 22))
    font = _get_chinese_font(font_size)

    if font is not None:
        # PIL drawing path
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw_ctx = ImageDraw.Draw(pil_img)

        text = f"预测: {predicted_class}    真实: {ground_truth}"
        text_color = (255, 255, 255)  # white in RGB

        # Center text
        bbox = draw_ctx.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (w - text_w) // 2
        text_y = (banner_h - text_h) // 2

        draw_ctx.text((text_x, text_y), text, font=font, fill=text_color)

        # Status icon (top-left)
        icon = "✓" if is_correct else "✗"
        icon_color = (0, 255, 0) if is_correct else (255, 50, 50)
        draw_ctx.text((8, (banner_h - font_size) // 2), icon, font=font, fill=icon_color)

        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        # Fallback: ASCII-only cv2 text
        text = f"Pred: {predicted_class}  |  GT: {ground_truth}"
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w, h) / 900
        thickness = max(1, int(font_scale * 2))
        text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = int(banner_h * 0.65)
        cv2.putText(img_bgr, text, (text_x, text_y), font_face, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        icon = "OK" if is_correct else "XX"
        icon_color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.putText(img_bgr, icon, (10, int(banner_h * 0.7)), font_face,
                    font_scale * 1.2, icon_color, thickness + 1, cv2.LINE_AA)

    return img_bgr


def draw_feature_boxes(img_bgr, feature_predictions):
    """
    Draw all 4 predicted feature bounding boxes + class labels.
    Feature names and class names are ASCII — cv2.putText is sufficient.
    """
    h, w = img_bgr.shape[:2]
    line_thickness = max(2, int(min(w, h) / 300))
    font_scale = min(w, h) / 1100
    text_thickness = max(1, int(font_scale * 1.5))

    # Draw a legend in the top-left corner
    legend_x, legend_y = 8, 8
    legend_row_h = max(18, int(h * 0.035))
    cv2.rectangle(img_bgr, (legend_x - 2, legend_y - 2),
                  (legend_x + 200, legend_y + len(FEATURE_TYPES) * legend_row_h + 4),
                  (40, 40, 40), -1)

    for i, feat_name in enumerate(FEATURE_TYPES):
        pred = feature_predictions[feat_name]
        cls_name = pred["class"]
        bbox = pred["bbox"]  # [xc, yc, w, h] in original pixel coords
        color = FEATURE_COLORS[feat_name]

        # --- Bounding box ---
        xc, yc, bw, bh = bbox
        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)

        # Clamp to image
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, line_thickness)

        # --- Corner markers for visibility ---
        corner_len = max(8, int(min(x2 - x1, y2 - y1) * 0.2))
        cv2.line(img_bgr, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2)
        cv2.line(img_bgr, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2)
        cv2.line(img_bgr, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 2)
        cv2.line(img_bgr, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 2)
        cv2.line(img_bgr, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), 2)
        cv2.line(img_bgr, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), 2)
        cv2.line(img_bgr, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 2)
        cv2.line(img_bgr, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 2)

        # --- Label near the box ---
        label = f"{feat_name}: {cls_name}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale, text_thickness)
        label_y = y1 - 6
        if label_y - th < 0:
            label_y = y2 + th + 6

        # Label background
        cv2.rectangle(img_bgr, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 4),
                      color, -1)
        cv2.putText(img_bgr, label, (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), text_thickness, cv2.LINE_AA)

        # --- Legend entry ---
        ly = legend_y + i * legend_row_h + legend_row_h - 4
        cv2.rectangle(img_bgr, (legend_x, ly - 10), (legend_x + 14, ly + 2), color, -1)
        cv2.putText(img_bgr, f"{feat_name}: {cls_name}",
                    (legend_x + 20, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    return img_bgr


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(img_bgr):
    """Resize and normalise a BGR image for model input. Returns (1, 3, H, W) tensor."""
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]),
                       interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


# ---------------------------------------------------------------------------
# Main visualisation loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_classification(model, output_dir):
    """Run classification inference and save annotated images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect samples
    samples = []
    for cls_name in BIRADS_CLASSES:
        cls_img_dir = TEST_CLASS / cls_name / "images"
        if not cls_img_dir.exists():
            print(f"  [WARN] Directory not found: {cls_img_dir}")
            continue
        for p in cls_img_dir.glob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                samples.append((p, cls_name))

    correct = 0
    for img_path, gt_class in tqdm(samples, desc="Classification"):
        img_bgr = imread_unicode(str(img_path))
        tensor = preprocess(img_bgr).to(DEVICE)

        birads_logits, _ = model(tensor)
        pred_idx = birads_logits.argmax(dim=1).item()
        pred_class = BIRADS_CLASSES[pred_idx]

        if pred_class == gt_class:
            correct += 1

        img_bgr = draw_classification(img_bgr, pred_class, gt_class)
        out_path = output_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img_bgr)

    acc = correct / max(len(samples), 1)
    print(f"Classification: {correct}/{len(samples)} correct ({acc:.2%})")
    print(f"Saved to {output_dir}")


@torch.no_grad()
def visualize_feature_detection(model, output_dir):
    """Run feature detection inference and save annotated images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = TEST_FUTURE / "images"
    if not images_dir.exists():
        print(f"  [ERROR] Directory not found: {images_dir}")
        return

    samples = [p for p in images_dir.glob("*")
               if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]

    for img_path in tqdm(samples, desc="Feature detection"):
        img_bgr = imread_unicode(str(img_path))
        orig_h, orig_w = img_bgr.shape[:2]
        tensor = preprocess(img_bgr).to(DEVICE)

        _, feature_outputs = model(tensor)

        predictions = {}
        for feat_name in FEATURE_TYPES:
            cls_logits, bbox_pred = feature_outputs[feat_name]
            cls_idx = cls_logits[0].argmax().item()
            bbox_norm = bbox_pred[0].cpu().numpy().tolist()  # [xc, yc, w, h] normalised

            # Denormalise to original pixel coordinates
            xc, yc, bw, bh = bbox_norm
            bbox_px = [xc * orig_w, yc * orig_h, bw * orig_w, bh * orig_h]

            cls_name = FEATURE_CLASS_NAMES[feat_name][cls_idx]
            predictions[feat_name] = {"class": cls_name, "bbox": bbox_px}

        img_bgr = draw_feature_boxes(img_bgr, predictions)
        out_path = output_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img_bgr)

    print(f"Saved {len(samples)} images to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise model predictions on test images")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Backbone override")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output root directory for annotated images")
    parser.add_argument("--skip_classification", action="store_true",
                        help="Skip classification test images")
    parser.add_argument("--skip_detection", action="store_true",
                        help="Skip feature detection test images")
    args = parser.parse_args()

    # --- Resolve checkpoint ---
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = MODEL_DIR / "best.pt"

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Train a model first, or specify --ckpt path/to/model.pt")
        sys.exit(1)

    # --- Output directory ---
    vis_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / "visualized"

    # --- Load model (legacy architecture — matches trained checkpoint) ---
    print(f"Loading model from {ckpt_path} ...")
    model = create_model(
        backbone_name=args.backbone,
        neck_type="",                      # no FPN neck
        classification_hidden_dims=[512],  # 2-layer classifier: 1536→512→6
        feature_cls_hidden_dim=0,          # cls head: direct Linear(128→2)
        feature_bbox_hidden_dim=64,        # bbox head: Linear(128→64→4)
    ).to(DEVICE)
    epoch, best_metric = load_checkpoint(ckpt_path, model)
    print(f"Checkpoint: epoch {epoch}, best metric {best_metric:.4f}")
    model.eval()

    # --- Classification ---
    if not args.skip_classification:
        print("\n=== Classification test images ===")
        visualize_classification(model, vis_dir / "classification")

    # --- Feature detection ---
    if not args.skip_detection:
        print("\n=== Feature detection test images ===")
        visualize_feature_detection(model, vis_dir / "feature_detection")

    print("\nDone!")


if __name__ == "__main__":
    main()
