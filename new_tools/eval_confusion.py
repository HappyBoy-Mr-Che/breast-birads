"""
Evaluate test-set results with confusion matrices.

Reads class_result.json and future_result.json from outputs/submission/,
compares against ground truth labels in test_A/, and produces:
  1. BI-RADS 6-class confusion matrix + per-class accuracy
  2. Per-feature 2-class confusion matrix + accuracy / precision / recall / F1
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Enable Chinese font rendering
_CN_FONT = "Microsoft YaHei"
for _f in fm.fontManager.ttflist:
    if _CN_FONT in _f.name:
        matplotlib.rcParams["font.sans-serif"] = [_CN_FONT, "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break

ROOT = Path(__file__).resolve().parent.parent

# ── paths ────────────────────────────────────────────────────────────────
SUBMISSION_DIR = ROOT / "new_tools" / "outputs" / "submission"
CLASS_RESULT   = SUBMISSION_DIR / "class_result.json"
FUTURE_RESULT  = SUBMISSION_DIR / "future_result.json"
FUTURE_GT_DIR  = ROOT / "test_A" / "future_test" / "A"
OUTPUT_DIR     = ROOT / "new_tools" / "outputs" / "logs"

BIRADS_CLASSES = ["2类", "3类", "4A类", "4B类", "4C类", "5类"]
FEATURE_TYPES  = ["boundary", "calcification", "shape", "direction"]
FEATURE_POS_CLASS = {
    "boundary":      "not_smooth",
    "calcification": "calcification",
    "shape":         "irregular",
    "direction":     "not_parallel",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. BI-RADS confusion matrix
# ═══════════════════════════════════════════════════════════════════════════

def build_birads_cm():
    with open(CLASS_RESULT, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = len(BIRADS_CLASSES)
    cm = np.zeros((n, n), dtype=int)
    cls_to_idx = {c: i for i, c in enumerate(BIRADS_CLASSES)}

    for item in data:
        gt = item["ground_truth"]
        pd = item["predicted_birads"]
        cm[cls_to_idx[gt], cls_to_idx[pd]] += 1

    acc_per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    total_acc = cm.diagonal().sum() / cm.sum()

    return cm, acc_per_class, total_acc


# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature confusion matrices
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_cm():
    with open(FUTURE_RESULT, "r", encoding="utf-8") as f:
        preds = json.load(f)

    # Build lookup: image_id → predicted class for each feature
    pred_lookup = {}
    for item in preds:
        img_id = item["image_id"]
        pred_lookup[img_id] = {
            feat: item[feat]["class"] for feat in FEATURE_TYPES
        }

    metrics = {}
    for feat in FEATURE_TYPES:
        cm = np.zeros((2, 2), dtype=int)
        label_dir = FUTURE_GT_DIR / f"{feat}_labels"
        pos_name = FEATURE_POS_CLASS[feat]

        for txt_path in label_dir.glob("*.txt"):
            stem = txt_path.stem
            if stem not in pred_lookup:
                continue

            # GT class from YOLO label (first token)
            raw = txt_path.read_text().strip()
            if not raw:
                gt_cls_id = 0
            else:
                gt_cls_id = int(raw.split()[0])

            gt_label = 1 if gt_cls_id == 1 else 0  # 1=positive, 0=negative

            # Predicted class
            pd_name = pred_lookup[stem][feat]
            pd_label = 1 if pd_name == pos_name else 0

            cm[gt_label, pd_label] += 1

        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]

        acc  = (tp + tn) / cm.sum() if cm.sum() else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0

        metrics[feat] = {
            "cm": cm,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1, "spec": spec,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "neg_name": "smooth" if feat == "boundary" else
                        "no_calcification" if feat == "calcification" else
                        "regular" if feat == "shape" else "parallel",
            "pos_name": pos_name,
        }

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 3. Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(ax, cm, row_labels, col_labels, title, cmap="Blues"):
    """Draw a styled confusion matrix on axis `ax`."""
    im = ax.imshow(cm, cmap=cmap, aspect="auto")
    n_rows, n_cols = cm.shape

    # values
    for i in range(n_rows):
        for j in range(n_cols):
            v = cm[i, j]
            color = "white" if v > cm.max() * 0.55 else "black"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)

    # per-row recall on the right
    row_sums = cm.sum(axis=1)
    for i in range(n_rows):
        r = cm[i, i] / row_sums[i] if row_sums[i] else 0
        ax.text(n_cols + 0.35, i, f"{r:.1%}", ha="center", va="center",
                fontsize=8, color="#333")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=12, fontweight="bold")

    return im


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── BI-RADS ──
    birads_cm, birads_acc, birads_total = build_birads_cm()

    print("=" * 60)
    print("  BI-RADS 6-Class Classification Results")
    print("=" * 60)
    for i, name in enumerate(BIRADS_CLASSES):
        n_correct = birads_cm[i, i]
        n_total   = birads_cm[i].sum()
        print(f"  {name:6s}  accuracy: {birads_acc[i]:.2%}  ({n_correct}/{n_total})")
    print(f"  {'Total':6s}  accuracy: {birads_total:.2%}")
    print()

    # ── Features ──
    feat_metrics = build_feature_cm()

    print("=" * 60)
    print("  Feature Detection Results (per-feature binary classification)")
    print("=" * 60)
    header = f"  {'Feature':<16s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Spec':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for feat in FEATURE_TYPES:
        m = feat_metrics[feat]
        print(f"  {feat:<16s} {m['acc']:>7.2%} {m['prec']:>7.2%} "
              f"{m['rec']:>7.2%} {m['f1']:>7.2%} {m['spec']:>7.2%}")
    print()

    # ── Plot ──
    fig = plt.figure(figsize=(18, 10))

    # ---- BI-RADS confusion matrix ----
    ax1 = fig.add_subplot(2, 3, (1, 2))
    plot_confusion_matrix(ax1, birads_cm, BIRADS_CLASSES, BIRADS_CLASSES,
                          f"BI-RADS Classification (Acc={birads_total:.2%})")
    ax1.text(6.5, 2.5, "Recall", rotation=90, fontsize=8, color="#666",
             va="center")

    # ---- Four feature confusion matrices ----
    positions = [(2, 3, 4), (2, 3, 5), (2, 3, 6)]
    # subplot layout: top row (1,2)->birads, bottom row 4 subplots
    subplot_map = [4, 5, 6]
    axes_2x2 = [fig.add_subplot(2, 3, idx) for idx in subplot_map]
    # boundary, calcification, shape, direction
    # boundary → first in bottom row
    axes_bottom = [fig.add_subplot(2, 3, 4),
                   fig.add_subplot(2, 3, 5),
                   fig.add_subplot(2, 3, 6)]

    # We have 4 features but only 3 bottom slots. Rearrange:
    # Use 2 rows: row1=birads(span 2 cols), row2=4 features in 4 cols
    fig.clear()
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1], hspace=0.35, wspace=0.35)

    # Row 1: BI-RADS (span left 2 cols) + text summary (span right 2 cols)
    ax_birads = fig.add_subplot(gs[0, :2])
    plot_confusion_matrix(ax_birads, birads_cm, BIRADS_CLASSES, BIRADS_CLASSES,
                          f"BI-RADS Classification (OA={birads_total:.2%})")

    # Right side: per-class accuracy bars
    ax_acc = fig.add_subplot(gs[0, 2:])
    colors = plt.cm.RdYlGn([(a - 0.3) / 0.7 for a in birads_acc])
    bars = ax_acc.barh(BIRADS_CLASSES[::-1], birads_acc[::-1], color=colors[::-1])
    ax_acc.set_xlim(0, 1)
    ax_acc.set_title("Per-Class Accuracy (Recall)", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, birads_acc[::-1]):
        ax_acc.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=10)
    ax_acc.axvline(birads_total, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax_acc.text(birads_total + 0.01, -0.5, f"OA={birads_total:.1%}", fontsize=8, color="gray")

    # Row 2: four feature confusion matrices
    for idx, feat in enumerate(FEATURE_TYPES):
        ax = fig.add_subplot(gs[1, idx])
        m = feat_metrics[feat]
        labels = [m["neg_name"], m["pos_name"]]
        plot_confusion_matrix(ax, m["cm"], labels, labels,
                              f"{feat}\nAcc={m['acc']:.1%} F1={m['f1']:.1%}",
                              cmap="Oranges")

    fig.suptitle("Test Set Evaluation — Confusion Matrices", fontsize=15,
                 fontweight="bold", y=1.01)

    save_path = OUTPUT_DIR / "confusion_matrices.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {save_path}")

    # ── Also save CSVs ──
    birads_path = OUTPUT_DIR / "birads_confusion.csv"
    np.savetxt(birads_path, birads_cm.astype(int), fmt="%d",
               delimiter=",", header=",".join(BIRADS_CLASSES))
    print(f"Saved → {birads_path}")

    for feat in FEATURE_TYPES:
        cm_path = OUTPUT_DIR / f"{feat}_confusion.csv"
        m = feat_metrics[feat]
        np.savetxt(cm_path, m["cm"].astype(int), fmt="%d", delimiter=",")
        print(f"Saved → {cm_path}")


if __name__ == "__main__":
    main()
