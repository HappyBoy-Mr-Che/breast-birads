"""
Visualize training metrics from training_log.json.

Usage:
    python visualize.py                          # default log path
    python visualize.py --log outputs/logs/training_log.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def plot_loss_curves(data, save_dir):
    """Total / cls / det loss curves."""
    epochs = [d["epoch"] for d in data]
    train_total = [d["train"]["total"] for d in data]
    val_total = [d["val"]["total"] for d in data]
    train_cls = [d["train"]["cls"] for d in data]
    val_cls = [d["val"]["cls"] for d in data]
    train_det = [d["train"]["det"] for d in data]
    val_det = [d["val"]["det"] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_total, "b-", label="Train")
    axes[0].plot(epochs, val_total, "r-", label="Val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_cls, "b-", label="Train")
    axes[1].plot(epochs, val_cls, "r-", label="Val")
    axes[1].set_title("BIRADS Classification Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, train_det, "b-", label="Train")
    axes[2].plot(epochs, val_det, "r-", label="Val")
    axes[2].set_title("Feature Detection Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / "loss_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_accuracy(data, save_dir):
    """BIRADS classification accuracy curve."""
    epochs = [d["epoch"] for d in data]
    train_acc = [d["train"].get("acc@1", 0) for d in data]
    val_acc = [d["val"].get("birads_acc", 0) for d in data]

    best_epoch = np.argmax(val_acc) + 1
    best_val = val_acc[best_epoch - 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_acc, "b-", label="Train Acc@1", alpha=0.7)
    ax.plot(epochs, val_acc, "r-", label="Val Acc", linewidth=2)
    ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5,
               label=f"Best epoch {best_epoch} ({best_val:.4f})")
    ax.set_title("BIRADS Classification Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = save_dir / "accuracy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_iou_curves(data, save_dir):
    """IoU curves for 4 feature detection heads."""
    epochs = [d["epoch"] for d in data]
    features = ["boundary", "calcification", "shape", "direction"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for feat, color in zip(features, colors):
        iou = [d["val"].get(f"{feat}_iou", 0) for d in data]
        ax.plot(epochs, iou, color=color, label=feat, linewidth=2)

    ax.set_title("Feature Detection IoU (positive samples only)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = save_dir / "iou_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_det_breakdown(data, save_dir):
    """Per-feature cls + bbox loss breakdown."""
    epochs = [d["epoch"] for d in data]
    features = ["boundary", "calcification", "shape", "direction"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        cls_key = f"{feat}_cls"
        bbox_key = f"{feat}_bbox"
        train_cls = [d["train"].get(cls_key, 0) for d in data]
        val_cls = [d["val"].get(cls_key, 0) for d in data]
        train_bbox = [d["train"].get(bbox_key, 0) for d in data]
        val_bbox = [d["val"].get(bbox_key, 0) for d in data]

        ax.plot(epochs, train_cls, "b-", alpha=0.5, label="Train cls")
        ax.plot(epochs, val_cls, "b-", label="Val cls")
        ax.plot(epochs, train_bbox, "r-", alpha=0.5, label="Train bbox")
        ax.plot(epochs, val_bbox, "r-", label="Val bbox")
        ax.set_title(feat.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Feature Detection Loss Breakdown", fontsize=14)
    plt.tight_layout()
    path = save_dir / "det_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_lr_curve(data, save_dir):
    """Learning rate schedule."""
    epochs = [d["epoch"] for d in data]
    lrs = [d["lr"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lrs, "g-", linewidth=2)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    path = save_dir / "lr_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_feature_classification_metrics(data, save_dir):
    """Per-feature precision, recall, F1 curves."""
    epochs = [d["epoch"] for d in data]
    features = ["boundary", "calcification", "shape", "direction"]
    metrics_per_feat = ["precision", "recall", "f1", "specificity"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_per_feat):
        ax = axes[i]
        for feat, color in zip(features, colors):
            vals = [d["val"].get(f"{feat}_{metric}", 0) for d in data]
            ax.plot(epochs, vals, color=color, label=feat, linewidth=2)
        ax.set_title(f"Feature {metric.capitalize()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle("Per-Feature Classification Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_dir / "feature_cls_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_feature_det_ap(data, save_dir):
    """Per-feature detection AP curves."""
    epochs = [d["epoch"] for d in data]
    features = ["boundary", "calcification", "shape", "direction"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for feat, color in zip(features, colors):
        vals = [d["val"].get(f"{feat}_det_ap@0.5", 0) for d in data]
        ax.plot(epochs, vals, color=color, label=feat, linewidth=2)

    ax.set_title("Feature Detection AP@0.5 (correct class + IoU>0.5)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP@0.5")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = save_dir / "detection_ap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_feature_cls_acc(data, save_dir):
    """Per-feature classification accuracy curves."""
    epochs = [d["epoch"] for d in data]
    features = ["boundary", "calcification", "shape", "direction"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for feat, color in zip(features, colors):
        vals = [d["val"].get(f"{feat}_cls_acc", 0) for d in data]
        ax.plot(epochs, vals, color=color, label=feat, linewidth=2)

    ax.set_title("Feature Classification Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = save_dir / "feature_cls_acc.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_summary(data, save_dir):
    """Combined summary dashboard."""
    epochs = [d["epoch"] for d in data]
    features = ["boundary", "calcification", "shape", "direction"]
    colors_iou = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig = plt.figure(figsize=(20, 16))

    # 1. Total Loss (top-left)
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(epochs, [d["train"]["total"] for d in data], "b-", label="Train")
    ax1.plot(epochs, [d["val"]["total"] for d in data], "r-", label="Val")
    ax1.set_title("Total Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. BIRADS Loss (top-center)
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(epochs, [d["train"]["cls"] for d in data], "b-", label="Train")
    ax2.plot(epochs, [d["val"]["cls"] for d in data], "r-", label="Val")
    ax2.set_title("BIRADS Classification Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Detection Loss (top-right)
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(epochs, [d["train"]["det"] for d in data], "b-", label="Train")
    ax3.plot(epochs, [d["val"]["det"] for d in data], "r-", label="Val")
    ax3.set_title("Feature Detection Loss")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Accuracy (middle-left)
    ax4 = fig.add_subplot(3, 3, 4)
    val_acc = [d["val"].get("birads_acc", 0) for d in data]
    train_acc = [d["train"].get("acc@1", 0) for d in data]
    best_epoch = np.argmax(val_acc) + 1
    ax4.plot(epochs, train_acc, "b-", alpha=0.5, label="Train Acc@1")
    ax4.plot(epochs, val_acc, "r-", linewidth=2, label="Val Acc")
    ax4.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5)
    ax4.set_title(f"BIRADS Accuracy (best epoch {best_epoch}: {max(val_acc):.4f})")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    # 5. IoU (middle-center)
    ax5 = fig.add_subplot(3, 3, 5)
    for feat, c in zip(features, colors_iou):
        iou = [d["val"].get(f"{feat}_iou", 0) for d in data]
        ax5.plot(epochs, iou, color=c, label=feat, linewidth=2)
    ax5.set_title("Feature Detection IoU")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)

    # 6. LR curve (middle-right)
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(epochs, [d["lr"] for d in data], "g-", linewidth=2)
    ax6.set_title("Learning Rate")
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale("log")

    # 7-10. Per-feature loss breakdown (bottom row spans all)
    for i, feat in enumerate(features):
        ax = fig.add_subplot(3, 4, 9 + i)
        cls_key, bbox_key = f"{feat}_cls", f"{feat}_bbox"
        ax.plot(epochs, [d["train"].get(cls_key, 0) for d in data], "b-", alpha=0.4, label="T cls")
        ax.plot(epochs, [d["val"].get(cls_key, 0) for d in data], "b-", label="V cls")
        ax.plot(epochs, [d["train"].get(bbox_key, 0) for d in data], "r-", alpha=0.4, label="T bbox")
        ax.plot(epochs, [d["val"].get(bbox_key, 0) for d in data], "r-", label="V bbox")
        ax.set_title(feat.capitalize())
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Summary Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = save_dir / "summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default=None,
                        help="Path to training_log.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
    else:
        log_path = Path(__file__).resolve().parent / "outputs" / "logs" / "training_log.json"

    if not log_path.exists():
        print(f"Log not found: {log_path}")
        print("Specify with: python visualize.py --log path/to/training_log.json")
        return

    save_dir = Path(args.output) if args.output else log_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_log(log_path)
    print(f"Loaded {len(data)} epochs from {log_path}")
    print(f"Best val acc: {max(d['val'].get('birads_acc',0) for d in data):.4f}")

    plot_loss_curves(data, save_dir)
    plot_accuracy(data, save_dir)
    plot_iou_curves(data, save_dir)
    plot_det_breakdown(data, save_dir)
    plot_lr_curve(data, save_dir)
    plot_feature_cls_acc(data, save_dir)
    plot_feature_classification_metrics(data, save_dir)
    plot_feature_det_ap(data, save_dir)
    plot_summary(data, save_dir)

    print(f"\nAll plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
