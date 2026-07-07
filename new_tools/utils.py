"""
Training utilities: metrics, averaging, checkpointing, logging.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


class MetricsTracker:
    """Tracks and prints training/validation metrics."""

    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, metrics: dict, batch_size: int = 1):
        for k, v in metrics.items():
            self.meters[k].update(v, batch_size)

    def averages(self) -> dict:
        return {k: m.avg for k, m in self.meters.items()}

    def reset(self):
        self.meters.clear()


def compute_accuracy(logits, targets, topk=(1,)):
    """Top-k accuracy for classification."""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item() / batch_size)
    return res


def compute_iou(pred_bbox, gt_bbox):
    """IoU between two normalised [xc, yc, w, h] boxes, batched."""
    # Convert to [x1, y1, x2, y2]
    def to_corners(b):
        xc, yc, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x1, y1 = xc - w / 2, yc - h / 2
        x2, y2 = xc + w / 2, yc + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    pred = to_corners(pred_bbox)
    gt = to_corners(gt_bbox)

    ix1 = torch.max(pred[:, 0], gt[:, 0])
    iy1 = torch.max(pred[:, 1], gt[:, 1])
    ix2 = torch.min(pred[:, 2], gt[:, 2])
    iy2 = torch.min(pred[:, 3], gt[:, 3])

    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = area_pred + area_gt - inter + 1e-6

    return (inter / union).mean().item()


# ---------------------------------------------------------------------------
# Feature detection evaluation metrics
# ---------------------------------------------------------------------------

def compute_feature_metrics(cls_logits, bbox_pred, targets, feat_name, iou_thresholds=(0.5,)):
    """
    Compute per-feature classification + detection metrics.

    Args:
        cls_logits:  (B, 2) class logits
        bbox_pred:   (B, 4) predicted bbox
        targets:     (B, 5) [cls_id, xc, yc, w, h]
        feat_name:   str, for logging
        iou_thresholds: tuple of IoU thresholds for AP-like metric

    Returns:
        dict with keys:
            {feat}_cls_acc, {feat}_cls_precision, {feat}_cls_recall, {feat}_cls_f1,
            {feat}_iou, {feat}_det_ap@t (per threshold),
            {feat}_pos_det_rate, {feat}_neg_det_rate
    """
    gt_cls = targets[:, 0].long()
    gt_bbox = targets[:, 1:5]
    pred_cls = cls_logits.argmax(dim=1)
    pos_mask = gt_cls > 0
    neg_mask = gt_cls == 0

    metrics = {}

    # -- Classification accuracy --
    cls_correct = (pred_cls == gt_cls).float()
    metrics[f"{feat_name}_cls_acc"] = cls_correct.mean().item()

    # -- Per-class precision / recall / F1 --
    # Positive = class 1 (feature present), Negative = class 0 (feature absent)
    tp = ((pred_cls == 1) & (gt_cls == 1)).float().sum().item()
    fp = ((pred_cls == 1) & (gt_cls == 0)).float().sum().item()
    fn = ((pred_cls == 0) & (gt_cls == 1)).float().sum().item()
    tn = ((pred_cls == 0) & (gt_cls == 0)).float().sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    specificity = tn / max(tn + fp, 1)

    metrics[f"{feat_name}_precision"] = precision
    metrics[f"{feat_name}_recall"] = recall
    metrics[f"{feat_name}_f1"] = f1
    metrics[f"{feat_name}_specificity"] = specificity

    # -- IoU on positive samples --
    if pos_mask.any():
        iou = compute_iou(bbox_pred[pos_mask], gt_bbox[pos_mask])
    else:
        iou = 0.0
    metrics[f"{feat_name}_iou"] = iou

    # -- Detection rate (correct class AND IoU > threshold) --
    if pos_mask.any():
        ious_pos = _compute_iou_per_sample(bbox_pred[pos_mask], gt_bbox[pos_mask])
        for t in iou_thresholds:
            det_ok = (pred_cls[pos_mask] == 1) & (ious_pos > t)
            metrics[f"{feat_name}_det_ap@{t}"] = det_ok.float().mean().item()

    # -- Positive / negative detection rates --
    if pos_mask.any():
        metrics[f"{feat_name}_pos_det_rate"] = (pred_cls[pos_mask] == 1).float().mean().item()
    if neg_mask.any():
        metrics[f"{feat_name}_neg_det_rate"] = (pred_cls[neg_mask] == 0).float().mean().item()

    return metrics


def _compute_iou_per_sample(pred_bbox, gt_bbox):
    """Compute per-sample IoU for batched [xc,yc,w,h] boxes."""
    def to_corners(b):
        xc, yc, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x1, y1 = xc - w / 2, yc - h / 2
        x2, y2 = xc + w / 2, yc + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    pred_c = to_corners(pred_bbox)
    gt_c = to_corners(gt_bbox)

    ix1 = torch.max(pred_c[:, 0], gt_c[:, 0])
    iy1 = torch.max(pred_c[:, 1], gt_c[:, 1])
    ix2 = torch.min(pred_c[:, 2], gt_c[:, 2])
    iy2 = torch.min(pred_c[:, 3], gt_c[:, 3])

    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    area_pred = (pred_c[:, 2] - pred_c[:, 0]) * (pred_c[:, 3] - pred_c[:, 1])
    area_gt = (gt_c[:, 2] - gt_c[:, 0]) * (gt_c[:, 3] - gt_c[:, 1])
    union = area_pred + area_gt - inter + 1e-6

    return inter / union


class FeatureMetricsAccumulator:
    """Accumulate per-feature predictions across batches for epoch-level metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = {}  # feat_name -> {"pred_cls": [], "gt_cls": [], "iou": [], "pos_mask": []}

    def update(self, feat_name, cls_logits, bbox_pred, targets):
        gt_cls = targets[:, 0].long().cpu()
        gt_bbox = targets[:, 1:5].cpu()
        pred_cls = cls_logits.argmax(dim=1).cpu()
        bbox_pred = bbox_pred.cpu()
        pos_mask = (gt_cls > 0).cpu()

        if feat_name not in self.data:
            self.data[feat_name] = {"pred_cls": [], "gt_cls": [], "ious": [], "pos_mask": []}

        self.data[feat_name]["pred_cls"].append(pred_cls)
        self.data[feat_name]["gt_cls"].append(gt_cls)
        self.data[feat_name]["pos_mask"].append(pos_mask)

        if pos_mask.any():
            ious = _compute_iou_per_sample(bbox_pred[pos_mask], gt_bbox[pos_mask])
            self.data[feat_name]["ious"].append(ious)

    def compute(self, iou_thresholds=(0.5,)) -> dict:
        """Compute epoch-level metrics from accumulated data."""
        metrics = {}
        for feat_name, d in self.data.items():
            all_pred = torch.cat(d["pred_cls"])
            all_gt = torch.cat(d["gt_cls"])
            all_pos = torch.cat(d["pos_mask"])
            all_ious = torch.cat(d["ious"]) if d["ious"] else torch.tensor([])

            cls_acc = (all_pred == all_gt).float().mean().item()
            tp = ((all_pred == 1) & (all_gt == 1)).float().sum().item()
            fp = ((all_pred == 1) & (all_gt == 0)).float().sum().item()
            fn = ((all_pred == 0) & (all_gt == 1)).float().sum().item()
            tn = ((all_pred == 0) & (all_gt == 0)).float().sum().item()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            specificity = tn / max(tn + fp, 1)

            metrics[f"{feat_name}_cls_acc"] = cls_acc
            metrics[f"{feat_name}_precision"] = precision
            metrics[f"{feat_name}_recall"] = recall
            metrics[f"{feat_name}_f1"] = f1
            metrics[f"{feat_name}_specificity"] = specificity
            metrics[f"{feat_name}_iou"] = all_ious.mean().item() if len(all_ious) > 0 else 0.0

            # Detection AP at each IoU threshold
            pos_preds = all_pred[all_pos]
            if len(all_ious) > 0:
                for t in iou_thresholds:
                    det_ok = (pos_preds == 1) & (all_ious > t)
                    metrics[f"{feat_name}_det_ap@{t}"] = det_ok.float().mean().item() if len(det_ok) > 0 else 0.0

            # Positive/negative detection rates
            pos_mask_sum = all_pos.sum().item()
            neg_mask = all_gt == 0
            if pos_mask_sum > 0:
                metrics[f"{feat_name}_pos_recall"] = (all_pred[all_pos] == 1).float().mean().item()
            if neg_mask.sum().item() > 0:
                metrics[f"{feat_name}_neg_specificity"] = (all_pred[neg_mask] == 0).float().mean().item()

        return metrics


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["best_metric"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_log.json"
        self.entries = []

    def log(self, entry: dict):
        entry["timestamp"] = time.time()
        self.entries.append(entry)
        with open(self.log_file, "w") as f:
            json.dump(self.entries, f, indent=2)

    def log_epoch(self, epoch, train_metrics, val_metrics, lr):
        self.log({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": lr,
        })


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, num_epochs, warmup_epochs=5, sched_type="cosine"):
    if sched_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    elif sched_type == "plateau":
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    else:
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if warmup_epochs > 0:
        def warmup_fn(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, main_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = main_scheduler

    return scheduler
