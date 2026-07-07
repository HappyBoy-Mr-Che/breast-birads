"""
Loss functions for multi-task training.

Key improvements:
  - Per-feature Focal Loss gamma (higher γ for shape/direction to suppress false positives)
  - sqrt(1/freq) class weights instead of raw 1/freq (less aggressive on minority classes)
  - Feature-specific task weighting
  - Uncertainty weighting (Kendall et al. 2018) for cls-vs-det balance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss with per-sample class weights and label smoothing."""

    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        n, c = inputs.shape

        with torch.no_grad():
            if self.label_smoothing > 0:
                smooth = self.label_smoothing / (c - 1)
                one_hot = torch.full_like(inputs, smooth)
                one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            else:
                one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1.0)

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        if self.class_weights is not None:
            w = self.class_weights.to(inputs.device)[targets]
        else:
            w = 1.0

        focal = self.alpha * (1 - probs) ** self.gamma
        loss = -(one_hot * log_probs).sum(dim=1) * focal.mean(dim=1) * w
        return loss.mean()


class DetectionLoss(nn.Module):
    """Per-feature detection loss: FocalLoss for class + SmoothL1 for bbox."""

    def __init__(self, bbox_weight=2.0, class_weights=None, focal_gamma=2.0):
        super().__init__()
        self.cls_loss_fn = FocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=0.0,
        )
        self.bbox_loss_fn = nn.SmoothL1Loss(reduction="none")
        self.bbox_weight = bbox_weight

    def forward(self, cls_logits, bbox_pred, targets):
        """targets: (B,5) — [cls_id, xc, yc, w, h]"""
        gt_cls = targets[:, 0].long()
        gt_bbox = targets[:, 1:5]
        positive_mask = gt_cls > 0

        cls_loss = self.cls_loss_fn(cls_logits, gt_cls)

        if positive_mask.any():
            bbox_loss = self.bbox_loss_fn(
                bbox_pred[positive_mask], gt_bbox[positive_mask]
            ).sum(dim=1).mean()
        else:
            bbox_loss = torch.tensor(0.0, device=bbox_pred.device)

        return cls_loss, bbox_loss * self.bbox_weight


# ---------------------------------------------------------------------------
# Dataset statistics (from prior analysis):
#   boundary:      neg=2817 (75.8%)  pos=687  (18.5%)  → ratio 4.1:1
#   calcification: neg=1764 (47.5%)  pos=1798 (48.4%)  → ratio ~1:1
#   shape:         neg=3047 (82.0%)  pos=605  (16.3%)  → ratio 5.0:1
#   direction:     neg=2927 (78.8%)  pos=616  (16.6%)  → ratio 4.75:1
#
# Using sqrt(1/freq) instead of raw 1/freq:
#   - Minority class gets a boost, but not so extreme that it causes over-prediction
#   - The precision drop was caused by weights of 5.0 making the model prefer "positive"
#     even when uncertain — sqrt balancing fixes this.
# ---------------------------------------------------------------------------

FEATURE_CLASS_WEIGHTS = {
    "boundary":      torch.tensor([1.15, 2.33]),
    "calcification": torch.tensor([1.02, 0.98]),
    "shape":         torch.tensor([1.10, 2.48]),
    "direction":     torch.tensor([1.10, 2.40]),
}

BIRADS_CLASS_WEIGHTS = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0, 1.0])


class MultiTaskLoss(nn.Module):
    """Combined loss with per-feature focal gamma + uncertainty weighting."""

    def __init__(self, cls_weight=1.0, det_weight=1.0, bbox_weight=2.0,
                 label_smoothing=0.05, feature_task_weights=None,
                 feature_focal_gamma=None):
        super().__init__()
        self.cls_loss_fn = FocalLoss(
            class_weights=BIRADS_CLASS_WEIGHTS,
            label_smoothing=label_smoothing,
        )

        if feature_focal_gamma is None:
            feature_focal_gamma = {f: 2.0 for f in ["boundary", "calcification", "shape", "direction"]}

        self.det_loss_fns = nn.ModuleDict({
            feat: DetectionLoss(
                bbox_weight=bbox_weight,
                class_weights=FEATURE_CLASS_WEIGHTS[feat],
                focal_gamma=feature_focal_gamma.get(feat, 2.0),
            )
            for feat in ["boundary", "calcification", "shape", "direction"]
        })

        if feature_task_weights is None:
            feature_task_weights = {f: 1.0 for f in ["boundary", "calcification", "shape", "direction"]}
        self.feature_task_weights = feature_task_weights

        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_det = nn.Parameter(torch.zeros(1))

    def forward(self, birads_logits, feature_outputs, targets):
        birads_targets = targets["birads"]
        valid_mask = birads_targets >= 0
        if valid_mask.any():
            cls_loss = self.cls_loss_fn(
                birads_logits[valid_mask], birads_targets[valid_mask]
            )
        else:
            cls_loss = torch.tensor(0.0, device=birads_logits.device)

        precision_cls = 1.0 / (2 * torch.exp(self.log_var_cls) + 1e-8)
        weighted_cls = precision_cls * cls_loss + self.log_var_cls

        total_det_loss = 0.0
        det_losses = {}
        for feat_name, det_fn in self.det_loss_fns.items():
            cls_logits, bbox_pred = feature_outputs[feat_name]
            cls_l, bbox_l = det_fn(cls_logits, bbox_pred, targets[feat_name])
            w = self.feature_task_weights.get(feat_name, 1.0)
            l = w * (cls_l + bbox_l)
            det_losses[feat_name] = {"cls": cls_l.item(), "bbox": bbox_l.item()}
            total_det_loss += l

        precision_det = 1.0 / (2 * torch.exp(self.log_var_det) + 1e-8)
        weighted_det = precision_det * total_det_loss + self.log_var_det

        total = weighted_cls + weighted_det

        loss_dict = {
            "total": total.item(),
            "cls": cls_loss.item(),
            "det": total_det_loss.item(),
        }
        loss_dict.update({f"{k}_cls": v["cls"] for k, v in det_losses.items()})
        loss_dict.update({f"{k}_bbox": v["bbox"] for k, v in det_losses.items()})

        return total, loss_dict
