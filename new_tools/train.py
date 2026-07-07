"""
Training script for the breast ultrasound multi-task model.

Usage:
    python train.py                          # train with default config
    python train.py --epochs 100 --lr 5e-4   # override hyperparams
    python train.py --backbone convnext_tiny # change backbone
"""
import argparse
import sys
import random as py_random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add parent to allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    AUG_CUTMIX_ALPHA,
    AUG_MIXUP_ALPHA,
    BACKBONE_FREEZE_EPOCH,
    BACKBONE_LR_FACTOR,
    BATCH_SIZE,
    CLS_LOSS_WEIGHT,
    DET_LOSS_WEIGHT,
    BBOX_LOSS_WEIGHT,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    FEATURE_FOCAL_GAMMA,
    FEATURE_TASK_WEIGHTS,
    FUTURE_TRAIN_IMAGES,
    FUTURE_TRAIN_LABELS,
    CLASSFY_TRAIN,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    LR_SCHEDULER,
    MODEL_DIR,
    LOG_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    RANDOM_SEED,
    USE_AMP,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from dataset import BreastUltrasoundDataset, collate_fn
from losses import MultiTaskLoss
from models.multitask import create_model
from utils import (
    AverageMeter,
    FeatureMetricsAccumulator,
    MetricsTracker,
    Logger,
    compute_accuracy,
    compute_iou,
    build_scheduler,
    save_checkpoint,
    load_checkpoint,
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    py_random.seed(seed)


def build_optimizer(model, lr, backbone_lr_factor, weight_decay):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW([
        {"params": head_params, "lr": lr},
        {"params": backbone_params, "lr": lr * backbone_lr_factor},
    ], weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# MixUp helpers
# ---------------------------------------------------------------------------

def mixup_batch(images, targets, alpha):
    """MixUp augmentation at batch level. Returns mixed images and (targets_a, targets_b, lam)."""
    if alpha <= 0 or images.size(0) <= 1:
        return images, targets, None

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # keep lam >= 0.5 for stability
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, targets, (index, lam)


def mixup_loss_fn(criterion, birads_logits, feature_outputs, targets, mixup_info):
    """Compute loss with MixUp mixing."""
    if mixup_info is None:
        return criterion(birads_logits, feature_outputs, targets)

    index, lam = mixup_info
    # Targets for shuffled samples
    targets_b = {}
    for k, v in targets.items():
        targets_b[k] = v[index]

    # Compute loss for both and mix
    loss_a, loss_dict_a = criterion(birads_logits, feature_outputs, targets)
    loss_b, loss_dict_b = criterion(birads_logits, feature_outputs, targets_b)

    loss = lam * loss_a + (1 - lam) * loss_b
    # Merge dicts (report weighted avg)
    loss_dict = {}
    for k in loss_dict_a:
        loss_dict[k] = lam * loss_dict_a[k] + (1 - lam) * loss_dict_b[k]
    return loss, loss_dict


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, logger):
    model.train()
    tracker = MetricsTracker()
    use_mixup = AUG_MIXUP_ALPHA > 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
    for images, targets, _ in pbar:
        images = images.to(DEVICE)
        for k in targets:
            targets[k] = targets[k].to(DEVICE)

        # MixUp
        mixup_info = None
        if use_mixup and py_random.random() < 0.5:
            images, targets, mixup_info = mixup_batch(images, targets, AUG_MIXUP_ALPHA)

        optimizer.zero_grad()

        if USE_AMP:
            with autocast():
                birads_logits, feature_outputs = model(images)
                if mixup_info is not None:
                    loss, loss_dict = mixup_loss_fn(criterion, birads_logits, feature_outputs, targets, mixup_info)
                else:
                    loss, loss_dict = criterion(birads_logits, feature_outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            birads_logits, feature_outputs = model(images)
            if mixup_info is not None:
                loss, loss_dict = mixup_loss_fn(criterion, birads_logits, feature_outputs, targets, mixup_info)
            else:
                loss, loss_dict = criterion(birads_logits, feature_outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        # Accuracy (only on samples with valid BIRADS labels, skip for mixup)
        if mixup_info is None and (targets["birads"] >= 0).any():
            valid = targets["birads"] >= 0
            acc1 = compute_accuracy(birads_logits[valid], targets["birads"][valid], topk=(1,))[0]
        elif mixup_info is not None:
            acc1 = 0.0
        else:
            acc1 = 0.0
        loss_dict["acc@1"] = acc1

        tracker.update(loss_dict, images.size(0))
        avgs = tracker.averages()
        pbar.set_postfix({k: f"{v:.4f}" for k, v in avgs.items() if k in ("cls", "det", "acc@1")})

    return tracker.averages()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    tracker = MetricsTracker()
    all_preds, all_labels = [], []
    feat_accum = FeatureMetricsAccumulator()

    for images, targets, _ in tqdm(loader, desc="Validation"):
        images = images.to(DEVICE)
        for k in targets:
            targets[k] = targets[k].to(DEVICE)

        birads_logits, feature_outputs = model(images)
        _, loss_dict = criterion(birads_logits, feature_outputs, targets)

        valid_mask = targets["birads"] >= 0
        if valid_mask.any():
            acc1 = compute_accuracy(birads_logits[valid_mask], targets["birads"][valid_mask], topk=(1,))[0]
        else:
            acc1 = 0.0
        loss_dict["acc@1"] = acc1

        tracker.update(loss_dict, images.size(0))

        all_preds.append(birads_logits[valid_mask].argmax(dim=1).cpu())
        all_labels.append(targets["birads"][valid_mask].cpu())

        # Accumulate per-feature metrics
        for feat_name, (cls_logits, bbox_pred) in feature_outputs.items():
            feat_accum.update(feat_name, cls_logits, bbox_pred, targets[feat_name])

    metrics = tracker.averages()
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics["birads_acc"] = (all_preds == all_labels).float().mean().item()

    # Compute epoch-level feature metrics
    feature_metrics = feat_accum.compute(iou_thresholds=(0.5,))
    metrics.update(feature_metrics)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = Logger(LOG_DIR)

    # ---- Data ----
    train_ds = BreastUltrasoundDataset(
        future_images_dir=FUTURE_TRAIN_IMAGES,
        future_labels_dir=FUTURE_TRAIN_LABELS,
        classfy_dir=CLASSFY_TRAIN,
        is_train=True,
    )
    val_ds = BreastUltrasoundDataset(
        future_images_dir=FUTURE_TRAIN_IMAGES,
        future_labels_dir=FUTURE_TRAIN_LABELS,
        classfy_dir=CLASSFY_TRAIN,
        is_train=False,
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        drop_last=True,  # avoid batch-size-1 issues in mixup
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    # ---- Model ----
    model = create_model(backbone_name=args.backbone).to(DEVICE)
    criterion = MultiTaskLoss(
        label_smoothing=LABEL_SMOOTHING,
        feature_task_weights=FEATURE_TASK_WEIGHTS,
        feature_focal_gamma=FEATURE_FOCAL_GAMMA,
    ).to(DEVICE)

    # ---- Optimizer ----
    optimizer = build_optimizer(model, args.lr, BACKBONE_LR_FACTOR, WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, args.epochs, WARMUP_EPOCHS, LR_SCHEDULER)
    scaler = GradScaler(enabled=USE_AMP)

    start_epoch = 1
    best_val_f1 = 0.0  # track best mean-F1 across features instead of BIRADS acc
    patience_counter = 0

    if args.resume:
        start_epoch, best_val_f1 = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, best f1: {best_val_f1:.4f}")

    best_model_path = MODEL_DIR / "best.pt"

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, logger)

        val_metrics = validate(model, val_loader, criterion)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)

        # Compute mean F1 across 4 features as the main metric
        f1_keys = ["boundary_f1", "calcification_f1", "shape_f1", "direction_f1"]
        mean_f1 = sum(val_metrics.get(k, 0) for k in f1_keys) / 4.0

        print(f"Epoch {epoch:3d} | LR {current_lr:.2e} | "
              f"Train cls={train_metrics.get('cls',0):.4f} det={train_metrics.get('det',0):.4f} "
              f"acc@1={train_metrics.get('acc@1',0):.4f} | "
              f"BIRADS acc={val_metrics.get('birads_acc',0):.4f} | "
              f"meanF1={mean_f1:.4f} | "
              f"P/R: "
              f"b={val_metrics.get('boundary_precision',0):.2f}/{val_metrics.get('boundary_recall',0):.2f} "
              f"c={val_metrics.get('calcification_precision',0):.2f}/{val_metrics.get('calcification_recall',0):.2f} "
              f"s={val_metrics.get('shape_precision',0):.2f}/{val_metrics.get('shape_recall',0):.2f} "
              f"d={val_metrics.get('direction_precision',0):.2f}/{val_metrics.get('direction_recall',0):.2f}")
        print(f"         F1: "
              f"b={val_metrics.get('boundary_f1',0):.3f} "
              f"c={val_metrics.get('calcification_f1',0):.3f} "
              f"s={val_metrics.get('shape_f1',0):.3f} "
              f"d={val_metrics.get('direction_f1',0):.3f} | "
              f"IoU: "
              f"b={val_metrics.get('boundary_iou',0):.3f} "
              f"c={val_metrics.get('calcification_iou',0):.3f} "
              f"s={val_metrics.get('shape_iou',0):.3f} "
              f"d={val_metrics.get('direction_iou',0):.3f}")

        # Save best (based on mean F1 across features)
        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_f1, best_model_path)
            print(f"  -> Saved best model (mean F1={best_val_f1:.4f})")
        else:
            patience_counter += 1

        # Step scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics.get("total", 0))
        else:
            scheduler.step()

        # Freeze backbone after BACKBONE_FREEZE_EPOCH
        if BACKBONE_FREEZE_EPOCH > 0 and epoch == BACKBONE_FREEZE_EPOCH:
            for name, param in model.named_parameters():
                if name.startswith("backbone"):
                    param.requires_grad = False
            print(f"  -> Backbone frozen at epoch {epoch}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Training complete. Best mean F1: {best_val_f1:.4f}")
    print(f"Model saved to {best_model_path}")


if __name__ == "__main__":
    main()
