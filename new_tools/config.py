"""
Configuration for breast ultrasound BIRADS classification + feature detection.
"""
from pathlib import Path
import torch

# --- Paths ---
# ROOT = Path(__file__).resolve().parent.parent
# ALL_DATA = ROOT / "all_data"

# Feature detection data
FUTURE_TRAIN_IMAGES = "/root/autodl-tmp/ruxian_detect/future/train/images"
FUTURE_TRAIN_LABELS = {
    "boundary": "/root/autodl-tmp/ruxian_detect/future/train/boundary_labels",
    "calcification": "/root/autodl-tmp/ruxian_detect/future/train/calcification_labels",
    "shape": "/root/autodl-tmp/ruxian_detect/future/train/shape_labels",
    "direction": "/root/autodl-tmp/ruxian_detect/future/train/direction_labels",
}

# Classification data (BIRADS)
CLASSFY_TRAIN = "/root/autodl-tmp/ruxian_detect/classfy/train"

# Test data
TEST_CLASS = "/root/autodl-tmp/ruxian_detect/test_A/class_test/A"
TEST_FUTURE = "/root/autodl-tmp/ruxian_detect/test_A/future_test/A"

# Output
OUTPUT_DIR = Path("/root/autodl-tmp/ruxian_detect/new_tools/outputs")
MODEL_DIR = Path("/root/autodl-tmp/ruxian_detect/new_tools/outputs/models")
LOG_DIR = Path("/root/autodl-tmp/ruxian_detect/new_tools/outputs/logs")

# --- BIRADS categories ---
BIRADS_CLASSES = ["2类", "3类", "4A类", "4B类", "4C类", "5类"]
NUM_BIRADS = len(BIRADS_CLASSES)

# --- Feature types ---
FEATURE_TYPES = ["boundary", "calcification", "shape", "direction"]
FEATURE_CLASS_NAMES = {
    "boundary": ["smooth", "not_smooth"],
    "calcification": ["no_calcification", "calcification"],
    "shape": ["regular", "irregular"],
    "direction": ["parallel", "not_parallel"],
}
NUM_FEATURE_CLASSES = {k: len(v) for k, v in FEATURE_CLASS_NAMES.items()}

# --- Model ---
MODEL_NAME = "efficientnet_b3"
PRETRAINED = True
BACKBONE_LR_FACTOR = 0.1
IMAGE_SIZE = (640, 320)           # increased from 512×256 for better feature discrimination
DROPOUT = 0.5
DROPOUT2D = 0.20                  # per-head spatial dropout

# --- Training ---
BATCH_SIZE = 12                    # reduced to fit larger images in memory
NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 3e-3                # increased from 2e-3 for stronger regularization
LR_SCHEDULER = "cosine"
WARMUP_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 25

# Freeze backbone after this epoch (0 to disable)
BACKBONE_FREEZE_EPOCH = 0

# Label smoothing
LABEL_SMOOTHING = 0.1

# Loss weights
CLS_LOSS_WEIGHT = 1.0
DET_LOSS_WEIGHT = 1.0
BBOX_LOSS_WEIGHT = 2.0

# Per-feature task weights: up-weight direction and shape
FEATURE_TASK_WEIGHTS = {
    "boundary":      1.5,
    "calcification": 1.0,
    "shape":         2.0,
    "direction":     2.5,
}

# Per-feature Focal Loss gamma — higher γ suppresses easy negatives more aggressively.
# This directly combats the false-positive problem: direction/shape need higher γ.
FEATURE_FOCAL_GAMMA = {
    "boundary":      2.5,
    "calcification": 2.0,
    "shape":         3.0,
    "direction":     3.0,
}

# Neck type: "fpn" adds a lightweight multi-scale feature pyramid between backbone and heads.
# Helps detection heads see both coarse and fine spatial features. Set to "" to disable.
NECK_TYPE = "fpn"
NECK_CHANNELS = 256

# Augmentation intensity
AUG_HFLIP_PROB = 0.5
AUG_ROTATION_DEG = 20
AUG_SCALE_RANGE = (0.85, 1.15)
AUG_BRIGHTNESS = 0.3
AUG_ELASTIC_ALPHA = 40
AUG_ELASTIC_PROB = 0.3
AUG_MIXUP_ALPHA = 0.2
AUG_CUTMIX_ALPHA = 0.2

# --- Data ---
TRAIN_VAL_SPLIT = 0.85
NUM_WORKERS = 8
RANDOM_SEED = 42

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True
