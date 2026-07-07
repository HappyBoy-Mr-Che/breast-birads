"""
Multi-task model: shared backbone + optional FPN neck + BIRADS classification head
+ 4 feature-detection heads.

Each feature head predicts:
  - class logits  (2 classes for each feature: e.g. smooth / not-smooth)
  - bbox regression (4 values: xc, yc, w, h in normalised coordinates)
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from config import (
    DROPOUT,
    DROPOUT2D,
    FEATURE_TYPES,
    MODEL_NAME,
    NECK_CHANNELS,
    NECK_TYPE,
    NUM_BIRADS,
    NUM_FEATURE_CLASSES,
    PRETRAINED,
)


def _get_backbone(name: str, pretrained: bool):
    """Return a backbone CNN + its output channel count + intermediate features for FPN."""
    if name == "resnet50":
        model = tv_models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        out_channels = 2048
        # Return layer1..layer4 outputs for FPN
        backbone = nn.ModuleDict({
            "stem": nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
        })
        fpn_channels = [256, 512, 1024, 2048]
    elif name == "efficientnet_b3":
        model = tv_models.efficientnet_b3(weights="IMAGENET1K_V1" if pretrained else None)
        out_channels = model.features[-1][0].out_channels  # 1536
        backbone = model.features
        fpn_channels = None  # single-scale output
    elif name == "convnext_tiny":
        model = tv_models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        out_channels = model.features[-1][-1].block[5].out_channels  # 768
        backbone = model.features
        fpn_channels = None
    elif name == "mobilenet_v3_large":
        model = tv_models.mobilenet_v3_large(weights="IMAGENET1K_V1" if pretrained else None)
        out_channels = model.features[-1][-1].out_channels  # 960
        backbone = model.features
        fpn_channels = None
    elif name == "resnet34":
        model = tv_models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        out_channels = 512
        backbone = nn.ModuleDict({
            "stem": nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
        })
        fpn_channels = [64, 128, 256, 512]
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return backbone, out_channels, fpn_channels


# ---------------------------------------------------------------------------
# Lightweight FPN neck
# ---------------------------------------------------------------------------

class SimpleNeck(nn.Module):
    """
    Lightweight multi-scale feature refinement neck.

    Takes a single feature map from the backbone, processes it at 3 scales
    (original, 0.5x, 2x via conv stride / interpolate), then fuses back.
    Gives each detection head access to richer spatial context.
    """

    def __init__(self, in_channels: int, neck_channels: int = 256):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, neck_channels, 1),
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(inplace=True),
        )
        # Multi-scale branches
        self.scale_orig = nn.Sequential(
            nn.Conv2d(neck_channels, neck_channels, 3, padding=1),
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(inplace=True),
        )
        self.scale_down = nn.Sequential(
            nn.Conv2d(neck_channels, neck_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(inplace=True),
        )
        self.scale_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(neck_channels, neck_channels, 3, padding=1),
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(inplace=True),
        )
        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(neck_channels * 3, neck_channels, 1),
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = neck_channels

    def forward(self, x):
        x = self.compress(x)
        f_orig = self.scale_orig(x)
        f_down = F.interpolate(self.scale_down(x), size=f_orig.shape[2:], mode="bilinear", align_corners=False)
        f_up = F.interpolate(self.scale_up(x), size=f_orig.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([f_orig, f_down, f_up], dim=1))


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """BIRADS 6-class classifier.

    Args:
        in_channels:  input feature channels.
        num_classes:  number of BIRADS classes.
        dropout:      dropout rate.
        hidden_dims:  list of hidden layer sizes. Default [512, 256] gives
                       in→512→256→num_classes.  Pass [512] for legacy
                       2-layer classifier (in→512→num_classes).
    """

    def __init__(self, in_channels: int, num_classes: int = NUM_BIRADS,
                 dropout: float = DROPOUT, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.pool = nn.AdaptiveAvgPool2d(1)

        layers = [nn.Flatten(), nn.Dropout(dropout)]
        prev_dim = in_channels
        for i, hd in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU(inplace=True))
            # progressively lower dropout in deeper layers
            d = dropout * (0.6 ** i)
            layers.append(nn.Dropout(d))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)


def _make_mlp(in_features, out_features, hidden_dim, dropout):
    """Build a small MLP: in → hidden → out, or in → out when hidden_dim=0."""
    if hidden_dim and hidden_dim > 0:
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, out_features),
        )
    return nn.Linear(in_features, out_features)


class FeatureDetectionHead(nn.Module):
    """Detection head with stronger regularization to combat false positives.

    Args:
        in_channels:    input feature channels.
        num_classes:    number of classes for this feature (usually 2).
        dropout:        dropout rate.
        dropout2d:      spatial (Conv2d) dropout rate.
        cls_hidden_dim: hidden dim inside cls MLP.  Default 64 gives
                        Linear(128→64→2).  Pass 0 for legacy single-Linear head.
        bbox_hidden_dim: hidden dim inside bbox MLP.  Default 64 gives
                         Linear(128→64→4).  Pass 0 for legacy single-Linear head.
    """

    def __init__(self, in_channels, num_classes=2, dropout=DROPOUT,
                 dropout2d=DROPOUT2D, cls_hidden_dim: int = 64,
                 bbox_hidden_dim: int = 64):
        super().__init__()
        mid = 256
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout2d),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout2d),
            nn.Conv2d(mid, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = _make_mlp(128, num_classes, cls_hidden_dim, dropout)
        self.bbox_head = _make_mlp(128, 4, bbox_hidden_dim, dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        cls_logits = self.cls_head(x)
        bbox = torch.sigmoid(self.bbox_head(x))
        return cls_logits, bbox


# ---------------------------------------------------------------------------
# Multi-task model
# ---------------------------------------------------------------------------

class MultiTaskModel(nn.Module):
    """
    Shared backbone → [optional FPN neck] → ClassificationHead + 4 × FeatureDetectionHead.

    Architecture compatibility:
        The default parameters produce the *current* model.  For checkpoints trained
        with the earlier (legacy) architecture, pass::

            neck_type=""
            classification_hidden_dims=[512]
            feature_cls_hidden_dim=0
            feature_bbox_hidden_dim=64
    """

    def __init__(
        self,
        backbone_name: str = MODEL_NAME,
        pretrained: bool = PRETRAINED,
        dropout: float = DROPOUT,
        neck_type: Optional[str] = NECK_TYPE,
        neck_channels: int = NECK_CHANNELS,
        classification_hidden_dims: list = None,
        feature_cls_hidden_dim: int = 64,
        feature_bbox_hidden_dim: int = 64,
    ):
        super().__init__()
        self.backbone, out_ch, _fpn_ch = _get_backbone(backbone_name, pretrained)
        self.backbone_name = backbone_name

        self.neck = None
        head_in_channels = out_ch
        if neck_type == "fpn":
            self.neck = SimpleNeck(out_ch, neck_channels)
            head_in_channels = neck_channels

        self.class_head = ClassificationHead(
            head_in_channels, NUM_BIRADS, dropout,
            hidden_dims=classification_hidden_dims,
        )

        self.feature_heads = nn.ModuleDict({
            feat: FeatureDetectionHead(
                head_in_channels, NUM_FEATURE_CLASSES[feat],
                dropout, DROPOUT2D,
                cls_hidden_dim=feature_cls_hidden_dim,
                bbox_hidden_dim=feature_bbox_hidden_dim,
            )
            for feat in FEATURE_TYPES
        })

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)

        birads_logits = self.class_head(feats)

        feature_outputs = {}
        for name, head in self.feature_heads.items():
            feature_outputs[name] = head(feats)

        return birads_logits, feature_outputs


def create_model(backbone_name: str = None, pretrained: bool = None, dropout: float = None,
                 neck_type: str = None, neck_channels: int = None,
                 classification_hidden_dims: list = None,
                 feature_cls_hidden_dim: int = None,
                 feature_bbox_hidden_dim: int = None):
    kwargs = {}
    if backbone_name is not None:
        kwargs["backbone_name"] = backbone_name
    if pretrained is not None:
        kwargs["pretrained"] = pretrained
    if dropout is not None:
        kwargs["dropout"] = dropout
    if neck_type is not None:
        kwargs["neck_type"] = neck_type
    if neck_channels is not None:
        kwargs["neck_channels"] = neck_channels
    if classification_hidden_dims is not None:
        kwargs["classification_hidden_dims"] = classification_hidden_dims
    if feature_cls_hidden_dim is not None:
        kwargs["feature_cls_hidden_dim"] = feature_cls_hidden_dim
    if feature_bbox_hidden_dim is not None:
        kwargs["feature_bbox_hidden_dim"] = feature_bbox_hidden_dim
    return MultiTaskModel(**kwargs)
