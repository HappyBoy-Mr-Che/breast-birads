# 🩺 乳腺超声影像 BI-RADS 分类与特征识别

> **Breast Ultrasound BI-RADS Classification & Feature Detection via Multi-Task Deep Learning**

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/cuda-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

---

## 📋 项目概述

本项目构建了一个**多任务深度学习模型**，用于乳腺超声影像的自动分析，同时完成两项核心任务：

1. **BI-RADS 等级分类** — 将乳腺肿瘤分为 6 个风险等级（2类、3类、4A类、4B类、4C类、5类）
2. **超声特征检测** — 识别 4 项关键影像学特征（边界、钙化、形状、方向），并预测其边界框位置

模型采用 **共享骨干网络 + 多头架构**，一次前向传播即可同时输出分类结果和多项特征检测结果，兼顾准确性与效率。

---

## 🎯 任务定义与临床背景

### 为什么需要这个项目？

乳腺超声是乳腺癌筛查的重要手段，但传统依赖医生人工阅片，存在以下痛点：

- **主观性强**：不同医生对同一影像的 BI-RADS 评级可能不一致
- **工作量大**：大量筛查影像需要逐一阅片，医生容易疲劳
- **特征标注繁琐**：BI-RADS 评级需要综合多项影像学特征（边界是否光滑、有无钙化、形状是否规则、方向是否平行），人工提取效率低

### 多任务学习框架

本项目的关键设计理念：**BI-RADS 分类与特征检测是高度相关的任务，共享底层视觉特征可以互相促进**。

| 任务类型 | 具体内容 | 输出形式 |
|:---:|:---|:---|
| **BI-RADS 分类** | 6 类风险等级判定 | 类别概率分布 |
| **边界检测** | 光滑 vs 不光滑 | 二分类 + 边界框 |
| **钙化检测** | 有钙化 vs 无钙化 | 二分类 + 边界框 |
| **形状检测** | 规则 vs 不规则 | 二分类 + 边界框 |
| **方向检测** | 平行 vs 不平行 | 二分类 + 边界框 |

---

## 🏗️ 模型架构

```
输入图像 (640×320)
      │
      ▼
┌──────────────────┐
│   Backbone CNN    │  EfficientNet-B3 / ResNet50 / ConvNeXt-Tiny / MobileNetV3
│  (ImageNet 预训练) │
└────────┬──────────┘
         │
         ▼
┌──────────────────┐
│    FPN Neck       │  轻量级多尺度特征融合（可选）
│  (SimpleFPN)      │  原始尺度 + 0.5×下采样 + 2×上采样 → 融合
└────────┬──────────┘
         │
    ┌────┴────────────┬────────────┬────────────┬────────────┐
    ▼                 ▼            ▼            ▼            ▼
┌─────────┐   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ BI-RADS │   │ Boundary │ │Calcification│ │  Shape  │ │Direction │
│  分类头  │   │  检测头   │ │   检测头    │ │  检测头   │ │  检测头   │
│ 6-class │   │ 2-cls +  │ │ 2-cls +   │ │ 2-cls +  │ │ 2-cls +  │
│         │   │  bbox    │ │  bbox     │ │  bbox    │ │  bbox    │
└─────────┘   └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 架构要点

- **共享骨干网络**：EfficientNet-B3 在 ImageNet 上预训练，提取通用视觉特征
- **轻量 FPN 颈**：对骨干特征进行 3 个尺度（原尺寸、0.5×、2×）的卷积处理后融合，增强对不同尺度病灶的感知能力
- **多层分类头**：1536→512→256→6，逐层 Dropout 衰减（0.5 → 0.3 → 0.18），防止过拟合
- **深层检测头**：3 层 Conv2d (256→256→128) + Dropout2d 空间正则化，每个头包含分类分支和边界框回归分支

---

## 🔬 关键技术创新与改进

### 1. 多任务不确定性加权（Kendall et al. 2018）

使用 **Homoscedastic Uncertainty** 自动平衡分类损失与检测损失，替代手工调参：

$$\mathcal{L}_{\text{total}} = \frac{1}{2\sigma_{\text{cls}}^2} \cdot \mathcal{L}_{\text{cls}} + \frac{1}{2\sigma_{\text{det}}^2} \cdot \mathcal{L}_{\text{det}} + \log \sigma_{\text{cls}} + \log \sigma_{\text{det}}$$

其中 $\sigma_{\text{cls}}$ 和 $\sigma_{\text{det}}$ 是可学习的噪声参数，训练过程中自动调整两个任务的相对权重。

### 2. 差异化 Focal Loss 设计

针对 4 项特征的**类别不平衡程度不同**，设计差异化 Focal Loss γ 参数：

| 特征 | 正负样本比 | Focal γ | 设计理由 |
|:---|:---:|:---:|:---|
| 边界 (boundary) | ≈1:4 | 2.5 | 中等不平衡，适度聚焦难样本 |
| 钙化 (calcification) | ≈1:1 | 2.0 | 相对平衡，使用基准 γ |
| 形状 (shape) | ≈1:5 | 3.0 | 严重不平衡 + 难区分，高 γ 抑制易负样本 |
| 方向 (direction) | ≈1:4.75 | 3.0 | 最难特征，高 γ 聚焦难样本 |

### 3. 平滑类别权重策略

使用 $\sqrt{1/\text{freq}}$ 代替传统的 $1/\text{freq}$ 作为类别权重：

- **问题**：传统逆频率加权在严重不平衡时（如 1:5），少数类权重可达 5.0，导致模型倾向预测为少数类（假阳性激增）
- **方案**：开方加权将极端值从 5.0 压缩到约 2.24，在不牺牲召回率的前提下显著提升精确率

### 4. 空间正则化（Spatial Dropout）

在检测头卷积层后引入 `Dropout2d`（按通道随机置零），迫使模型学习更鲁棒的特征表达，有效抑制假阳性检测。

### 5. 混合增强策略

| 增强方法 | 参数 | 作用 |
|:---|:---|:---|
| 随机仿射变换 | ±20° 旋转 + 0.85~1.15 缩放 | 模拟不同扫描角度和深度 |
| 弹性形变 | α=40, σ=4 | 模拟超声探头的组织形变 |
| 亮度/对比度 | ±0.3 亮度 + Gamma 校正 | 模拟不同设备参数 |
| MixUp | α=0.2, 50% 概率 | 样本间插值，提升泛化 |
| 水平翻转 | 50% 概率 | 模拟不同扫描方向 |

### 6. 训练策略优化

- **差异学习率**：检测头 lr = 3e-4，骨干网络 lr = 3e-5（0.1×）
- **Cosine 退火** + 5 epoch Warmup：避免初期训练不稳定
- **混合精度训练（AMP）**：加速训练，降低显存占用
- **Early Stopping**：以 4 项特征的平均 F1 为监控指标，patience=25
- **梯度裁剪**：max_norm=3.0，防止梯度爆炸

---

## 📊 实验结果

### 核心指标（验证集）

| 指标 | 符号 | 数值 |
|:---|:---:|:---:|
| BI-RADS 6 类分类准确率 | ACC | **64.05%** |
| 4 特征平均分类准确率 | FEA | **83.09%** |
| 4 特征平均召回率（灵敏度） | SEN | **79.67%** |
| 4 特征平均特异性 | SPE | **83.90%** |
| 4 特征平均 F1 分数 | F1 | **51.64%** |

### 各特征检测性能（二分类）

| 特征 | 准确率 | 精确率 | 召回率 | F1 分数 | 特异性 | IoU |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 边界 (boundary) | 80.6% | 42.1% | 79.3% | 55.0% | 80.8% | 47.8% |
| 钙化 (calcification) | 78.4% | 68.9% | 74.5% | 71.6% | 80.7% | 48.2% |
| 形状 (shape) | 79.0% | 35.1% | 74.0% | 47.7% | 79.7% | 46.2% |
| 方向 (direction) | 94.3% | 19.6% | 90.9% | 32.3% | 94.4% | 46.3% |

> ℹ️ *以上数据来自验证集最佳 epoch（第 29 轮），基于 EfficientNet-B3 + FPN 模型。方向(direction)是区分难度最大的特征——召回率 90.9% 但精确率仅 19.6%，说明模型倾向于过度预测"不平行"，这是后续优化的重点方向。*

> 💡 *在测试集上运行 `eval_confusion.py` 可生成完整的混淆矩阵和 per-class 指标。上述数值基于验证集最佳 checkpoint。*

### 训练曲线可视化

训练过程自动记录至 `outputs/logs/training_log.json`，可使用可视化工具生成以下图表：

```
outputs/logs/
├── loss_curves.png           # 总损失 / 分类损失 / 检测损失
├── accuracy.png              # BI-RADS 分类准确率
├── iou_curves.png            # 4 项特征 IoU 曲线
├── det_breakdown.png         # 每项特征的分类+回归损失
├── feature_cls_acc.png       # 特征分类准确率
├── feature_cls_metrics.png   # 特征 Precision/Recall/F1/Specificity
├── detection_ap.png          # 特征检测 AP@0.5
├── confusion_matrices.png    # 混淆矩阵
└── summary.png               # 综合训练仪表盘
```

---

## 📁 项目结构

```
ruxain_shibie/
├── README.md                          # 本文档
├── .gitignore                         # Git 忽略规则
├── 环境配置.txt                        # 训练环境配置记录
│
├── new_tools/                         # 核心代码
│   ├── config.py                      # 全局配置（路径/超参数/增强参数）
│   ├── dataset.py                     # 数据集加载 + 增强变换
│   ├── models/
│   │   ├── __init__.py
│   │   └── multitask.py               # 模型架构（骨干+颈+多头）
│   ├── losses.py                      # 损失函数（Focal Loss + 多任务损失）
│   ├── utils.py                       # 训练工具（指标/日志/检查点/调度器）
│   ├── train.py                       # 训练主脚本
│   ├── infer.py                       # 推理脚本（测试集预测）
│   ├── preprocess.py                  # 数据预处理与统计分析
│   ├── visualize.py                   # 训练曲线可视化
│   ├── visualize_predictions.py       # 预测结果可视化（画框+分类标签）
│   ├── eval_confusion.py              # 混淆矩阵评估
│   ├── requirements.txt               # Python 依赖
│   └── outputs/                       # 输出目录（被 .gitignore 排除）
│       ├── models/                    # 模型检查点
│       ├── logs/                      # 训练日志 + 可视化图表
│       ├── submission/                # 推理结果 JSON
│       └── visualized/                # 预测结果图像
│
├── document/                          # LaTeX 课程报告
│   ├── document.tex                   # 报告源码
│   ├── document.pdf                   # 编译后 PDF
│   └── images/                        # 报告插图
│
└── create_ppt.py                      # PPT 汇报生成脚本
```

> ⚠️ **注意**：训练数据（`classfy/`, `future/`, `test_A/`）和文档源文件（`document_source/`）因包含大量影像数据和大型文件，已通过 `.gitignore` 排除，不上传到 GitHub。

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+ (CUDA 11.8 推荐)
- NVIDIA GPU (推荐 RTX 4090 / 24GB VRAM)

### 安装

```bash
# 克隆仓库
git clone https://github.com/HappyBoy-Mr-Che/breast-birads.git
cd breast-birads

# 安装依赖
pip install -r new_tools/requirements.txt
```

### 数据准备

按以下结构组织数据集（需自行准备）：

```
ruxain_shibie/
├── classfy/train/           # BI-RADS 分类数据
│   ├── 2类/images/*.jpg
│   ├── 3类/images/*.jpg
│   ├── 4A类/images/*.jpg
│   ├── 4B类/images/*.jpg
│   ├── 4C类/images/*.jpg
│   └── 5类/images/*.jpg
│
├── future/train/            # 特征检测数据
│   ├── images/*.jpg
│   ├── boundary_labels/*.txt     # YOLO 格式标签
│   ├── calcification_labels/*.txt
│   ├── shape_labels/*.txt
│   └── direction_labels/*.txt
│
└── test_A/                  # 测试数据
    ├── class_test/A/...
    └── future_test/A/...
```

### 训练

```bash
cd new_tools

# 使用默认配置训练 (EfficientNet-B3)
python train.py

# 指定参数训练
python train.py --backbone convnext_tiny --epochs 100 --lr 5e-4 --batch_size 16

# 从检查点恢复训练
python train.py --resume outputs/models/checkpoint.pt
```

### 推理

```bash
# 对测试集运行推理
python infer.py --ckpt outputs/models/best.pt

# 指定 checkpoint 和 batch size
python infer.py --ckpt outputs/models/best.pt --batch_size 64
```

### 评估与可视化

```bash
# 生成训练曲线图表
python visualize.py --log outputs/logs/training_log.json

# 生成混淆矩阵（需先运行 infer.py）
python eval_confusion.py

# 预测结果可视化（画框 + 分类标签）
python visualize_predictions.py --ckpt outputs/models/best.pt
```

---

## 🛠️ 技术栈

| 类别 | 技术 |
|:---|:---|
| **深度学习框架** | PyTorch 2.1, torchvision |
| **模型架构** | EfficientNet-B3, ResNet50, ConvNeXt-Tiny, MobileNetV3-Large |
| **训练技巧** | Mixed Precision (AMP), Cosine Annealing, Warmup, MixUp, Label Smoothing |
| **损失函数** | Focal Loss, SmoothL1 Loss, Uncertainty Weighting |
| **图像处理** | OpenCV, Albumentations |
| **可视化** | Matplotlib, PIL |
| **环境** | CUDA 11.8, RTX 4090 (24GB) |

---

## 📝 技能展示（简历用）

本项目展示了以下技能与能力：

- ✅ **多任务学习**：设计并实现共享骨干 + 多任务头的端到端模型架构
- ✅ **损失函数设计**：深入理解 Focal Loss、类别不平衡、不确定性加权，并针对数据特点定制损失策略
- ✅ **训练调优**：差异学习率、Warmup、Cosine 退火、Early Stopping、AMP 混合精度等训练技巧
- ✅ **数据增强**：针对医学影像特点设计增强管线（弹性形变、MixUp 等）
- ✅ **工程能力**：完整的训练/推理/评估/可视化管线，模块化代码结构
- ✅ **医学影像领域知识**：理解 BI-RADS 分级标准、乳腺超声特征解读
- ✅ **科学写作**：LaTeX 撰写课程报告，包含详细的实验设计和结果分析

---

## 📄 许可证

本项目仅用于学习和研究目的。数据集版权归原始提供者所有。

---

## 📮 联系方式

如有问题或建议，欢迎通过 GitHub Issues 联系。

---

*Last updated: 2026-07*
