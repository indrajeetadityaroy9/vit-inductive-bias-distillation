# Mitigating Inductive Bias Mismatch in Heterogeneous Knowledge Distillation

A research framework for studying knowledge transfer between architecturally different neural networks, with a focus on CNN-to-ViT distillation and the "negative transfer" phenomenon.

## Core Finding

> **"The negative transfer observed when distilling CNNs into Vision Transformers stems from conflicting inductive biases—not weak teachers."**

We demonstrate that:
1. **Alignment > Capacity**: A weaker teacher (ConvNeXt, 93.1%) produces a *better* student (90.11%) than a stronger teacher (ResNet-18, 95.1% → 90.05%) due to architectural alignment
2. **Structure is 99.5% Sufficient**: Self-supervised structural distillation (DINOv2 → DeiT) achieves 89.69% accuracy using *only geometric feature alignment*—no task-specific labels required
3. **The "Locality Curse"**: CNN-distilled ViTs develop local attention patterns (layer 2: 1.85 patch distance) while DINOv2-distilled ViTs maintain global attention (layer 2: 3.49 patch distance, +88.6%)

## Motivation

Traditional knowledge distillation assumes that a stronger teacher produces a better student. However, when the teacher (CNN) and student (ViT) have fundamentally different inductive biases:
- **CNNs**: Local receptive fields, translation equivariance, hierarchical features
- **ViTs**: Global self-attention, position-aware, flat feature hierarchy

The student learns to "think like a CNN" rather than exploiting its native architectural strengths. This manifests as:
- Collapsed attention distances in middle layers
- Redundant late-layer representations
- Suboptimal accuracy despite strong teacher supervision

## Methodology

We compare three distillation approaches on CIFAR-10 with DeiT-Tiny students:

| Experiment | Teacher | Distillation Type | Hypothesis |
|------------|---------|-------------------|------------|
| **EXP-1** | ResNet-18 (classic CNN) | Soft KL | Baseline - expect locality bias transfer |
| **EXP-2** | ConvNeXt V2 (modern CNN) | Soft KL | Reduced bias mismatch due to ViT-like design |
| **EXP-3** | DINOv2 (self-supervised ViT) | CKA Structural | No bias mismatch - structural alignment only |

**Analytics Pipeline:**
- **CKA Self-Similarity**: Measures layer-wise representation diversity
- **Mean Attention Distance**: Quantifies local vs. global attention patterns
- **Transfer Efficiency**: Student accuracy / Teacher accuracy

## Features

1. **Adaptive Architecture Design**: Dataset-specific CNN configurations with ResidualBlocks and SE attention
2. **Vision Transformer (DeiT/ViT)**: Data-efficient Image Transformer with distillation token support
3. **Multiple Distillation Modes**:
   - Soft KL distillation (CNN → ViT)
   - CKA structural distillation (DINOv2 → ViT)
   - Token representation + correlation distillation (CST-style)
4. **Research Analytics**: CKA heatmaps, attention distance analysis, transfer efficiency metrics
5. **Multi-GPU Training**: Distributed Data Parallel (DDP) with H100 optimizations
6. **Advanced Augmentation**: RandAugment, MixUp, CutMix, Cutout, AutoAugment
7. **Modern Training**: Mixed precision (BF16), torch.compile, cosine annealing, label smoothing

## Architecture

**MNIST Configuration** (6 residual blocks, 709K parameters):
```
Input (1×28×28)
    ↓
Conv(1→32, 3×3) + BN + ReLU + MaxPool     → 32×14×14
    ↓
ResidualBlock(32→32) × 2 + SE              → 32×14×14
    ↓
ResidualBlock(32→64, stride=2) + SE        → 64×7×7
ResidualBlock(64→64) + SE                  → 64×7×7
    ↓
ResidualBlock(64→128, stride=2) + SE       → 128×4×4
ResidualBlock(128→128) + SE                → 128×4×4
    ↓
AdaptiveAvgPool + Classifier(128→64→10)
```

**CIFAR-10 Configuration** (11 residual blocks, 17.6M parameters):
```
Input (3×32×32)
    ↓
Conv(3→64, 3×3) + BN + ReLU               → 64×32×32
    ↓
ResidualBlock(64→64) × 2 + SE              → 64×32×32
    ↓
ResidualBlock(64→128, stride=2) + SE       → 128×16×16
ResidualBlock(128→128) × 2 + SE            → 128×16×16
    ↓
ResidualBlock(128→256, stride=2) + SE      → 256×8×8
ResidualBlock(256→256) × 2 + SE            → 256×8×8
    ↓
ResidualBlock(256→512, stride=2) + SE      → 512×4×4
ResidualBlock(512→512) × 2 + SE            → 512×4×4
    ↓
AdaptiveAvgPool + Classifier(512→256→10)
```

Both CNN architectures employ:
- **Residual Connections** for gradient flow in deep networks
- **Squeeze-and-Excitation (SE) Blocks** for channel attention
- **Batch Normalization** after each convolutional layer
- **Dropout** (p=0.3) for regularization
- **ReLU activations** for non-linearity

### DeiT/ViT Architecture

**DeiT-Tiny Configuration** (~4M parameters):
```
Input (C×H×W)
    ↓
Hybrid Patch Embedding (Conv Stem + Projection)
    ↓
[CLS] + [DIST]* + Patch Tokens + Positional Embedding
    ↓
Transformer Encoder × 12
  - Multi-Head Self-Attention (3 heads)
  - MLP (ratio=2.0)
  - Drop Path (0.1)
  - Layer Normalization
    ↓
Classification Head (CLS token → num_classes)
Distillation Head* (DIST token → num_classes)

* Only present when distillation=true
```

**Key Features:**
- **Hybrid Patch Embedding**: Conv stem for better local feature extraction before patch projection
- **Class Token Dropout**: Regularization by replacing CLS token with patch mean (10% probability)
- **Flexible Inference**: Choose between CLS, DIST, or averaged outputs
- **Distillation Token**: Optional token for knowledge transfer from CNN teachers

## Dataset Preprocessing

**MNIST Pipeline**:
- Grayscale conversion with automatic brightness inversion (threshold: 127)
- Resize to 28×28 pixels
- Normalization: μ=0.1307, σ=0.3081
- **Training augmentation**: Random rotation (±10°), random affine translation (10%), Cutout

**CIFAR-10 Pipeline**:
- **Training augmentation**: AutoAugment (CIFAR10 policy), CutMix (α=1.0), random crop (32×32, padding=4), random horizontal flip, Cutout
- **Test preprocessing**: Resize to 32×32 pixels
- Normalization: μ=[0.4914, 0.4822, 0.4465], σ=[0.2470, 0.2435, 0.2616]

## Training

**Optimization**:
- Optimizer: AdamW (MNIST) / SGD with Nesterov momentum (CIFAR-10)
- Loss function: Label Smoothing Cross-Entropy (smoothing=0.1)
- Gradient clipping: max norm = 1.0
- Learning rate scheduler: Cosine Annealing with warmup
- Stochastic Weight Averaging (SWA) in final 25% of training

**Training Configuration**:
- Multi-GPU: DDP with 2× NVIDIA H100 80GB
- Batch size: 512 per GPU (1024 effective)
- MNIST epochs: 50
- CIFAR-10 epochs: 100
- Mixed precision training (AMP) enabled
- Checkpointing: Save best model based on validation accuracy

## Experimental Setup

### Datasets

**MNIST** (Handwritten Digits):
- Training samples: 60,000
- Test samples: 10,000
- Image size: 28×28 (grayscale)
- Classes: 10 (digits 0-9)

**CIFAR-10** (Natural Images):
- Training samples: 50,000
- Test samples: 10,000
- Image size: 32×32 (RGB)
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1-Score**: Macro-averaged across all classes
- **Confusion Matrix**: Per-class error analysis
- **Top-K Accuracy**: Top-1 and Top-5 predictions

## Results

### Primary Research Results (Inductive Bias Study)

| Experiment | Teacher | Teacher Acc | Student Acc | Transfer Efficiency |
|------------|---------|-------------|-------------|---------------------|
| Baseline | None | — | 86.02% | — |
| **EXP-1** | ResNet-18 | 95.10% | 90.05% | 94.7% |
| **EXP-2** | ConvNeXt V2 | 93.10% | **90.11%** | **96.8%** |
| **EXP-3** | DINOv2 | N/A (self-sup) | 89.69% | — |

### Attention Distance Analysis (The "Locality Curse")

Mean attention distance measures how far each query token attends on average (in patch units). Higher = more global attention.

| Layer | EXP-1 (ResNet) | EXP-2 (ConvNeXt) | EXP-3 (DINOv2) | DINO Δ vs ResNet |
|-------|----------------|------------------|----------------|------------------|
| 0 | 3.83 | 3.78 | 4.01 | +4.7% |
| 1 | 2.98 | 2.62 | 2.79 | -6.4% |
| 2 | **1.85** | 2.20 | **3.49** | **+88.6%** |
| 3 | 2.68 | 2.33 | **3.56** | **+32.8%** |
| 4 | 2.94 | 2.49 | **3.36** | **+14.3%** |
| 5 | 2.57 | 2.82 | **3.74** | **+45.5%** |
| 6 | 3.22 | 3.50 | **3.77** | **+17.1%** |
| 7 | 3.83 | 3.84 | 3.95 | +3.1% |
| 8 | 3.83 | 3.83 | 3.93 | +2.6% |
| 9 | 3.97 | 3.85 | 3.96 | -0.3% |
| 10 | 4.01 | 4.04 | 4.08 | +1.7% |
| 11 | 4.04 | 4.01 | 4.08 | +1.0% |
| **Mean** | **3.31** | **3.28** | **3.73** | **+12.7%** |

**Key Observation:** CNN-distilled students (EXP-1, EXP-2) exhibit attention collapse in layers 2-6, mimicking CNN's local receptive field behavior. DINOv2-distilled students maintain consistently high attention distances.

### CKA Self-Similarity Analysis

| Metric | EXP-1 (ResNet) | EXP-2 (ConvNeXt) | EXP-3 (DINOv2) |
|--------|----------------|------------------|----------------|
| Early-Late CKA (L0-L11) | 0.358 | 0.312 | **0.515** |
| Mid-Late CKA (L5-L11) | 0.887 | 0.862 | **0.743** |
| Layer Diversity Score* | 0.56 | 0.58 | **0.69** |

*Layer Diversity = 1 - mean(off-diagonal CKA). Higher = more diverse representations.

**Interpretation:** DINOv2 distillation creates more diverse, less redundant representations. CNN-distilled models show high redundancy in late layers (mid-late CKA ~0.86-0.89).

### Practical Recommendations

| Scenario | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| Maximum accuracy, labeled teacher | ConvNeXt → ViT soft KL | 96.8% transfer efficiency |
| No labeled data for teacher | DINOv2 → ViT structural | 99.5% of supervised performance |
| Interpretability important | DINOv2 → ViT structural | Preserves global attention |
| Classic CNN teacher available | Use it, expect locality bias | Still +4% over baseline |

---

### Additional Model Results

#### Teacher Models (CIFAR-10)

| Model | Parameters | Accuracy |
|-------|------------|----------|
| ResNet-18 (modified stem) | 11.2M | 95.10% |
| ConvNeXt V2 Tiny | 28.6M | 93.10% |
| AdaptiveCNN | 17.6M | 82.90% |

#### DeiT-Tiny Student Variants

| Distillation Method | Teacher | CIFAR-10 Acc |
|---------------------|---------|--------------|
| None (baseline) | — | 86.02% |
| Soft KL | AdaptiveCNN | 84.39% |
| Soft KL | ResNet-18 | 90.05% |
| Soft KL | ConvNeXt V2 | **90.11%** |
| CKA Structural | DINOv2 | 89.69% |
| CST Token+Rel | DINOv2 | 89.18% |

### Negative Transfer Effect (Historical Context)

Early experiments showed **CNN-distilled DeiT underperforms ViT without distillation** when using a weak teacher:

```
ViT (86.02%) > DeiT-CNN (84.39%) > AdaptiveCNN (82.90%)
       +1.63%            +1.49%
```

This motivated the current research into stronger teachers and structural distillation methods.

### Training Time (2× NVIDIA H100 80GB)

| Model | Dataset | Training Time | Epochs |
|-------|---------|---------------|--------|
| AdaptiveCNN | MNIST | ~1 minute | 20 |
| AdaptiveCNN | CIFAR-10 | ~4 minutes | 40 |
| ResNet-18 Teacher | CIFAR-10 | ~8 minutes | 200 |
| ConvNeXt V2 Teacher | CIFAR-10 | ~12 minutes | 200 |
| DeiT (CNN Distilled) | CIFAR-10 | ~12 minutes | 100 |
| DeiT (CKA Structural) | CIFAR-10 | ~20 minutes | 200 |
| DeiT (CST-SSL) | CIFAR-10 | ~25 minutes | 100 |
| ViT Baseline | CIFAR-10 | ~17 minutes | 100 |

## Usage

### Training

**AdaptiveCNN:**
```bash
# Train MNIST on single GPU
python main.py train configs/mnist_improved_config.yaml

# Train MNIST on multiple GPUs with DDP
python main.py train configs/mnist_improved_config.yaml --num-gpus 2

# Train CIFAR-10 on multiple GPUs with DDP
python main.py train configs/cifar_improved_config.yaml --num-gpus 2
```

**ViT (without distillation):**
```bash
# Train ViT on MNIST
python main.py train configs/vit_mnist_config.yaml --num-gpus 2

# Train ViT on CIFAR-10
python main.py train configs/vit_cifar_config.yaml --num-gpus 2
```

**DeiT (with CNN knowledge distillation):**
```bash
# First train the teacher (AdaptiveCNN)
python main.py train configs/cifar_improved_config.yaml --num-gpus 2

# Then train DeiT with distillation from the teacher
python main.py train-distill configs/deit_cifar_config.yaml --num-gpus 2
```

**DeiT (with CST-style self-supervised distillation):**
```bash
# Train DeiT with token distillation from DINOv2 (no CNN teacher needed)
python main.py train-ss-distill configs/deit_ss_distill_cifar_config.yaml --num-gpus 1
```

**DeiT (with CKA structural distillation):**
```bash
# Train DeiT with CKA structural alignment to DINOv2
python main.py train-ss-distill configs/deit_ss_distill_cka_cifar_config.yaml --num-gpus 2
```

### Research Analytics

```bash
# Run analytics on trained model (CKA heatmap + attention distance)
python main.py analyze configs/deit_ss_distill_cka_cifar_config.yaml \
    outputs/checkpoints/exp3_dino/best_model.pth \
    --metrics cka,attention \
    --output-dir outputs/analytics/exp3_dino

# Generate comparison plots across all experiments
python scripts/generate_comparison_plots.py
```

### Evaluation

```bash
# Evaluate MNIST model
python main.py evaluate configs/mnist_improved_config.yaml ./outputs/checkpoints/best_model.pth

# Evaluate CIFAR-10 model
python main.py evaluate configs/cifar_improved_config.yaml ./outputs/checkpoints/best_model.pth
```

### Single Image Inference

```bash
# Inference with test-time augmentation
python main.py test configs/cifar_improved_config.yaml ./outputs/checkpoints/best_model.pth image.jpg --tta
```

## Project Structure

```
├── configs/                    # Configuration files
│   ├── mnist_improved_config.yaml         # AdaptiveCNN for MNIST
│   ├── cifar_improved_config.yaml         # AdaptiveCNN for CIFAR-10
│   ├── vit_mnist_config.yaml              # ViT for MNIST (no distillation)
│   ├── vit_cifar_config.yaml              # ViT for CIFAR-10 (no distillation)
│   ├── deit_mnist_config.yaml             # DeiT for MNIST (CNN distillation)
│   ├── deit_cifar_config.yaml             # DeiT for CIFAR-10 (CNN distillation)
│   ├── deit_ss_distill_cifar_config.yaml  # DeiT for CIFAR-10 (CST-SSL distillation)
│   ├── deit_ss_distill_cka_cifar_config.yaml  # DeiT with CKA structural distillation
│   ├── resnet18_cifar_config.yaml         # ResNet-18 teacher
│   └── convnext_v2_cifar_config.yaml      # ConvNeXt V2 teacher
├── src/
│   ├── config.py              # Configuration management
│   ├── models.py              # AdaptiveCNN with ResidualBlock + SE
│   ├── teachers.py            # ResNet-18, ConvNeXt V2 teacher models
│   ├── vit.py                 # DeiT/ViT implementation
│   ├── distillation.py        # Knowledge distillation (KL, CKA, CST)
│   ├── analytics.py           # CKA heatmaps, attention distance analysis
│   ├── datasets.py            # Data loading and augmentation
│   ├── training.py            # Trainer and DDPTrainer classes
│   ├── evaluation.py          # Metrics and visualization
│   └── visualization.py       # GradCAM and feature maps
├── scripts/
│   └── generate_comparison_plots.py  # Research visualization generator
├── outputs/
│   ├── checkpoints/           # Model checkpoints
│   ├── evaluation/            # Evaluation plots
│   └── analytics/             # Research analytics outputs
│       ├── ANALYSIS_REPORT.md         # Full research report
│       ├── cka_comparison.png         # 3-panel CKA heatmap
│       ├── accuracy_comparison.png    # Accuracy bar chart
│       ├── attention_comparison.png   # Attention distance plot
│       ├── transfer_efficiency.png    # Teacher vs student scatter
│       ├── exp1_resnet/               # EXP-1 analytics
│       ├── exp2_convnext/             # EXP-2 analytics
│       └── exp3_dino/                 # EXP-3 analytics
└── main.py                    # CLI entry point
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- timm >= 0.9.0 (for ConvNeXt V2, DINOv2)
- numpy
- tqdm
- matplotlib
- scikit-learn

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{inductive_bias_distillation_2025,
  title={Mitigating Inductive Bias Mismatch in Heterogeneous Knowledge Distillation},
  author={[Author]},
  year={2025},
  note={GitHub repository},
  howpublished={\url{https://github.com/[repo]}}
}
```
