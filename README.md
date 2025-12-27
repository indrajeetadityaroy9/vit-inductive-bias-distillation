# Dataset Adaptive Image Classification

A modular CNN classification pipeline that dynamically adjusts its architecture based on dataset characteristics. The pipeline implements dataset-specific model configurations, preprocessing strategies, and augmentation techniques to optimize performance across diverse image classification tasks.

## Features

1. **Adaptive Architecture Design**: A single model class that dynamically instantiates dataset-specific layer configurations with ResidualBlocks and SE attention
2. **Vision Transformer (DeiT/ViT)**: Data-efficient Image Transformer with optional knowledge distillation from CNN teachers
3. **Self-Supervised Distillation (CST-style)**: Token representation and correlation distillation from DINOv2 pretrained teachers
4. **Multi-GPU Training**: Distributed Data Parallel (DDP) support for efficient multi-GPU training
5. **Knowledge Distillation**: Hard/soft distillation with configurable alpha scheduling (constant, linear, cosine)
6. **Advanced Augmentation**: RandAugment, MixUp, CutMix, Cutout, and traditional augmentations
7. **Modern Training Techniques**: Mixed precision (AMP/BF16), Stochastic Weight Averaging (SWA), label smoothing, cosine annealing
8. **Interpretability Tools**: GradCAM, feature map visualization, confusion matrices, ROC curves

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

### Model Comparison

| Model | MNIST | CIFAR-10 |
|-------|-------|----------|
| **AdaptiveCNN** | 99.08% | 82.90% |
| **DeiT (CNN Distilled)** | 99.54% | 84.39% |
| **ViT (No Distillation)** | 99.64% | 86.02% |
| **DeiT (CST-SSL Distilled)** | - | **89.18%** |

### AdaptiveCNN (Teacher) Performance

**MNIST** (709K parameters):

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.08%** |
| **Precision (Macro)** | 0.9908 |
| **Recall (Macro)** | 0.9908 |
| **F1-Score (Macro)** | 0.9908 |

**CIFAR-10** (17.6M parameters):

| Metric | Score |
|--------|-------|
| **Accuracy** | **82.90%** |
| **Precision (Macro)** | 0.8290 |
| **Recall (Macro)** | 0.8290 |
| **F1-Score (Macro)** | 0.8290 |

### DeiT with Knowledge Distillation

**MNIST** (3.9M parameters, distilled from AdaptiveCNN):

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.54%** |
| **Precision (Macro)** | 0.9954 |
| **Recall (Macro)** | 0.9954 |
| **F1-Score (Macro)** | 0.9954 |

**CIFAR-10** (4.3M parameters, distilled from AdaptiveCNN):

| Metric | Score |
|--------|-------|
| **Accuracy** | **84.39%** |
| **Precision (Macro)** | 0.8431 |
| **Recall (Macro)** | 0.8439 |
| **F1-Score (Macro)** | 0.8422 |

### ViT without Distillation (Baseline)

**MNIST** (3.9M parameters):

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.64%** |
| **Precision (Macro)** | 0.9964 |
| **Recall (Macro)** | 0.9964 |
| **F1-Score (Macro)** | 0.9964 |
| **AUC (Macro)** | 1.0000 |

**CIFAR-10** (4.3M parameters):

| Metric | Score |
|--------|-------|
| **Accuracy** | **86.02%** |
| **Precision (Macro)** | 0.8600 |
| **Recall (Macro)** | 0.8602 |
| **F1-Score (Macro)** | 0.8591 |
| **AUC (Macro)** | 0.9886 |

### DeiT with Self-Supervised Distillation (CST-style)

**CIFAR-10** (4.3M parameters, distilled from DINOv2 ViT-S/14):

| Metric | Score |
|--------|-------|
| **Accuracy** | **89.18%** |
| **Precision (Macro)** | 0.8920 |
| **Recall (Macro)** | 0.8918 |
| **F1-Score (Macro)** | 0.8911 |
| **AUC (Macro)** | 0.9903 |
| **Top-3 Accuracy** | 98.08% |
| **Top-5 Accuracy** | 99.47% |

**Method:** CST-style (Contrastive Self-supervised Token) distillation using:
- **Token Representation Loss (L_tok)**: Aligns intermediate layer tokens between student and DINOv2 teacher via learnable projectors
- **Token Correlation Loss (L_rel)**: Matches token-token relational structure using KL divergence on correlation matrices
- **Staged Training**: L_tok only for epochs 0-9, then L_tok + L_rel for epochs 10+
- **Loss**: `L = L_ce + 1.0 × L_tok + 0.1 × L_rel`

### Negative Transfer Effect Analysis

A key finding from our experiments: **CNN-distilled DeiT underperforms ViT without distillation**.

**CIFAR-10 Results (CNN Distillation):**
```
ViT (86.02%) > DeiT-CNN (84.39%) > CNN (82.90%)
       +1.63%            +1.49%
```

**Root Cause:** The teacher CNN (82.90%) was weaker than what the student could achieve independently. This caused:

1. **Negative Knowledge Transfer** - DeiT was constrained by the teacher's suboptimal representations
2. **Teacher Ceiling Effect** - With α=0.6, 60% of the loss pushed toward the teacher's 82.9% accuracy ceiling
3. **Constrained Optimization** - Hard distillation forced matching teacher predictions instead of learning optimal features

**Solution: Self-Supervised Distillation (CST-style)**

By replacing the weak CNN teacher with DINOv2 (a self-supervised pretrained ViT) and distilling token representations instead of logits, we avoid the negative transfer:

```
DeiT-CST (89.18%) > ViT (86.02%) > DeiT-CNN (84.39%) > CNN (82.90%)
         +3.16%           +1.63%            +1.49%
```

**Why CST-SSL Works:**
1. **No Logit Imitation** - Student learns relational structure, not teacher's classification decisions
2. **Strong Teacher Features** - DINOv2's self-supervised features provide rich supervision without task-specific bias
3. **Intermediate Alignment** - Token-level distillation preserves inductive biases while transferring knowledge

**Key Insight:** Knowledge distillation with weak teachers causes negative transfer. Self-supervised relational distillation avoids this by transferring feature structure rather than classification outputs.

### Training Time (2× NVIDIA H100 80GB)

| Model | Dataset | Training Time | Epochs |
|-------|---------|---------------|--------|
| AdaptiveCNN | MNIST | ~1 minute | 20 |
| AdaptiveCNN | CIFAR-10 | ~4 minutes | 40 |
| DeiT (CNN Distilled) | MNIST | ~4 minutes | 50 |
| DeiT (CNN Distilled) | CIFAR-10 | ~12 minutes | 100 |
| ViT | MNIST | ~4 minutes | 50 |
| ViT | CIFAR-10 | ~17 minutes | 100 |
| DeiT (CST-SSL) | CIFAR-10 | ~25 minutes | 100 |

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
│   └── deit_ss_distill_cifar_config.yaml  # DeiT for CIFAR-10 (CST-SSL distillation)
├── src/
│   ├── config.py              # Configuration management
│   ├── models.py              # AdaptiveCNN with ResidualBlock + SE
│   ├── vit.py                 # DeiT/ViT implementation
│   ├── distillation.py        # Knowledge distillation trainer
│   ├── datasets.py            # Data loading and augmentation
│   ├── training.py            # Trainer and DDPTrainer classes
│   ├── evaluation.py          # Metrics and visualization
│   └── visualization.py       # GradCAM and feature maps
├── outputs/                    # Training outputs
│   ├── checkpoints/           # Model checkpoints
│   └── evaluation/            # Evaluation plots
└── main.py                    # CLI entry point
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- tqdm
- matplotlib
- scikit-learn
