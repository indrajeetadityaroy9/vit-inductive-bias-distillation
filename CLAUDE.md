# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dataset-adaptive image classification framework with modular CNN and Vision Transformer training pipelines. Supports AdaptiveCNN (ResNet-style with SE blocks), DeiT/ViT, knowledge distillation (CNN→ViT), and self-supervised distillation (DINOv2→DeiT).

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Quick smoke test (trains MNIST for 2 epochs)
bash scripts/quick_test.sh

# Train AdaptiveCNN
python main.py train configs/mnist_improved_config.yaml
python main.py train configs/cifar_improved_config.yaml --num-gpus 2

# Train ViT (no distillation)
python main.py train configs/vit_cifar_config.yaml --num-gpus 2

# Train DeiT with CNN knowledge distillation (requires trained CNN teacher)
python main.py train-distill configs/deit_cifar_config.yaml --num-gpus 2

# Train DeiT with self-supervised distillation (DINOv2 teacher)
python main.py train-ss-distill configs/deit_ss_distill_cifar_config.yaml

# Evaluate trained model
python main.py evaluate configs/cifar_improved_config.yaml ./outputs/checkpoints/best_model.pth

# Single image inference with test-time augmentation
python main.py test configs/cifar_improved_config.yaml ./outputs/checkpoints/best_model.pth image.jpg --tta

# Code quality
black src/
flake8 src/
mypy src/
isort src/

# Run tests
pytest
pytest -v --cov=src  # with coverage
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `train CONFIG [--num-gpus N]` | Train AdaptiveCNN or ViT model |
| `train-distill CONFIG [--num-gpus N]` | Train DeiT with CNN teacher distillation |
| `train-ss-distill CONFIG [--num-gpus N]` | Train DeiT with DINOv2 self-supervised distillation |
| `evaluate CONFIG CHECKPOINT` | Evaluate model on test set |
| `test CONFIG CHECKPOINT IMAGE [--tta]` | Single image inference |

## Architecture

### Source Code Structure (`src/`)

| File | Purpose |
|------|---------|
| `config.py` | YAML config loading, typed config classes (`DataConfig`, `ModelConfig`, `TrainingConfig`, `ViTConfig`, `DistillationConfig`, `SelfSupervisedDistillationConfig`) |
| `models.py` | AdaptiveCNN with ResidualBlocks and SEBlock; `ModelFactory` for registration |
| `vit.py` | DeiT/ViT implementation with PatchEmbed, TransformerBlocks, class/distillation tokens |
| `datasets.py` | DatasetManager, augmentations (Cutout, MixUp, CutMix, AutoAugment) |
| `training.py` | Trainer, DDPTrainer, EarlyStopping, LabelSmoothingCrossEntropy |
| `distillation.py` | DistillationTrainer (CNN→ViT), SelfSupervisedDistillationTrainer (DINOv2→ViT) |
| `evaluation.py` | ModelEvaluator, TestTimeAugmentation, metrics computation |
| `visualization.py` | GradCAM, FeatureMapVisualizer, TrainingVisualizer |

### Data Flow

```
YAML Config → ConfigManager → Config objects
                                   ↓
Dataset (MNIST/CIFAR-10) → DatasetManager → DataLoaders
                                              ↓
                    ModelFactory.create_model()
                    ├→ AdaptiveCNN
                    └→ DeiT
                            ↓
                    Training Loop
                    ├→ Trainer/DDPTrainer (standard)
                    ├→ DistillationTrainer (CNN teacher)
                    └→ SelfSupervisedDistillationTrainer (DINOv2 teacher)
                            ↓
                    ModelEvaluator → Results
```

### Key Design Patterns

- **Factory Pattern**: `ModelFactory.create_model(type, config)` with registered models (`adaptive_cnn`, `deit`)
- **Dataset-Adaptive Architecture**: MNIST uses 6 residual blocks, CIFAR-10 uses 11 blocks (determined by dataset config)
- **DDP Multi-GPU**: `DDPTrainer` wraps model in `DistributedDataParallel`, NCCL backend
- **Mixed Precision**: BF16 for H100 (no GradScaler), FP16 for other GPUs (with GradScaler)

### Distillation Modes

1. **Standard Distillation** (`train-distill`): CNN teacher → DeiT student
   - Hard distillation: student matches teacher's argmax
   - Soft distillation: KL divergence with temperature

2. **Self-Supervised Distillation** (`train-ss-distill`): DINOv2 → DeiT
   - Token representation loss (L_tok) + Token correlation loss (L_rel)
   - Avoids "negative transfer" from weak CNN teachers

## Configuration System

All configs in `configs/` directory follow this structure:

```yaml
data:
  dataset: mnist|cifar
  batch_size: int
  augmentation: {random_crop, random_flip, auto_augment, cutmix, mixup, ...}

model:
  model_type: adaptive_cnn|deit
  use_se: bool  # Squeeze-and-Excitation blocks

training:
  optimizer: adamw|sgd|adam
  scheduler: cosine|step|plateau
  use_amp: bool
  use_swa: bool
  use_compile: bool  # torch.compile (PyTorch 2.0+)
  use_bf16: bool     # H100 optimization

vit:  # For DeiT models
  variant: tiny|small|base
  distillation: bool

distillation:  # For CNN→ViT
  teacher_checkpoint: path
  distillation_type: hard|soft

ss_distillation:  # For DINOv2→ViT
  teacher_type: dinov2
  lambda_tok: float
  lambda_rel: float
```

## Key Implementation Details

- **Batch size in DDP**: Config batch_size is per-GPU; effective batch = per-GPU × num-GPUs
- **ViT inference modes**: 'cls' (class token), 'dist' (distillation token), 'avg' (average pooling)
- **SWA activation**: Starts at 75% of training by default
- **Early stopping**: Patience=15 epochs by default
- **Dataset auto-download**: MNIST/CIFAR-10 downloaded to `./data/` via torchvision

### Checkpoint Format

Saved checkpoints contain:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `best_val_acc`: Best validation accuracy
- `epoch`: Current epoch number
- `config`: Training configuration

To load a checkpoint for evaluation:
```python
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
```
