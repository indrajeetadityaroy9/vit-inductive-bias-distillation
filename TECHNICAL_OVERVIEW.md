# Dataset-Adaptive Image Classification: Technical Overview

This document provides a comprehensive technical reference for understanding the architecture, algorithms, and design decisions in this project.

## 1. Project Context and Motivation

This project addresses a fundamental challenge in deep learning: **how to efficiently train image classifiers on small-scale datasets** (MNIST, CIFAR-10) while leveraging modern techniques like Vision Transformers and knowledge distillation.

### Key Research Questions Addressed

1. Can CNNs with attention mechanisms achieve near-optimal performance on small datasets?
2. Does knowledge distillation always improve student models?
3. How can we avoid "negative transfer" when the teacher is weaker than what the student could achieve independently?

### Core Finding: Negative Transfer Effect

The project demonstrates that **CNN-distilled DeiT underperforms ViT without distillation**:

```
ViT (86.02%) > DeiT-CNN (84.39%) > CNN Teacher (82.90%)
```

This occurs because the teacher constrains the student to its suboptimal decision boundary.

**Solution**: Self-supervised distillation from DINOv2 achieves **89.18%** by transferring relational structure rather than classification logits.

---

## 2. Model Architectures

### 2.1 AdaptiveCNN (Teacher Model)

**Design Philosophy**: Dataset-specific architecture that adapts depth and width based on task complexity.

#### Squeeze-and-Excitation (SE) Block

Channel attention mechanism that recalibrates feature responses:

```
SE(x) = x · σ(W₂ · ReLU(W₁ · GAP(x)))
```

Where:
- `GAP(x)` = Global Average Pooling: `(1/HW) Σᵢⱼ x_{c,i,j}`
- `W₁ ∈ ℝ^{C/r × C}` (reduction ratio r=16)
- `W₂ ∈ ℝ^{C × C/r}`
- `σ` = Sigmoid activation

**Implementation** (`src/models.py:10-26`):

```python
y = self.avg_pool(x).view(b, c)           # (B, C, H, W) → (B, C)
y = self.fc(y).view(b, c, 1, 1)           # (B, C) → (B, C, 1, 1)
return x * y.expand_as(x)                  # Channel-wise scaling
```

#### Residual Block with SE

```
ResBlock(x) = ReLU(SE(BN(Conv(ReLU(BN(Conv(x)))))) + Shortcut(x))
```

#### MNIST Architecture (709K parameters)

| Stage | Channels | Resolution | Blocks |
|-------|----------|------------|--------|
| Initial | 1→32 | 28→14 | Conv+MaxPool |
| Stage 1 | 32 | 14×14 | 2 ResBlocks |
| Stage 2 | 64 | 7×7 | 2 ResBlocks |
| Stage 3 | 128 | 4×4 | 2 ResBlocks |
| Classifier | 128→64→10 | - | FC |

#### CIFAR-10 Architecture (17.6M parameters)

| Stage | Channels | Resolution | Blocks |
|-------|----------|------------|--------|
| Initial | 3→64 | 32×32 | Conv |
| Stage 1 | 64 | 32×32 | 2 ResBlocks |
| Stage 2 | 128 | 16×16 | 3 ResBlocks |
| Stage 3 | 256 | 8×8 | 3 ResBlocks |
| Stage 4 | 512 | 4×4 | 3 ResBlocks |
| Classifier | 512→256→10 | - | FC |

---

### 2.2 Vision Transformer (DeiT)

**Design Philosophy**: Lightweight ViT for small images with optional knowledge distillation.

#### Patch Embedding

Converts image into sequence of patch tokens:

```
PatchEmbed(x) = Flatten(Conv2d(x, kernel=P, stride=P))
```

For CIFAR-10 (32×32, patch_size=4):
- Number of patches: `N = (32/4)² = 64`
- Each patch: `4×4×3 = 48` pixels → projected to `d=192` dimensions

**Hybrid Patch Embedding** (optional conv stem):

```
HybridPatchEmbed(x) = PatchProj(GELU(BN(Conv₂(GELU(BN(Conv₁(x)))))))
```

Adds locality bias before patch projection.

#### Positional Embedding

Learnable embeddings added to patch tokens:

```
z₀ = [x_cls; x_dist; E·x_patches] + E_pos
```

Where:
- `x_cls ∈ ℝ^d` = Class token (learnable)
- `x_dist ∈ ℝ^d` = Distillation token (learnable, optional)
- `E ∈ ℝ^{d×P²C}` = Patch projection
- `E_pos ∈ ℝ^{(N+2)×d}` = Positional embeddings

**Positional Interpolation** (`src/vit.py:27-75`):

For different resolutions, interpolates positional embeddings using bicubic interpolation:

```python
patch_pos = F.interpolate(patch_pos, size=(H_target, W_target), mode='bicubic')
```

#### Multi-Head Self-Attention (MHSA)

```
MHSA(z) = Concat(head₁, ..., headₕ)W^O

headᵢ = Attention(zW^Q_i, zW^K_i, zW^V_i)

Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Scaled Dot-Product Attention** (`src/vit.py:206-216`):

```python
if HAS_SDPA:
    # Flash Attention v2 on H100 (25-40% faster, 50% less memory)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
else:
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    x = attn @ v
```

#### Stochastic Depth (DropPath)

Regularization that randomly drops entire residual branches:

```
DropPath(x) = x / keep_prob · Bernoulli(keep_prob)
```

Drop rates increase linearly with depth: `drop_path[i] = i/(L-1) × max_rate`

#### DeiT-Tiny Configuration

| Parameter | Value |
|-----------|-------|
| Embedding dim | 192 |
| Depth | 12 blocks |
| Heads | 3 |
| MLP ratio | 2.0 (code default: 4.0) |
| Patch size | 4 |
| Drop path rate | 0.1 |
| Parameters | ~4M |

---

## 3. Knowledge Distillation

### 3.1 Standard Distillation (CNN → DeiT)

**Two Distillation Modes** (`src/distillation.py:30-101`):

#### Hard Distillation

Student matches teacher's argmax predictions:

```
L_hard = (1-α)·CE(y_cls, y_true) + α·CE(y_dist, argmax(y_teacher))
```

#### Soft Distillation

Temperature-scaled KL divergence (Hinton et al.):

```
L_soft = (1-α)·CE(y_cls, y_true) + α·τ²·KL(σ(y_dist/τ), σ(y_teacher/τ))
```

Where:
- `α ∈ [0,1]` = Distillation weight
- `τ > 0` = Temperature (default: 3.0)
- `σ` = Softmax function

**Temperature Scaling Intuition**:
- Higher τ → softer probability distributions → more knowledge about inter-class relationships
- τ² scaling compensates for gradient magnitude reduction

#### Alpha Scheduling

Progressive distillation weight adjustment:

| Schedule | Formula |
|----------|---------|
| Constant | `α(t) = α` |
| Linear | `α(t) = α_start + (α_end - α_start) × t/T` |
| Cosine | `α(t) = α_start + (α_end - α_start) × (1 - cos(πt/T))/2` |

---

### 3.2 Self-Supervised Distillation (DINOv2 → DeiT)

**Design Philosophy**: Avoid negative transfer by distilling relational structure, not classification decisions.

#### Two-Stage Training

- **Stage A** (epochs 0-9): `L = L_CE + λ_tok·L_tok`
- **Stage B** (epochs 10+): `L = L_CE + λ_tok·L_tok + λ_rel·L_rel`

#### Token Representation Loss (L_tok)

Aligns intermediate token representations via learnable projectors:

```
L_tok = (1/L) Σₗ (1 - cos(P_s(z_s^l), P_t(z_t^l)))
```

Where:
- `z_s^l, z_t^l` = Student/teacher tokens at layer `l`
- `P_s, P_t` = Learnable projection heads (Linear→LN→GELU→Linear→LN)
- `cos` = Cosine similarity

**Projection Head Architecture** (`src/distillation.py:653-674`):

```python
self.proj = nn.Sequential(
    nn.Linear(in_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, out_dim),
    nn.LayerNorm(out_dim)
)
```

#### Token Correlation Loss (L_rel)

Matches token-token correlation matrices:

```
L_rel = KL(softmax(C_s/τ), softmax(C_t/τ))
```

Where correlation matrix `C`:

```
C = normalize(z) · normalize(z)^T
```

**Pooled Mode** (efficient, O(B²) instead of O(N²)):

```python
student_pooled = student_tokens.mean(dim=1)  # (B, N, D) → (B, D)
student_corr = student_norm @ student_norm.T  # (B, B)
```

#### Token Interpolation

Aligns teacher tokens (196 for DINOv2 @ 224×224) to student tokens (64 for DeiT @ 32×32):

```python
# (B, 196, D) → (B, 14, 14, D) → interpolate → (B, 8, 8, D) → (B, 64, D)
tokens = tokens.transpose(1, 2).reshape(B, D, H_t, H_t)
tokens = F.interpolate(tokens, size=(H_s, H_s), mode='bilinear')
tokens = tokens.reshape(B, D, -1).transpose(1, 2)
```

#### Loss Configuration

| Parameter | Value | Role |
|-----------|-------|------|
| λ_tok | 1.0 | Token representation weight |
| λ_rel | 0.1 | Token correlation weight |
| Projection dim | 256 | Common embedding space |
| Token layers | [6, 11] | Intermediate extraction points |
| rel_warmup_epochs | 10 | Delay L_rel activation |

---

## 4. Training Infrastructure

### 4.1 Distributed Data Parallel (DDP)

**Process Model** (`src/training.py:276-533`):
- Each GPU runs independent process with rank ∈ [0, world_size)
- NCCL backend for GPU-to-GPU communication
- `DistributedSampler` ensures non-overlapping data shards

**Metric Aggregation**:

```python
dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
avg_loss = loss_tensor.item() / (batch_count * world_size)
```

**Batch Size**: Config specifies per-GPU batch; effective batch = `batch_size × num_gpus`

### 4.2 Mixed Precision Training

**BF16 (H100 optimized)**:
- Same exponent range as FP32 → no GradScaler needed
- 2× memory savings, minimal accuracy impact

**FP16 (other GPUs)**:
- Requires GradScaler for numerical stability
- Loss scaling prevents underflow

```python
if config.training.use_bf16:
    self.autocast_kwargs = {'device_type': 'cuda', 'dtype': torch.bfloat16}
else:
    self.scaler = GradScaler()
    self.autocast_kwargs = {'device_type': 'cuda'}
```

### 4.3 Label Smoothing Cross-Entropy

Prevents overconfident predictions:

```
L_smooth = (1-ε)·CE(y, k) + ε·(1/K)·Σᵢ CE(y, i)
```

**Handles Soft Labels** (from MixUp/CutMix):

```python
if target.dim() > 1:  # Soft labels (B, K)
    loss = -(target * log_softmax(pred)).sum(dim=-1)
```

### 4.4 Stochastic Weight Averaging (SWA)

Averages weights in final 25% of training:

```
w_SWA = (1/n) Σᵢ wᵢ
```

Activates at epoch `⌊0.75 × num_epochs⌋`, uses reduced LR (default: 0.0005).

**Post-Training**: Updates BatchNorm statistics on full training set via `torch.optim.swa_utils.update_bn()`. See `src/training.py:600` and `src/distillation.py:554,1946`.

### 4.5 Gradient Clipping

Prevents exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 5. Data Augmentation

### 5.1 Cutout

Randomly masks rectangular regions:

```python
mask[y1:y2, x1:x2] = 0.0
img = img * mask
```

### 5.2 MixUp

Linear interpolation of images and labels:

```
x̃ = λ·x₁ + (1-λ)·x₂
ỹ = λ·y₁ + (1-λ)·y₂

λ ~ Beta(α, α)
```

### 5.3 CutMix

Rectangular patch mixing with area-proportional labels:

```
x̃[:, bbx1:bbx2, bby1:bby2] = x₂[:, bbx1:bbx2, bby1:bby2]
λ = 1 - (area_cut / area_total)
ỹ = λ·y₁ + (1-λ)·y₂
```

**Bounding Box Sampling**:

```python
cut_rat = np.sqrt(1 - lam)
cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
cx, cy = np.random.randint(W), np.random.randint(H)
```

### 5.4 AutoAugment

Learned augmentation policies (CIFAR-10 policy includes: ShearX/Y, TranslateX/Y, Rotate, AutoContrast, Equalize, Solarize, Posterize, etc.)

### 5.5 RandAugment

Random N operations with magnitude M:

```python
transforms.RandAugment(num_ops=2, magnitude=9)
```

---

## 6. Optimization Strategies

### 6.1 Optimizers

| Optimizer | Use Case | Key Features |
|-----------|----------|--------------|
| AdamW | MNIST, ViT | Decoupled weight decay |
| SGD+Nesterov | CIFAR CNN | Momentum with lookahead |

**Fused Optimizers** (PyTorch 2.0+, H100): Single CUDA kernel for 10-30% speedup.

### 6.2 Learning Rate Schedules

**Cosine Annealing**:

```
η(t) = η_min + (η_max - η_min) × (1 + cos(πt/T)) / 2
```

**Warmup** (first N epochs):

```
η(t) = η_max × (t+1) / warmup_epochs
```

**Implementation**: Warmup is applied at the training loop level via `GradualWarmupScheduler` or manual LR adjustment in epoch callbacks. See `src/training.py` scheduler integration and distillation trainer epoch loops.

### 6.3 Gradient Accumulation

Simulates larger batch sizes:

```python
loss = loss / grad_accum_steps
loss.backward()
if (batch_idx + 1) % grad_accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

## 7. H100-Specific Optimizations

| Optimization | Speedup | Memory Savings |
|--------------|---------|----------------|
| BF16 (no scaler) | 25-40% | 50% |
| torch.compile | 30-50% | - |
| Fused optimizers | 10-30% | - |
| Flash Attention v2 | 25-40% | 50% |
| TF32 matmul | Up to 8× | - |

**torch.compile Modes**:
- `max-autotune`: Best runtime, slowest compilation
- `reduce-overhead`: Balanced
- `default`: Fastest compilation

---

## 8. Experimental Results Summary

| Model | MNIST | CIFAR-10 | Parameters |
|-------|-------|----------|------------|
| AdaptiveCNN | 99.08% | 82.90% | 709K / 17.6M |
| DeiT (CNN distilled) | 99.54% | 84.39% | 4M |
| ViT (no distillation) | 99.64% | 86.02% | 4M |
| DeiT (CST-SSL) | - | **89.18%** | 4M |

**Key Insight**: Self-supervised distillation from DINOv2 outperforms both CNN distillation (+4.79%) and standalone ViT (+3.16%) by transferring rich feature structure without task-specific bias.

---

## 9. File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 977 | CLI entry point, unified DDP training orchestration |
| `src/config.py` | 439 | Configuration classes, YAML loading |
| `src/models.py` | 253 | AdaptiveCNN, SEBlock, ResidualBlock, ModelFactory |
| `src/vit.py` | 715 | DeiT, PatchEmbed, MHSA, TransformerBlock |
| `src/training.py` | 605 | Trainer, DDPTrainer, EarlyStopping; utilities: `build_optimizer`, `build_scheduler`, `build_checkpoint_dict` |
| `src/distillation.py` | 1951 | DistillationTrainer, SelfSupervisedDistillationTrainer, CKALoss, GramMatrixLoss |
| `src/datasets.py` | 585 | DatasetManager, Cutout, MixingDataset (unified MixUp/CutMix) |
| `src/evaluation.py` | 326 | ModelEvaluator, TestTimeAugmentation |
| `src/visualization.py` | 305 | GradCAM, FeatureMapVisualizer, TrainingVisualizer |
| `src/analytics.py` | 1099 | HessianAnalyzer, CKAAnalyzer, AttentionDistanceAnalyzer, AnalyticsVisualizer |
| `src/teachers.py` | 309 | ResNet18CIFAR, ConvNeXtV2Tiny teacher models |

---

## 10. References

1. **DeiT**: Touvron et al., "Training data-efficient image transformers & distillation through attention" (2021)
2. **SE-Net**: Hu et al., "Squeeze-and-Excitation Networks" (2018)
3. **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
4. **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (2023)
5. **CutMix**: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers" (2019)
6. **MixUp**: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018)
7. **AutoAugment**: Cubuk et al., "AutoAugment: Learning Augmentation Strategies from Data" (2019)
8. **Stochastic Depth**: Huang et al., "Deep Networks with Stochastic Depth" (2016)
