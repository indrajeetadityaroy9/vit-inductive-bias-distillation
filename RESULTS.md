# Mitigating Inductive Bias Mismatch in Heterogeneous Knowledge Distillation

## Research Question
Does the "negative transfer" observed when distilling CNNs into Vision Transformers stem from conflicting inductive biases? Can structural distillation (CKA loss) from a self-supervised teacher (DINOv2) outperform traditional soft-label distillation?

## Experimental Matrix

| Experiment | Teacher Architecture | Distillation Type | Teacher Acc | Student Acc | Δ vs Baseline | Inductive Bias (Attn Dist) | Gen. Gap (Hessian Trace) |
|:-----------|:---------------------|:------------------|:------------|:------------|:--------------|:---------------------------|:-------------------------|
| **Baseline** | None | Vanilla CE | N/A | ~86.02% | — | *Baseline* | *Baseline* |
| **EXP-1** | ResNet-18 | Soft KL (τ=4) | *pending* | *pending* | *expected: <0%* | *Short (expected)* | *High (expected)* |
| **EXP-2** | ConvNeXt V2 | Soft KL (τ=4) | *pending* | *pending* | *expected: >EXP-1* | *Medium* | *Medium* |
| **EXP-3** | DINOv2-S/14 | **Structural (CKA)** | N/A | *pending* | *expected: +3%* | *Long (expected)* | *Low (expected)* |

## Phase 1: Teacher Qualification

### ResNet-18 (Classic CNN)
- **Target:** >94% validation accuracy
- **Architecture:** Modified stem (3x3 conv, stride 1, no max pool)
- **Training:** SGD + Cosine LR, 200 epochs, CutMix augmentation

| Metric | Value |
|--------|-------|
| Final Val Accuracy | *pending* |
| Best Epoch | *pending* |
| Training Time | *pending* |

### ConvNeXt V2-Tiny (Modern CNN Bridge)
- **Target:** >95% validation accuracy
- **Architecture:** ImageNet pretrained, reinitialized stem (2x2, stride 2)
- **Training:** AdamW + Cosine LR, 100 epochs, fine-tuning

| Metric | Value |
|--------|-------|
| Final Val Accuracy | *pending* |
| Best Epoch | *pending* |
| Training Time | *pending* |

## Phase 2: Distillation Results

### EXP-1: DeiT + ResNet-18 (Soft KL Distillation)
**Hypothesis:** Classic CNN inductive biases (local receptive fields, translation equivariance) may conflict with ViT's global attention, causing negative transfer.

| Metric | Value |
|--------|-------|
| Student Val Accuracy | *pending* |
| Δ vs Baseline | *pending* |
| Mean Attention Distance | *pending* |
| Hessian Trace | *pending* |

### EXP-2: DeiT + ConvNeXt V2 (Soft KL Distillation)
**Hypothesis:** ConvNeXt's "Transformer-like" macro design (large kernels, LayerNorm, inverted bottlenecks) should transfer better to ViT.

| Metric | Value |
|--------|-------|
| Student Val Accuracy | *pending* |
| Δ vs Baseline | *pending* |
| Δ vs EXP-1 | *pending* |
| Mean Attention Distance | *pending* |
| Hessian Trace | *pending* |

### EXP-3: DeiT + DINOv2 (CKA Structural Distillation)
**Hypothesis:** Structural alignment via CKA loss transfers semantic representations without imposing conflicting inductive biases.

| Metric | Value |
|--------|-------|
| Student Val Accuracy | *pending* |
| Δ vs Baseline | *pending* |
| Δ vs EXP-1 | *pending* |
| Δ vs EXP-2 | *pending* |
| Mean Attention Distance | *pending* |
| Hessian Trace | *pending* |

## Phase 3: Analytics Summary

### CKA Heatmaps
*Location: `outputs/analytics/cka_heatmaps/`*

| Comparison | Diagonal Alignment | Notes |
|------------|-------------------|-------|
| EXP-1 Student vs ResNet-18 | *pending* | Expected: Weak diagonal |
| EXP-2 Student vs ConvNeXt V2 | *pending* | Expected: Cleaner diagonal |
| EXP-3 Student vs DINOv2 | *pending* | Expected: Strong diagonal |

### Hessian Analysis (Loss Landscape Geometry)
*Higher trace = sharper minima = worse generalization*

| Model | Hessian Trace | Top Eigenvalue | Interpretation |
|-------|---------------|----------------|----------------|
| Baseline DeiT | *pending* | *pending* | Reference |
| EXP-1 Student | *pending* | *pending* | Expected: High (sharp) |
| EXP-2 Student | *pending* | *pending* | Expected: Medium |
| EXP-3 Student | *pending* | *pending* | Expected: Low (flat) |

### Mean Attention Distance
*Longer distance = more global attention patterns*

| Model | Layer 3 | Layer 6 | Layer 9 | Layer 11 |
|-------|---------|---------|---------|----------|
| Baseline DeiT | *pending* | *pending* | *pending* | *pending* |
| EXP-1 Student | *pending* | *pending* | *pending* | *pending* |
| EXP-2 Student | *pending* | *pending* | *pending* | *pending* |
| EXP-3 Student | *pending* | *pending* | *pending* | *pending* |

## Key Findings

### Primary Result
*To be filled after experiments complete*

### Supporting Evidence
1. **Accuracy Ranking:** *pending*
2. **Attention Pattern Analysis:** *pending*
3. **Loss Landscape Geometry:** *pending*

## Conclusions
*To be written after analysis*

---

## Experimental Setup

### Hardware
- 2x NVIDIA H100 80GB HBM3
- Intel Xeon Platinum 8480+ (52 cores)
- 442GB System RAM

### Software
- PyTorch 2.x with torch.compile
- BF16 mixed precision (native H100 support)
- DDP for multi-GPU training

### Hyperparameters
| Parameter | EXP-1/2 | EXP-3 |
|-----------|---------|-------|
| Batch Size (per GPU) | 1024 | 512 |
| Learning Rate | 5e-4 | 5e-4 |
| Epochs | 200 | 200 |
| Weight Decay | 0.05 | 0.05 |
| Distillation α | 0.5 | N/A |
| Temperature τ | 4.0 | N/A |
| CKA λ | N/A | 0.5 |
| Token Loss λ | N/A | 0.5 |

---
*Generated: 2025-12-30*
*Status: Phase 1 in progress*
