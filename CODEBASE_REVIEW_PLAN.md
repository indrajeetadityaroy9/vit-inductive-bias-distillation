# Codebase Review Plan

## Objective
Systematically verify the implementation against TECHNICAL_OVERVIEW.md and CLAUDE.md specifications, identifying integration gaps, interface mismatches, and deviations.

---

## Stage 1: Model Architecture Verification

### 1.1 AdaptiveCNN (src/models.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 2.1

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| SE Block formula: `SE(x) = x * sigmoid(W2 * ReLU(W1 * GAP(x)))` | Lines 39-46 | |
| Reduction ratio r=16 | Line 45 | |
| Residual Block formula: `ReLU(SE(BN(Conv(BN(Conv(x))))) + Shortcut(x))` | Lines 57-61 | |
| MNIST: 6 blocks, 709K params, channels 32→64→128 | Lines 63-71 | |
| CIFAR-10: 11 blocks, 17.6M params, channels 64→128→256→512 | Lines 73-82 | |

### 1.2 DeiT/ViT (src/vit.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 2.2

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| PatchEmbed: Conv2d(kernel=P, stride=P), flatten | Lines 90-100 | |
| Hybrid patch embed (conv stem): Conv1→BN→GELU→Conv2→BN→GELU→Proj | Lines 102-108 | |
| Positional embedding: `z₀ = [x_cls; x_dist; E·x_patches] + E_pos` | Lines 110-122 | |
| Positional interpolation via bicubic | Lines 124-130 | |
| MHSA with Flash Attention v2 fallback | Lines 142-151 | |
| DropPath linear scaling: `drop_path[i] = i/(L-1) * max_rate` | Lines 160-162 | |
| DeiT-Tiny defaults: embed_dim=192, depth=12, heads=3, mlp_ratio=4.0 | Lines 166-174 | |

---

## Stage 2: Knowledge Distillation Verification

### 2.1 Standard Distillation (src/distillation.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 3.1

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| Hard distillation: `L = (1-α)*CE(y_cls, y_true) + α*CE(y_dist, argmax(y_teacher))` | Lines 184-189 | |
| Soft distillation: `L = (1-α)*CE + α*τ²*KL(σ(y_dist/τ), σ(y_teacher/τ))` | Lines 191-197 | |
| Temperature default: τ=3.0 | Line 202 | |
| Alpha scheduling: constant, linear, cosine | Lines 209-217 | |

### 2.2 Self-Supervised Distillation (src/distillation.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 3.2

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| Two-stage training: Stage A (CE+L_tok), Stage B (CE+L_tok+L_rel) | Lines 225-228 | |
| Token representation loss: `L_tok = (1/L) Σ (1 - cos(P_s(z_s^l), P_t(z_t^l)))` | Lines 230-241 | |
| Projection head: Linear→LN→GELU→Linear→LN | Lines 243-252 | |
| Token correlation loss: `L_rel = KL(softmax(C_s/τ), softmax(C_t/τ))` | Lines 255-261 | |
| Pooled mode for O(B²) efficiency | Lines 269-274 | |
| Token interpolation: 196→64 via bilinear | Lines 276-285 | |
| Default params: λ_tok=1.0, λ_rel=0.1, proj_dim=256, layers=[6,11], warmup=10 | Lines 289-295 | |

---

## Stage 3: Training Infrastructure Verification

### 3.1 DDP (src/training.py, main.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 4.1

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| NCCL backend initialization | Lines 304-306 | |
| DistributedSampler for non-overlapping shards | Line 306 | |
| Metric aggregation: `dist.all_reduce(loss, op=ReduceOp.SUM)` | Lines 308-312 | |
| Effective batch = batch_size × num_gpus | Line 315 | |

### 3.2 Mixed Precision (src/training.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 4.2

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| BF16: no GradScaler, dtype=torch.bfloat16 | Lines 319-321, 328-329 | |
| FP16: GradScaler required | Lines 323-325, 330-332 | |

### 3.3 Label Smoothing (src/training.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 4.3

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| Formula: `L = (1-ε)*CE(y,k) + ε*(1/K)*Σ CE(y,i)` | Lines 337-341 | |
| Soft label handling: `-(target * log_softmax(pred)).sum(dim=-1)` | Lines 343-347 | |

### 3.4 SWA (src/training.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 4.4

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| Activates at 75% of training | Line 358 | |
| Post-training BatchNorm update | Line 360 | |

### 3.5 Gradient Clipping (src/training.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 4.5

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| clip_grad_norm with max_norm=1.0 default | Lines 362-368 | |

---

## Stage 4: Data Augmentation Verification

### 4.1 Cutout (src/datasets.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 5.1

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| Mask rectangular region: `mask[y1:y2, x1:x2] = 0.0` | Lines 376-380 | |

### 4.2 MixUp (src/datasets.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 5.2

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| Linear interpolation: `x̃ = λ*x₁ + (1-λ)*x₂` | Lines 385-388 | |
| Lambda from Beta(α, α) | Line 391 | |

### 4.3 CutMix (src/datasets.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 5.3

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| Patch replacement: `x̃[:, bbx1:bbx2, bby1:bby2] = x₂[...]` | Lines 396-400 | |
| Area-proportional lambda: `λ = 1 - (area_cut / area_total)` | Line 400 | |
| Bounding box: `cut_rat = sqrt(1-λ)` | Lines 404-410 | |

### 4.4 AutoAugment/RandAugment (src/datasets.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Sections 5.4-5.5

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| AutoAugment CIFAR-10 policy | Lines 412-414 | |
| RandAugment(num_ops=2, magnitude=9) | Lines 416-422 | |

---

## Stage 5: Optimization Verification

### 5.1 Optimizers (src/training.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 6.1

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| AdamW for MNIST/ViT with decoupled weight decay | Lines 430-432 | |
| SGD+Nesterov for CIFAR CNN | Line 433 | |
| Fused optimizers on H100 | Line 435 | |

### 5.2 Learning Rate Schedules (src/training.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 6.2

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| Cosine annealing: `η = η_min + (η_max - η_min) * (1 + cos(πt/T)) / 2` | Lines 437-443 | |
| Linear warmup: `η = η_max * (t+1) / warmup_epochs` | Lines 445-449 | |

### 5.3 Gradient Accumulation (src/training.py)
**Specification Reference**: TECHNICAL_OVERVIEW.md Section 6.3

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| Loss scaling: `loss = loss / grad_accum_steps` | Lines 451-461 | |
| Step on accumulation boundary | Lines 458-460 | |

---

## Stage 6: H100 Optimizations Verification

**Specification Reference**: TECHNICAL_OVERVIEW.md Section 7

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| BF16 mode without GradScaler | Lines 467-469 | |
| torch.compile with mode selection | Lines 470, 475-478 | |
| Flash Attention v2 via F.scaled_dot_product_attention | Lines 471-472 | |
| TF32 matmul enablement | Line 473 | |

---

## Stage 7: Configuration System Verification

**Specification Reference**: CLAUDE.md Section "Configuration System"

| Verification Item | Spec Reference | Check |
|-------------------|----------------|-------|
| DataConfig with dataset, batch_size, augmentation | Lines 123-126 | |
| ModelConfig with model_type, use_se | Lines 128-130 | |
| TrainingConfig with optimizer, scheduler, use_amp, use_swa, use_compile, use_bf16 | Lines 132-138 | |
| ViTConfig with variant, distillation | Lines 140-142 | |
| DistillationConfig with teacher_checkpoint, distillation_type | Lines 144-146 | |
| SelfSupervisedDistillationConfig with teacher_type, lambda_tok, lambda_rel | Lines 148-151 | |

---

## Stage 8: File Reference Accuracy

**Specification Reference**: TECHNICAL_OVERVIEW.md Section 9

| File | Spec Lines | Actual Lines | Delta | Status |
|------|------------|--------------|-------|--------|
| main.py | 1094 | TBD | | |
| src/config.py | 393 | TBD | | |
| src/models.py | 231 | TBD | | |
| src/vit.py | 663 | TBD | | |
| src/training.py | 534 | TBD | | |
| src/distillation.py | 1342 | TBD | | |
| src/datasets.py | 386 | TBD | | |
| src/evaluation.py | 326 | TBD | | |
| src/visualization.py | 305 | TBD | | |

---

## Stage 9: Integration Points Verification

### 9.1 Model ↔ Training Integration
- DDPTrainer correctly wraps models with DistributedDataParallel
- Loss functions handle both hard labels and soft labels (from MixUp/CutMix)
- Checkpoint save/load maintains all necessary state

### 9.2 Config ↔ Implementation Integration
- All config fields are actually used in code
- No hardcoded values that should be configurable
- Config validation catches invalid combinations

### 9.3 Distillation ↔ Teacher Model Integration
- Teacher model loading and freezing
- Feature extraction at correct layers
- Token dimension alignment (teacher 196 → student 64)

### 9.4 Data Pipeline ↔ Training Integration
- DistributedSampler correctly shuffles per-epoch
- Augmentation applied at correct stage (before/after tensor conversion)
- Dual-augment dataset for SS distillation

---

## Execution Plan

| Phase | Stages | Scope |
|-------|--------|-------|
| **Phase 1** | 1-2 | Model architectures and distillation |
| **Phase 2** | 3-5 | Training infrastructure and optimization |
| **Phase 3** | 6-7 | H100 optimizations and config system |
| **Phase 4** | 8-9 | File references and integration points |

---

## Review Output Format

For each verification item, document:
1. **Status**: PASS / FAIL / DEVIATION / MISSING
2. **Location**: File:line_number
3. **Finding**: Description of issue (if any)
4. **Severity**: Critical / High / Medium / Low
5. **Recommendation**: Suggested fix

