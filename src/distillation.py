"""
Knowledge Distillation for DeiT (Data-efficient Image Transformer).

Implements distillation training where a Vision Transformer (student) learns from
a pre-trained CNN (teacher) using a distillation token.

Based on: "Training data-efficient image transformers & distillation through attention"
(Touvron et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import logging
from pathlib import Path
from tqdm import tqdm
import time
from collections import defaultdict

from src.training import DDPTrainer, LabelSmoothingCrossEntropy, build_checkpoint_dict
from src.config import (
    Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig,
    ViTConfig, DistillationConfig
)

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Combined loss for DeiT distillation training.

    Supports two distillation modes:
    1. Hard distillation (default): Uses argmax of teacher predictions
       loss = (1-alpha)*CE(cls_out, targets) + alpha*CE(dist_out, argmax(teacher))

    2. Soft distillation: Uses temperature-scaled KL divergence
       loss = (1-alpha)*CE(cls_out, targets) + alpha*tau^2*KL(dist_out/tau, teacher/tau)

    Args:
        base_criterion: Loss function for ground truth (e.g., LabelSmoothingCrossEntropy)
        distillation_type: 'hard' or 'soft'
        alpha: Weight for distillation loss (0 = no distillation, 1 = only distillation)
        tau: Temperature for soft distillation
    """

    def __init__(self, base_criterion, distillation_type='hard', alpha=0.5, tau=3.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.distillation_type = distillation_type

        # Validate alpha is in [0, 1]
        if not 0 <= alpha <= 1:
            raise ValueError(f"Distillation alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha

        # Validate tau is positive (for soft distillation)
        if tau <= 0:
            raise ValueError(f"Distillation tau must be positive, got {tau}")
        self.tau = tau

    def forward(self, student_cls_output, student_dist_output, targets, teacher_output):
        """
        Compute distillation loss.

        Args:
            student_cls_output: Student [CLS] token predictions (B, num_classes)
            student_dist_output: Student [DIST] token predictions (B, num_classes)
            targets: Ground truth labels (B,) or (B, num_classes) for soft labels
            teacher_output: Teacher model predictions (B, num_classes)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Ground truth loss on [CLS] token
        cls_loss = self.base_criterion(student_cls_output, targets)

        if self.distillation_type == 'hard':
            # Hard labels from teacher (argmax)
            teacher_labels = teacher_output.argmax(dim=1)
            dist_loss = F.cross_entropy(student_dist_output, teacher_labels)
        else:
            # Soft distillation with temperature scaling
            soft_teacher = F.softmax(teacher_output / self.tau, dim=1)
            soft_student = F.log_softmax(student_dist_output / self.tau, dim=1)
            dist_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
            # Scale by tau^2 as per Hinton et al.
            dist_loss = dist_loss * (self.tau ** 2)

        # Combined loss
        total_loss = (1 - self.alpha) * cls_loss + self.alpha * dist_loss

        loss_dict = {
            'cls_loss': cls_loss.item(),
            'dist_loss': dist_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


class DistillationTrainer(DDPTrainer):
    """
    DDP Trainer for knowledge distillation with DeiT.

    Extends DDPTrainer with distillation-specific functionality:
    - Manages both student (DeiT) and teacher (CNN) models
    - Uses DistillationLoss for combined training
    - Tracks additional metrics: cls_loss, dist_loss, agreement_rate
    - Supports distillation warmup (no distillation for first N epochs)

    Args:
        student_model: DeiT model wrapped with DDP
        teacher_model: Pre-trained CNN teacher model (frozen)
        config: Training configuration
        device: CUDA device
        rank: DDP process rank
        world_size: Number of DDP processes
    """

    def __init__(self, student_model, teacher_model, config, device, rank, world_size):
        # Initialize parent with student model
        super().__init__(student_model, config, device, rank, world_size)

        # Store teacher model (frozen, eval mode)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Get distillation config
        distill_config = config.distillation
        self.distillation_type = distill_config.distillation_type
        self.distillation_alpha = distill_config.alpha
        self.distillation_tau = distill_config.tau
        self.distillation_warmup_epochs = distill_config.distillation_warmup_epochs

        # Alpha scheduling parameters
        self.alpha_schedule = getattr(distill_config, 'alpha_schedule', 'constant')
        self.alpha_start = getattr(distill_config, 'alpha_start', 0.0)
        self.alpha_end = getattr(distill_config, 'alpha_end', distill_config.alpha)
        self.num_epochs = config.training.num_epochs

        # Create distillation loss (alpha will be updated per epoch for scheduling)
        self.distillation_criterion = DistillationLoss(
            base_criterion=self.criterion,
            distillation_type=self.distillation_type,
            alpha=self.distillation_alpha,
            tau=self.distillation_tau
        )

        # Additional metrics tracking
        self.distillation_metrics = defaultdict(list)

        if self.is_main_process:
            logger.info(f"Distillation Trainer initialized:")
            logger.info(f"  - Type: {self.distillation_type}")
            logger.info(f"  - Alpha: {self.distillation_alpha}")
            logger.info(f"  - Alpha schedule: {self.alpha_schedule}")
            if self.alpha_schedule != 'constant':
                logger.info(f"  - Alpha range: {self.alpha_start} -> {self.alpha_end}")
            logger.info(f"  - Tau: {self.distillation_tau}")
            logger.info(f"  - Warmup epochs: {self.distillation_warmup_epochs}")

    def get_scheduled_alpha(self, epoch):
        """
        Get the scheduled alpha value for the current epoch.

        Supports 'constant', 'linear', and 'cosine' schedules.
        Alpha scheduling starts after warmup epochs.
        """
        if self.alpha_schedule == 'constant':
            return self.distillation_alpha

        # Calculate progress excluding warmup epochs
        effective_epoch = max(0, epoch - self.distillation_warmup_epochs)
        effective_total = max(1, self.num_epochs - self.distillation_warmup_epochs)
        progress = min(1.0, effective_epoch / effective_total)

        if self.alpha_schedule == 'linear':
            return self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        elif self.alpha_schedule == 'cosine':
            # Cosine annealing from alpha_start to alpha_end
            import math
            return self.alpha_start + (self.alpha_end - self.alpha_start) * (1 - math.cos(progress * math.pi)) / 2
        else:
            return self.distillation_alpha

    def train_epoch_ddp(self, train_loader, train_sampler):
        """
        Train one epoch with distillation.

        Handles dual outputs from DeiT (cls_logits, dist_logits) and
        computes distillation loss from frozen teacher.
        """
        train_sampler.set_epoch(self.current_epoch)

        self.ddp_model.train()
        self.teacher_model.eval()  # Ensure teacher stays in eval mode

        total_loss = 0
        total_cls_loss = 0
        total_dist_loss = 0
        correct = 0
        total = 0
        agreement_total = 0
        batch_count = 0

        # Check if we're in warmup phase (no distillation)
        in_warmup = self.current_epoch < self.distillation_warmup_epochs

        # Update alpha based on schedule (after warmup)
        if not in_warmup:
            current_alpha = self.get_scheduled_alpha(self.current_epoch)
            self.distillation_criterion.alpha = current_alpha
        else:
            current_alpha = 0.0

        if self.is_main_process:
            desc = f"Epoch {self.current_epoch + 1}"
            if in_warmup:
                desc += " [Warmup - No Distillation]"
            elif self.alpha_schedule != 'constant':
                desc += f" [α={current_alpha:.3f}]"
            pbar = tqdm(train_loader, desc=desc)
        else:
            pbar = train_loader

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.use_amp:
                with autocast(**self.autocast_kwargs):
                    # Student forward pass - returns (cls_logits, dist_logits) during training
                    student_output = self.ddp_model(inputs)

                    if isinstance(student_output, tuple):
                        cls_output, dist_output = student_output
                    else:
                        # Fallback if model returns single output
                        cls_output = student_output
                        dist_output = student_output

                    # Teacher forward pass (no grad)
                    with torch.no_grad():
                        teacher_output = self.teacher_model(inputs)

                    if in_warmup:
                        # Warmup: only use classification loss
                        loss = self.criterion(cls_output, targets)
                        cls_loss_val = loss.item()
                        dist_loss_val = 0.0
                    else:
                        # Full distillation loss
                        loss, loss_dict = self.distillation_criterion(
                            cls_output, dist_output, targets, teacher_output
                        )
                        cls_loss_val = loss_dict['cls_loss']
                        dist_loss_val = loss_dict['dist_loss']

                    loss = loss / self.grad_accum_steps

                if self.scaler is not None:
                    # FP16 with GradScaler
                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        if self.config.training.gradient_clip_val > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.ddp_model.parameters(),
                                self.config.training.gradient_clip_val
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    # BF16 without GradScaler
                    loss.backward()

                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        if self.config.training.gradient_clip_val > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.ddp_model.parameters(),
                                self.config.training.gradient_clip_val
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            else:
                # Non-AMP path
                student_output = self.ddp_model(inputs)

                if isinstance(student_output, tuple):
                    cls_output, dist_output = student_output
                else:
                    cls_output = student_output
                    dist_output = student_output

                with torch.no_grad():
                    teacher_output = self.teacher_model(inputs)

                if in_warmup:
                    loss = self.criterion(cls_output, targets)
                    cls_loss_val = loss.item()
                    dist_loss_val = 0.0
                else:
                    loss, loss_dict = self.distillation_criterion(
                        cls_output, dist_output, targets, teacher_output
                    )
                    cls_loss_val = loss_dict['cls_loss']
                    dist_loss_val = loss_dict['dist_loss']

                loss = loss / self.grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.config.training.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.ddp_model.parameters(),
                            self.config.training.gradient_clip_val
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.grad_accum_steps
            total_cls_loss += cls_loss_val
            total_dist_loss += dist_loss_val

            # Accuracy based on cls head
            _, predicted = cls_output.max(1)

            # Handle one-hot encoded targets
            if len(targets.shape) > 1:
                targets_idx = targets.argmax(1)
            else:
                targets_idx = targets

            correct += predicted.eq(targets_idx).sum().item()
            total += targets_idx.size(0)

            # Agreement rate: % where student distillation head agrees with teacher
            if not in_warmup:
                teacher_preds = teacher_output.argmax(dim=1)
                student_dist_preds = dist_output.argmax(dim=1)
                agreement_total += (student_dist_preds == teacher_preds).sum().item()

            batch_count += 1

            if self.is_main_process and hasattr(pbar, 'set_postfix'):
                postfix = {
                    'Loss': f'{total_loss/batch_count:.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                }
                if not in_warmup:
                    postfix['Agree'] = f'{100.*agreement_total/total:.1f}%'
                pbar.set_postfix(postfix)

            self.global_step += 1

        # Aggregate metrics across all GPUs
        loss_tensor = torch.tensor([total_loss], device=self.device)
        cls_loss_tensor = torch.tensor([total_cls_loss], device=self.device)
        dist_loss_tensor = torch.tensor([total_dist_loss], device=self.device)
        correct_tensor = torch.tensor([correct], device=self.device)
        total_tensor = torch.tensor([total], device=self.device)
        agreement_tensor = torch.tensor([agreement_total], device=self.device)

        self.dist.all_reduce(loss_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(cls_loss_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(dist_loss_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(correct_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(total_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(agreement_tensor, op=self.dist.ReduceOp.SUM)

        avg_loss = loss_tensor.item() / (batch_count * self.world_size)
        avg_cls_loss = cls_loss_tensor.item() / (batch_count * self.world_size)
        avg_dist_loss = dist_loss_tensor.item() / (batch_count * self.world_size)
        avg_acc = 100. * correct_tensor.item() / total_tensor.item()
        avg_agreement = 100. * agreement_tensor.item() / total_tensor.item() if not in_warmup else 0.0

        metrics = {
            'train_loss': avg_loss,
            'train_cls_loss': avg_cls_loss,
            'train_dist_loss': avg_dist_loss,
            'train_acc': avg_acc,
            'train_agreement': avg_agreement
        }

        return metrics

    @torch.no_grad()
    def validate_ddp(self, val_loader):
        """
        Validate with detailed metrics for distillation.

        Reports:
        - Overall accuracy (averaged cls + dist outputs)
        - cls_only_acc: accuracy using only cls head
        - dist_only_acc: accuracy using only dist head
        """
        self.ddp_model.eval()
        total_loss = 0
        correct_avg = 0
        correct_cls = 0
        correct_dist = 0
        total = 0

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Get separate outputs using the model's method
            model = self.ddp_model.module if hasattr(self.ddp_model, 'module') else self.ddp_model

            if hasattr(model, 'get_classifier_outputs'):
                cls_out, dist_out, avg_out = model.get_classifier_outputs(inputs)
            else:
                # Fallback
                outputs = self.ddp_model(inputs)
                if isinstance(outputs, tuple):
                    cls_out, dist_out = outputs
                    avg_out = (cls_out + dist_out) / 2
                else:
                    cls_out = outputs
                    dist_out = outputs
                    avg_out = outputs

            loss = self.criterion(avg_out, targets)
            total_loss += loss.item() * inputs.size(0)

            # Handle one-hot encoded targets
            if len(targets.shape) > 1:
                targets = targets.argmax(1)

            # Track accuracy for each output
            correct_avg += avg_out.argmax(1).eq(targets).sum().item()
            correct_cls += cls_out.argmax(1).eq(targets).sum().item()
            if dist_out is not None:
                correct_dist += dist_out.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        # Aggregate metrics across all GPUs
        loss_tensor = torch.tensor([total_loss], device=self.device)
        correct_avg_tensor = torch.tensor([correct_avg], device=self.device)
        correct_cls_tensor = torch.tensor([correct_cls], device=self.device)
        correct_dist_tensor = torch.tensor([correct_dist], device=self.device)
        total_tensor = torch.tensor([total], device=self.device)

        self.dist.all_reduce(loss_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(correct_avg_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(correct_cls_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(correct_dist_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(total_tensor, op=self.dist.ReduceOp.SUM)

        avg_loss = loss_tensor.item() / total_tensor.item()
        avg_acc = 100. * correct_avg_tensor.item() / total_tensor.item()
        cls_acc = 100. * correct_cls_tensor.item() / total_tensor.item()
        dist_acc = 100. * correct_dist_tensor.item() / total_tensor.item()

        metrics = {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'val_cls_acc': cls_acc,
            'val_dist_acc': dist_acc
        }

        return metrics

    def train_ddp(self, train_loader, train_sampler, val_loader, num_epochs=None):
        """
        Full distillation training loop.
        """
        num_epochs = num_epochs or self.config.training.num_epochs

        if self.is_main_process:
            logger.info(f"Starting DeiT distillation training for {num_epochs} epochs")
            logger.info(f"World Size: {self.world_size}")
            logger.info(f"Effective Batch Size: {self.config.data.batch_size * self.world_size}")
            logger.info(f"Distillation warmup: {self.distillation_warmup_epochs} epochs")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # LR Warmup phase (separate from distillation warmup)
            if epoch < self.config.training.warmup_epochs:
                warmup_lr = self.config.training.learning_rate * (epoch + 1) / self.config.training.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Training
            train_metrics = self.train_epoch_ddp(train_loader, train_sampler)

            # Validation
            val_metrics = self.validate_ddp(val_loader)

            # Update scheduler
            if self.use_swa and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.ddp_model.module)
                self.swa_scheduler.step()
            elif self.scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            # Only rank 0 logs and saves
            if self.is_main_process:
                in_warmup = epoch < self.distillation_warmup_epochs
                warmup_indicator = " [Warmup]" if in_warmup else ""

                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}]{warmup_indicator} "
                    f"Loss: {train_metrics['train_loss']:.4f}, "
                    f"Train Acc: {train_metrics['train_acc']:.2f}%, "
                    f"Val Acc: {val_metrics['val_acc']:.2f}% "
                    f"(cls: {val_metrics['val_cls_acc']:.2f}%, dist: {val_metrics['val_dist_acc']:.2f}%), "
                    f"Agreement: {train_metrics['train_agreement']:.1f}%, "
                    f"LR: {current_lr:.6f}, "
                    f"Time: {epoch_time:.2f}s"
                )

                # Save metrics history
                for key, value in {**train_metrics, **val_metrics}.items():
                    self.metrics_history[key].append(value)

                # Save checkpoint if best (based on averaged accuracy)
                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)

                # Periodic checkpoint
                if (epoch + 1) % self.config.logging.save_frequency == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_metrics)

                # Early stopping
                if self.early_stopping:
                    if self.early_stopping(val_metrics['val_loss']):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            # Synchronize all processes after each epoch
            self.dist.barrier()

        # SWA finalization
        if self.use_swa and self.is_main_process:
            logger.info("Updating batch normalization statistics for SWA model")
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)

        if self.is_main_process:
            logger.info(f"Distillation training completed. Best Val Acc: {self.best_val_acc:.2f}%")

        return dict(self.metrics_history)

    def save_checkpoint(self, filename, epoch, metrics):
        """Save checkpoint with distillation-specific information using shared utility."""
        checkpoint_dir = Path(self.config.output_dir) / self.config.experiment_name / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get model (unwrap DDP)
        model = self.ddp_model.module if hasattr(self.ddp_model, 'module') else self.ddp_model
        swa_model = self.swa_model if self.use_swa else None
        scaler = self.scaler if self.use_amp else None

        # Distillation-specific metadata
        extra_metadata = {
            'distillation_config': {
                'type': self.distillation_type,
                'alpha': self.distillation_alpha,
                'tau': self.distillation_tau,
                'warmup_epochs': self.distillation_warmup_epochs,
                'teacher_checkpoint': self.config.distillation.teacher_checkpoint
            }
        }

        checkpoint = build_checkpoint_dict(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=scaler,
            swa_model=swa_model,
            epoch=epoch,
            metrics=metrics,
            config=self.config,
            best_val_acc=self.best_val_acc,
            metrics_history=self.metrics_history,
            extra_metadata=extra_metadata
        )

        save_path = checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")


# =============================================================================
# Self-Supervised Token Correlation Distillation (CST-style)
# =============================================================================

def load_dino_teacher(teacher_type, model_name, device):
    """
    Load a pretrained DINO/DINOv2 model as teacher.

    Args:
        teacher_type: 'dino' or 'dinov2'
        model_name: Model identifier (e.g., 'dinov2_vits14', 'dino_vits16')
        device: Target device

    Returns:
        teacher_model: Frozen pretrained ViT teacher
        embed_dim: Teacher embedding dimension
    """
    # Embedding dimension lookup
    embed_dim_map = {
        # DINOv2 models
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536,
        # DINO models
        'dino_vits16': 384,
        'dino_vits8': 384,
        'dino_vitb16': 768,
        'dino_vitb8': 768,
    }

    if teacher_type == 'dinov2':
        logger.info(f"Loading DINOv2 teacher: {model_name}")
        teacher_model = torch.hub.load('facebookresearch/dinov2', model_name)
    elif teacher_type == 'dino':
        logger.info(f"Loading DINO teacher: {model_name}")
        teacher_model = torch.hub.load('facebookresearch/dino:main', model_name)
    else:
        raise ValueError(f"Unknown teacher_type: {teacher_type}. Use 'dino' or 'dinov2'.")

    embed_dim = embed_dim_map.get(model_name, 384)

    # Freeze teacher
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    logger.info(f"Loaded {teacher_type} teacher: {model_name} (embed_dim={embed_dim}, frozen)")
    return teacher_model, embed_dim


class ProjectionHead(nn.Module):
    """
    Learnable projection head to align student/teacher embedding dimensions.

    Architecture: Linear -> LayerNorm -> GELU -> Linear -> LayerNorm
    This stabilizes cosine similarity and allows dimension mismatch handling.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim * 2

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class TokenRepresentationLoss(nn.Module):
    """
    Token representation distillation loss (L_tok) - PRIMARY SIGNAL.

    Matches intermediate layer embeddings between teacher and student
    using learnable projection heads for dimension alignment.
    """

    def __init__(self, student_dim, teacher_dim, projection_dim, num_layers, loss_type='cosine'):
        super().__init__()
        self.loss_type = loss_type
        self.num_layers = num_layers

        # Separate projectors per layer - allows layer-specific alignment
        self.student_projectors = nn.ModuleList([
            ProjectionHead(student_dim, projection_dim)
            for _ in range(num_layers)
        ])
        self.teacher_projectors = nn.ModuleList([
            ProjectionHead(teacher_dim, projection_dim)
            for _ in range(num_layers)
        ])

    def forward(self, student_intermediates, teacher_intermediates, layer_indices):
        """
        Compute token representation loss.

        Args:
            student_intermediates: Dict[layer_idx] -> (B, N_s, d_s)
            teacher_intermediates: Dict[layer_idx] -> (B, N_s, d_t) - already interpolated
            layer_indices: List of layer indices

        Returns:
            loss: Scalar loss value
            loss_dict: Per-layer losses
        """
        total_loss = 0.0
        loss_dict = {}

        for i, layer_idx in enumerate(layer_indices):
            student_tokens = student_intermediates[layer_idx]  # (B, N, d_s)
            teacher_tokens = teacher_intermediates[layer_idx]  # (B, N, d_t)

            # Project to common space
            proj_student = self.student_projectors[i](student_tokens)  # (B, N, proj_dim)
            proj_teacher = self.teacher_projectors[i](teacher_tokens)  # (B, N, proj_dim)

            # Compute loss
            if self.loss_type == 'cosine':
                # Negative cosine similarity (minimize = maximize similarity)
                proj_student_norm = F.normalize(proj_student, dim=-1)
                proj_teacher_norm = F.normalize(proj_teacher, dim=-1)
                layer_loss = 1 - (proj_student_norm * proj_teacher_norm).sum(dim=-1).mean()
            else:  # mse
                layer_loss = F.mse_loss(proj_student, proj_teacher)

            total_loss += layer_loss
            loss_dict[f'tok_loss_layer_{layer_idx}'] = layer_loss.item()

        # Average over layers
        total_loss = total_loss / len(layer_indices)
        loss_dict['tok_loss_total'] = total_loss.item()

        return total_loss, loss_dict


class TokenCorrelationLoss(nn.Module):
    """
    Token correlation distillation loss (L_rel) - LIGHTWEIGHT REGULARIZER.

    Matches token-token correlation matrices between teacher and student
    for structural consistency. Uses patch-mean pooling to avoid O(N²) cost.
    """

    def __init__(self, temperature=0.1, loss_type='kl', use_pooled=True):
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.use_pooled = use_pooled

    def forward(self, student_tokens, teacher_tokens):
        """
        Compute token correlation loss.

        Args:
            student_tokens: (B, N_s, d_s) patch tokens from student
            teacher_tokens: (B, N_t, d_t) patch tokens from teacher

        Returns:
            loss: Scalar loss value
        """
        if self.use_pooled:
            # Patch-mean pooling: (B, N, D) → (B, D) - avoids N² correlation matrix
            student_pooled = student_tokens.mean(dim=1)  # (B, D)
            teacher_pooled = teacher_tokens.mean(dim=1)  # (B, D)

            # Compute batch correlation: (B, B) matrix
            student_norm = F.normalize(student_pooled, dim=-1)
            teacher_norm = F.normalize(teacher_pooled, dim=-1)

            student_corr = student_norm @ student_norm.T  # (B, B)
            teacher_corr = teacher_norm @ teacher_norm.T  # (B, B)
        else:
            # Full correlation (expensive for large N)
            student_norm = F.normalize(student_tokens, dim=-1)
            teacher_norm = F.normalize(teacher_tokens, dim=-1)

            # (B, N, N) correlation matrices
            student_corr = torch.bmm(student_norm, student_norm.transpose(1, 2))
            teacher_corr = torch.bmm(teacher_norm, teacher_norm.transpose(1, 2))

        # Apply temperature and normalize
        student_prob = F.softmax(student_corr / self.temperature, dim=-1)
        teacher_prob = F.softmax(teacher_corr / self.temperature, dim=-1)

        if self.loss_type == 'kl':
            # KL divergence (more stable for probability matrices)
            loss = F.kl_div(
                student_prob.log(),
                teacher_prob,
                reduction='batchmean'
            )
        else:  # frobenius
            loss = torch.norm(student_prob - teacher_prob, p='fro') / student_prob.numel()

        return loss


class CKALoss(nn.Module):
    """
    Centered Kernel Alignment (CKA) Loss for structural distillation.

    CKA measures representational similarity by comparing gram matrices of
    activations. This captures structural (relational) knowledge rather than
    just token-wise matching.

    CKA = HSIC(K_S, K_T) / sqrt(HSIC(K_S, K_S) * HSIC(K_T, K_T))
    Loss = 1 - CKA

    Reference: Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019.
    """

    def __init__(self, kernel_type='linear', eps=1e-8, normalize_features=True):
        """
        Args:
            kernel_type: 'linear' or 'rbf' kernel for gram matrix computation
            eps: Small constant for numerical stability
            normalize_features: If True, L2 normalize features before gram computation
                              (recommended for numerical stability across batch sizes)
        """
        super().__init__()
        self.kernel_type = kernel_type
        self.eps = eps
        self.normalize_features = normalize_features

    def _compute_gram(self, x):
        """
        Compute gram matrix (kernel matrix) from feature representations.

        Args:
            x: Feature tensor (B, N, D) - batch of token sequences

        Returns:
            K: Gram matrix (B, N, N)
        """
        # Optional L2 normalization for numerical stability
        if self.normalize_features:
            x = F.normalize(x, p=2, dim=-1)

        if self.kernel_type == 'linear':
            # Linear kernel: K = X @ X^T
            K = torch.bmm(x, x.transpose(1, 2))  # (B, N, N)
        else:  # rbf
            # RBF kernel: K_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
            # Use median heuristic for sigma
            x_norm = (x ** 2).sum(-1, keepdim=True)  # (B, N, 1)
            dist_sq = x_norm + x_norm.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
            sigma = torch.median(dist_sq.view(-1)).item() + self.eps
            K = torch.exp(-dist_sq / (2 * sigma))
        return K

    def _center_gram(self, K):
        """
        Center the gram matrix (subtract row/column means).

        Args:
            K: Gram matrix (B, N, N)

        Returns:
            K_centered: Centered gram matrix (B, N, N)
        """
        B, N, _ = K.shape
        # Centering matrix: H = I - (1/N) * 1 * 1^T
        # K_centered = H @ K @ H
        row_mean = K.mean(dim=2, keepdim=True)  # (B, N, 1)
        col_mean = K.mean(dim=1, keepdim=True)  # (B, 1, N)
        total_mean = K.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)

        K_centered = K - row_mean - col_mean + total_mean
        return K_centered

    def _hsic(self, K1, K2):
        """
        Compute Hilbert-Schmidt Independence Criterion (HSIC).

        HSIC(K1, K2) = (1/(N-1)^2) * trace(K1_centered @ K2_centered)

        Args:
            K1, K2: Centered gram matrices (B, N, N)

        Returns:
            hsic: HSIC value (B,)
        """
        B, N, _ = K1.shape
        # trace(K1 @ K2) = sum(K1 * K2^T)
        hsic = (K1 * K2).sum(dim=(1, 2)) / ((N - 1) ** 2 + self.eps)
        return hsic

    def _compute_batch_cka(self, student_tokens, teacher_tokens):
        """
        Compute batch-wise CKA for CLS-only mode (single token per sample).

        Instead of computing token-token correlations within each sample,
        compute sample-sample correlations across the batch.

        Args:
            student_tokens: (B, 1, D_s) - CLS tokens
            teacher_tokens: (B, 1, D_t) - CLS tokens

        Returns:
            cka: Scalar CKA similarity
        """
        # Squeeze to (B, D)
        s = student_tokens.squeeze(1)  # (B, D_s)
        t = teacher_tokens.squeeze(1)  # (B, D_t)

        # L2 normalize for numerical stability
        if self.normalize_features:
            s = F.normalize(s, p=2, dim=-1)
            t = F.normalize(t, p=2, dim=-1)

        # Compute batch gram matrices (B, B)
        K_s = s @ s.T  # (B, B)
        K_t = t @ t.T  # (B, B)

        # Center gram matrices
        B = K_s.shape[0]
        row_mean_s = K_s.mean(dim=1, keepdim=True)
        col_mean_s = K_s.mean(dim=0, keepdim=True)
        total_mean_s = K_s.mean()
        K_s_centered = K_s - row_mean_s - col_mean_s + total_mean_s

        row_mean_t = K_t.mean(dim=1, keepdim=True)
        col_mean_t = K_t.mean(dim=0, keepdim=True)
        total_mean_t = K_t.mean()
        K_t_centered = K_t - row_mean_t - col_mean_t + total_mean_t

        # Compute HSIC values
        hsic_st = (K_s_centered * K_t_centered).sum() / ((B - 1) ** 2 + self.eps)
        hsic_ss = (K_s_centered * K_s_centered).sum() / ((B - 1) ** 2 + self.eps)
        hsic_tt = (K_t_centered * K_t_centered).sum() / ((B - 1) ** 2 + self.eps)

        # Compute CKA
        denominator = torch.sqrt(hsic_ss * hsic_tt + self.eps)
        cka = hsic_st / denominator

        return cka

    def forward(self, student_tokens, teacher_tokens):
        """
        Compute CKA loss between student and teacher representations.

        Args:
            student_tokens: (B, N_s, D_s) student token representations
            teacher_tokens: (B, N_t, D_t) teacher token representations
                           (should be interpolated to match N_s)

        Returns:
            loss: 1 - CKA (minimize to maximize CKA similarity)
        """
        B, N, _ = student_tokens.shape

        # CLS-only mode: use batch-wise CKA instead of token-wise
        if N == 1:
            cka = self._compute_batch_cka(student_tokens, teacher_tokens)
            loss = 1.0 - cka
            return loss

        # Standard token-wise CKA for multiple tokens
        # Compute gram matrices
        K_s = self._compute_gram(student_tokens)  # (B, N, N)
        K_t = self._compute_gram(teacher_tokens)  # (B, N, N)

        # Center gram matrices
        K_s = self._center_gram(K_s)
        K_t = self._center_gram(K_t)

        # Compute HSIC values
        hsic_st = self._hsic(K_s, K_t)  # Cross-similarity
        hsic_ss = self._hsic(K_s, K_s)  # Student self-similarity
        hsic_tt = self._hsic(K_t, K_t)  # Teacher self-similarity

        # Compute CKA
        denominator = torch.sqrt(hsic_ss * hsic_tt + self.eps)
        cka = hsic_st / denominator  # (B,)

        # Loss = 1 - CKA (minimize to maximize alignment)
        loss = 1.0 - cka.mean()

        return loss


class GramMatrixLoss(nn.Module):
    """
    Gram Matrix Loss for structural distillation (ablation baseline).

    Directly compares gram matrices using Frobenius norm.
    Simpler than CKA but less invariant to affine transformations.

    Loss = ||G_S - G_T||_F / num_elements
    """

    def __init__(self, normalize=True, eps=1e-8):
        """
        Args:
            normalize: Whether to L2-normalize features before gram computation
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def _compute_gram(self, x):
        """
        Compute normalized gram matrix.

        Args:
            x: Feature tensor (B, N, D)

        Returns:
            G: Gram matrix (B, N, N)
        """
        if self.normalize:
            x = F.normalize(x, dim=-1)
        G = torch.bmm(x, x.transpose(1, 2))  # (B, N, N)
        return G

    def forward(self, student_tokens, teacher_tokens):
        """
        Compute gram matrix loss.

        Args:
            student_tokens: (B, N_s, D_s) student representations
            teacher_tokens: (B, N_t, D_t) teacher representations

        Returns:
            loss: Frobenius norm difference between gram matrices
        """
        G_s = self._compute_gram(student_tokens)  # (B, N, N)
        G_t = self._compute_gram(teacher_tokens)  # (B, N, N)

        # Frobenius norm of difference, normalized by number of elements
        diff = G_s - G_t
        loss = torch.norm(diff, p='fro', dim=(1, 2))  # (B,)
        loss = loss / (G_s.shape[1] ** 2)  # Normalize by N^2
        loss = loss.mean()

        return loss


class LayerWiseStructuralLoss(nn.Module):
    """
    Layer-wise structural distillation with CKA and optional Gram matrix losses.

    Applies structural loss at multiple intermediate layers with learnable
    projection heads for dimension alignment.
    """

    def __init__(self, student_dim, teacher_dim, projection_dim, num_layers,
                 use_cka=True, use_gram=False, cka_kernel='linear'):
        """
        Args:
            student_dim: Student embedding dimension
            teacher_dim: Teacher embedding dimension
            projection_dim: Projection head output dimension
            num_layers: Number of layers to match
            use_cka: Whether to use CKA loss
            use_gram: Whether to use Gram matrix loss (ablation)
            cka_kernel: 'linear' or 'rbf' kernel for CKA
        """
        super().__init__()
        self.use_cka = use_cka
        self.use_gram = use_gram
        self.num_layers = num_layers

        # Projectors for dimension alignment
        self.student_projectors = nn.ModuleList([
            ProjectionHead(student_dim, projection_dim)
            for _ in range(num_layers)
        ])
        self.teacher_projectors = nn.ModuleList([
            ProjectionHead(teacher_dim, projection_dim)
            for _ in range(num_layers)
        ])

        # Loss functions
        if use_cka:
            self.cka_loss = CKALoss(kernel_type=cka_kernel)
        if use_gram:
            self.gram_loss = GramMatrixLoss(normalize=True)

    def forward(self, student_intermediates, teacher_intermediates, layer_indices):
        """
        Compute layer-wise structural loss.

        Args:
            student_intermediates: Dict[layer_idx] -> (B, N_s, D_s)
            teacher_intermediates: Dict[layer_idx] -> (B, N_s, D_t)
            layer_indices: List of layer indices

        Returns:
            loss: Combined structural loss
            loss_dict: Per-layer and per-loss-type breakdowns
        """
        total_cka_loss = 0.0
        total_gram_loss = 0.0
        loss_dict = {}

        for i, layer_idx in enumerate(layer_indices):
            student_tokens = student_intermediates[layer_idx]
            teacher_tokens = teacher_intermediates[layer_idx]

            # Project to common space
            proj_student = self.student_projectors[i](student_tokens)
            proj_teacher = self.teacher_projectors[i](teacher_tokens)

            if self.use_cka:
                layer_cka = self.cka_loss(proj_student, proj_teacher)
                total_cka_loss += layer_cka
                loss_dict[f'cka_loss_layer_{layer_idx}'] = layer_cka.item()

            if self.use_gram:
                layer_gram = self.gram_loss(proj_student, proj_teacher)
                total_gram_loss += layer_gram
                loss_dict[f'gram_loss_layer_{layer_idx}'] = layer_gram.item()

        # Average over layers
        num_layers = len(layer_indices)
        if self.use_cka:
            total_cka_loss = total_cka_loss / num_layers
            loss_dict['cka_loss_total'] = total_cka_loss.item()
        if self.use_gram:
            total_gram_loss = total_gram_loss / num_layers
            loss_dict['gram_loss_total'] = total_gram_loss.item()

        # Combined loss
        total_loss = total_cka_loss if self.use_cka else 0.0
        if self.use_gram:
            total_loss = total_loss + total_gram_loss

        return total_loss, loss_dict


class SelfSupervisedDistillationLoss(nn.Module):
    """
    Combined loss for CST-style self-supervised distillation.

    L = L_ce + lambda_tok * L_tok + lambda_rel * L_rel + lambda_cka * L_cka + lambda_gram * L_gram

    Supports staged training:
    - Stage A (first rel_warmup_epochs): L = L_ce + L_tok
    - Stage B (remaining epochs): Full loss with L_rel
    - CKA/Gram losses enabled after cka_warmup_epochs

    Structural losses (CKA, Gram) are the PRIMARY signal for structural distillation
    experiments. Token losses (L_tok, L_rel) can be disabled when using structural losses.
    """

    def __init__(self, base_criterion, student_dim, teacher_dim, config):
        """
        Args:
            base_criterion: Base classification loss (e.g., LabelSmoothingCE)
            student_dim: Student embedding dimension
            teacher_dim: Teacher embedding dimension
            config: SelfSupervisedDistillationConfig
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.config = config

        # Token representation loss - PRIMARY
        self.token_rep_loss = TokenRepresentationLoss(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            projection_dim=config.projection_dim,
            num_layers=len(config.token_layers),
            loss_type=config.token_loss_type
        )

        # Token correlation loss - REGULARIZER
        self.token_corr_loss = TokenCorrelationLoss(
            temperature=config.correlation_temperature,
            loss_type=config.correlation_loss_type,
            use_pooled=config.use_pooled_correlation
        )

        # CKA structural loss (optional)
        self.use_cka_loss = getattr(config, 'use_cka_loss', False)
        if self.use_cka_loss:
            self.structural_loss = LayerWiseStructuralLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                projection_dim=config.projection_dim,
                num_layers=len(config.token_layers),
                use_cka=True,
                use_gram=getattr(config, 'use_gram_loss', False),
                cka_kernel=getattr(config, 'cka_kernel_type', 'linear')
            )
            self.lambda_cka = getattr(config, 'lambda_cka', 0.5)
            self.cka_warmup_epochs = getattr(config, 'cka_warmup_epochs', 5)
        else:
            self.structural_loss = None
            self.lambda_cka = 0.0
            self.cka_warmup_epochs = 0

        # Gram matrix loss (ablation - standalone without CKA)
        self.use_gram_loss = getattr(config, 'use_gram_loss', False)
        if self.use_gram_loss and not self.use_cka_loss:
            # Gram loss without CKA - create separate loss module
            self.gram_only_loss = LayerWiseStructuralLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                projection_dim=config.projection_dim,
                num_layers=len(config.token_layers),
                use_cka=False,
                use_gram=True
            )
            self.lambda_gram = getattr(config, 'lambda_gram', 0.5)
        else:
            self.gram_only_loss = None
            self.lambda_gram = getattr(config, 'lambda_gram', 0.5) if self.use_cka_loss else 0.0

        self.lambda_tok = config.lambda_tok
        self.lambda_rel = config.lambda_rel
        self.token_layers = config.token_layers

    def get_effective_lambda_rel(self, epoch):
        """Get effective lambda_rel considering warmup."""
        if epoch < self.config.rel_warmup_epochs:
            return 0.0  # Stage A: no correlation loss
        return self.lambda_rel  # Stage B: add L_rel

    def get_effective_lambda_cka(self, epoch):
        """Get effective lambda_cka considering warmup."""
        if not self.use_cka_loss:
            return 0.0
        if epoch < self.cka_warmup_epochs:
            return 0.0
        return self.lambda_cka

    def forward(self, student_output, targets,
                student_intermediates, teacher_intermediates,
                student_patch_tokens, teacher_patch_tokens,
                epoch):
        """
        Compute combined distillation loss.

        Args:
            student_output: (cls_logits, dist_logits) or cls_logits
            targets: Ground truth labels
            student_intermediates: Dict of intermediate student tokens
            teacher_intermediates: Dict of intermediate teacher tokens
            student_patch_tokens: Final student patch tokens (B, N, d_s)
            teacher_patch_tokens: Final teacher patch tokens (B, N, d_t)
            epoch: Current epoch for staged training

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Classification loss (on student CLS head ONLY - critical!)
        if isinstance(student_output, tuple):
            cls_output, dist_output = student_output
        else:
            cls_output = student_output

        ce_loss = self.base_criterion(cls_output, targets)

        # Token representation loss (PRIMARY - always on)
        tok_loss, tok_loss_dict = self.token_rep_loss(
            student_intermediates, teacher_intermediates, self.token_layers
        )

        # Token correlation loss (REGULARIZER - staged)
        effective_lambda_rel = self.get_effective_lambda_rel(epoch)
        if effective_lambda_rel > 0:
            rel_loss = self.token_corr_loss(student_patch_tokens, teacher_patch_tokens)
        else:
            rel_loss = torch.tensor(0.0, device=ce_loss.device)

        # CKA/Gram structural loss (optional - staged)
        effective_lambda_cka = self.get_effective_lambda_cka(epoch)
        cka_loss = torch.tensor(0.0, device=ce_loss.device)
        gram_loss = torch.tensor(0.0, device=ce_loss.device)
        structural_loss_dict = {}

        if self.structural_loss is not None and effective_lambda_cka > 0:
            struct_loss, structural_loss_dict = self.structural_loss(
                student_intermediates, teacher_intermediates, self.token_layers
            )
            if self.use_cka_loss:
                cka_loss = struct_loss
            if self.use_gram_loss and 'gram_loss_total' in structural_loss_dict:
                gram_loss = torch.tensor(structural_loss_dict['gram_loss_total'], device=ce_loss.device)

        # Gram-only loss (ablation without CKA)
        if self.gram_only_loss is not None and epoch >= self.cka_warmup_epochs:
            gram_struct_loss, gram_loss_dict = self.gram_only_loss(
                student_intermediates, teacher_intermediates, self.token_layers
            )
            gram_loss = gram_struct_loss
            structural_loss_dict.update(gram_loss_dict)

        # Combined loss
        total_loss = (
            ce_loss
            + self.lambda_tok * tok_loss
            + effective_lambda_rel * rel_loss
            + effective_lambda_cka * cka_loss
        )
        if self.gram_only_loss is not None:
            total_loss = total_loss + self.lambda_gram * gram_loss

        loss_dict = {
            'ce_loss': ce_loss.item(),
            'tok_loss': tok_loss.item(),
            'rel_loss': rel_loss.item() if isinstance(rel_loss, torch.Tensor) else 0.0,
            'cka_loss': cka_loss.item() if isinstance(cka_loss, torch.Tensor) else 0.0,
            'gram_loss': gram_loss.item() if isinstance(gram_loss, torch.Tensor) else 0.0,
            'effective_lambda_rel': effective_lambda_rel,
            'effective_lambda_cka': effective_lambda_cka,
            'total_loss': total_loss.item(),
            **tok_loss_dict,
            **structural_loss_dict
        }

        return total_loss, loss_dict


class SelfSupervisedDistillationTrainer(DDPTrainer):
    """
    Trainer for CST-style self-supervised token correlation distillation.

    Uses a pretrained DINO/DINOv2 teacher instead of a weaker CNN.
    Implements two-stage training:
    - Stage A: L_ce + L_tok (representation matching)
    - Stage B: L_ce + L_tok + L_rel (add correlation matching)
    """

    def __init__(self, student_model, teacher_model, teacher_embed_dim,
                 config, device, rank, world_size):
        # Initialize base trainer
        super().__init__(student_model, config, device, rank, world_size)

        # Store teacher
        self.teacher_model = teacher_model
        self.teacher_embed_dim = teacher_embed_dim

        # Get configs
        self.ss_config = config.ss_distillation
        self.token_layers = self.ss_config.token_layers

        # Get student embedding dim
        student_module = student_model.module if hasattr(student_model, 'module') else student_model
        self.student_embed_dim = student_module.embed_dim

        # Compute student token count for interpolation
        # For CIFAR-10 with patch_size=4: 32/4 = 8, so 8x8 = 64 tokens
        self.student_num_tokens = student_module.patch_embed.num_patches

        # Create distillation loss
        self.distillation_criterion = SelfSupervisedDistillationLoss(
            base_criterion=self.criterion,
            student_dim=self.student_embed_dim,
            teacher_dim=self.teacher_embed_dim,
            config=self.ss_config
        ).to(device)

        # CRITICAL: Add projection head params to optimizer
        self.optimizer.add_param_group({
            'params': self.distillation_criterion.token_rep_loss.parameters(),
            'lr': config.training.learning_rate
        })

        # Add structural loss parameters if CKA/Gram enabled
        if self.distillation_criterion.structural_loss is not None:
            self.optimizer.add_param_group({
                'params': self.distillation_criterion.structural_loss.parameters(),
                'lr': config.training.learning_rate
            })
        if self.distillation_criterion.gram_only_loss is not None:
            self.optimizer.add_param_group({
                'params': self.distillation_criterion.gram_only_loss.parameters(),
                'lr': config.training.learning_rate
            })

        # Track structural loss usage
        self.use_cka = getattr(self.ss_config, 'use_cka_loss', False)
        self.use_gram = getattr(self.ss_config, 'use_gram_loss', False)

        # Track dual-augment mode
        self.use_dual_augment = getattr(self.ss_config, 'use_dual_augment', False)

        # CLS-only mode for CKA (cleaner global semantic alignment)
        self.use_cls_only = getattr(self.ss_config, 'use_cls_only', False)

        # Create MixUp function for training loop (applied only to student images)
        self.mixup_fn = None
        self.num_classes = config.model.num_classes
        aug_config = config.data.augmentation
        if aug_config.get('mixup') or aug_config.get('cutmix'):
            self.mixup_alpha = aug_config.get('mixup_alpha', 1.0) if aug_config.get('mixup') else 0.0
            self.cutmix_alpha = aug_config.get('cutmix_alpha', 1.0) if aug_config.get('cutmix') else 0.0
            if self.is_main_process:
                logger.info(f"  - MixUp/CutMix enabled (mixup_alpha={self.mixup_alpha}, cutmix_alpha={self.cutmix_alpha})")

        if self.is_main_process:
            logger.info("=" * 60)
            logger.info("Self-Supervised Distillation Trainer initialized:")
            logger.info(f"  - Teacher: {self.ss_config.teacher_model_name} (dim={self.teacher_embed_dim})")
            logger.info(f"  - Student: DeiT (dim={self.student_embed_dim})")
            logger.info(f"  - Token layers: {self.token_layers}")
            logger.info(f"  - Lambda_tok: {self.ss_config.lambda_tok}")
            logger.info(f"  - Lambda_rel: {self.ss_config.lambda_rel}")
            logger.info(f"  - Rel warmup epochs: {self.ss_config.rel_warmup_epochs}")
            logger.info(f"  - Dual-augment: {self.use_dual_augment}")
            if self.use_cka:
                logger.info(f"  - CKA Loss: ENABLED (lambda={getattr(self.ss_config, 'lambda_cka', 0.5)}, "
                           f"kernel={getattr(self.ss_config, 'cka_kernel_type', 'linear')})")
            if self.use_gram:
                logger.info(f"  - Gram Loss: ENABLED (lambda={getattr(self.ss_config, 'lambda_gram', 0.5)})")
            logger.info(f"  - Student tokens: {self.student_num_tokens}")
            if self.use_cls_only:
                logger.info(f"  - CLS-only mode: ENABLED (global semantic alignment)")
            logger.info("=" * 60)

    def apply_mixup(self, images, targets):
        """
        Apply MixUp or CutMix augmentation to images and targets.

        Args:
            images: Batch of images (B, C, H, W)
            targets: Integer class labels (B,)

        Returns:
            mixed_images: Augmented images
            mixed_targets: Soft labels (B, num_classes)
        """
        B = images.shape[0]
        device = images.device

        # Randomly choose between MixUp and CutMix
        use_cutmix = self.cutmix_alpha > 0 and (self.mixup_alpha == 0 or torch.rand(1).item() > 0.5)

        if use_cutmix:
            alpha = self.cutmix_alpha
        else:
            alpha = self.mixup_alpha

        if alpha <= 0:
            # No mixing, just convert to one-hot
            targets_onehot = torch.zeros(B, self.num_classes, device=device)
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
            return images, targets_onehot

        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)

        # Get random permutation for mixing
        rand_idx = torch.randperm(B, device=device)

        if use_cutmix:
            # CutMix: cut and paste patches
            _, _, H, W = images.shape
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            mixed_images = images.clone()
            mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_idx, :, bby1:bby2, bbx1:bbx2]

            # Adjust lambda based on actual cut size
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        else:
            # MixUp: linear interpolation
            mixed_images = lam * images + (1 - lam) * images[rand_idx]

        # Create soft labels
        targets_onehot = torch.zeros(B, self.num_classes, device=device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

        targets_shuffled = torch.zeros(B, self.num_classes, device=device)
        targets_shuffled.scatter_(1, targets[rand_idx].unsqueeze(1), 1.0)

        mixed_targets = lam * targets_onehot + (1 - lam) * targets_shuffled

        return mixed_images, mixed_targets

    def get_teacher_intermediates(self, x, layer_indices):
        """
        Extract intermediate tokens from teacher (DINO/DINOv2).

        Handles resolution mismatch:
        1. Upsample input to 224x224
        2. Get teacher tokens (196 for 14x14 patches) OR CLS token only
        3. Interpolate teacher tokens to match student count (if using patches)

        Args:
            x: Input tensor (B, C, H, W) - original resolution
            layer_indices: List of layer indices to extract

        Returns:
            intermediates: Dict[layer_idx] -> (B, N_student, teacher_dim) or (B, 1, teacher_dim) for CLS
            patch_tokens: Final teacher patch/CLS tokens
        """
        with torch.no_grad():
            B = x.shape[0]

            # Upsample input to 224x224 for teacher (bicubic for sharper edges)
            if x.shape[-1] < 224:
                teacher_input = F.interpolate(x, size=224, mode='bicubic', align_corners=False)
            else:
                teacher_input = x

            intermediates = {}

            # DINOv2 supports get_intermediate_layers
            if hasattr(self.teacher_model, 'get_intermediate_layers'):
                if self.use_cls_only:
                    # CLS-only mode: Extract CLS tokens for global semantic alignment
                    # This avoids noisy spatial interpolation and focuses on global representation
                    outputs = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=layer_indices, return_class_token=True
                    )
                    for i, layer_idx in enumerate(layer_indices):
                        # outputs[i] is tuple (patch_tokens, cls_token) when return_class_token=True
                        if isinstance(outputs[i], tuple):
                            cls_token = outputs[i][1]  # (B, D)
                        else:
                            # Fallback: first token is CLS
                            cls_token = outputs[i][:, 0, :]  # (B, D)
                        # Reshape to (B, 1, D) for consistency with token-based losses
                        intermediates[layer_idx] = cls_token.unsqueeze(1)

                    # Get final CLS token
                    final_output = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=[max(layer_indices)], return_class_token=True
                    )[0]
                    if isinstance(final_output, tuple):
                        patch_tokens = final_output[1].unsqueeze(1)  # CLS token (B, 1, D)
                    else:
                        patch_tokens = final_output[:, 0, :].unsqueeze(1)
                else:
                    # Patch token mode: Extract and interpolate patch tokens
                    outputs = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=layer_indices, return_class_token=False
                    )
                    for i, layer_idx in enumerate(layer_indices):
                        teacher_tokens = outputs[i]  # (B, 196, D) for 224x224 with patch_size=14
                        # Interpolate to student token count
                        teacher_tokens = self._interpolate_tokens(teacher_tokens, self.student_num_tokens)
                        intermediates[layer_idx] = teacher_tokens

                    # Get final patch tokens
                    final_output = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=[11], return_class_token=False
                    )[0]
                    patch_tokens = self._interpolate_tokens(final_output, self.student_num_tokens)
            else:
                # Fallback: manual extraction via forward hooks
                hooks = []
                captured = {}

                def make_hook(idx):
                    def hook(module, input, output):
                        # output is (B, N+1, D) - includes CLS token
                        if isinstance(output, tuple):
                            if self.use_cls_only:
                                captured[idx] = output[0][:, 0:1, :]  # CLS only
                            else:
                                captured[idx] = output[0][:, 1:, :]  # Skip CLS
                        else:
                            if self.use_cls_only:
                                captured[idx] = output[:, 0:1, :]  # CLS only
                            else:
                                captured[idx] = output[:, 1:, :]  # Skip CLS
                    return hook

                # Register hooks
                for idx in layer_indices:
                    if hasattr(self.teacher_model, 'blocks'):
                        hook = self.teacher_model.blocks[idx].register_forward_hook(make_hook(idx))
                        hooks.append(hook)

                # Forward pass
                _ = self.teacher_model(teacher_input)

                # Remove hooks
                for hook in hooks:
                    hook.remove()

                # Process captured tokens
                for idx in layer_indices:
                    if idx in captured:
                        if self.use_cls_only:
                            intermediates[idx] = captured[idx]  # Already (B, 1, D)
                        else:
                            intermediates[idx] = self._interpolate_tokens(captured[idx], self.student_num_tokens)

                # Get final tokens from last layer
                last_idx = max(layer_indices)
                patch_tokens = intermediates[last_idx]

        return intermediates, patch_tokens

    def _interpolate_tokens(self, tokens, target_num_tokens):
        """
        Interpolate teacher tokens to match student token count.

        Teacher tokens: (B, N_teacher, D) where N_teacher = 196 (14x14)
        Target: (B, N_student, D) where N_student = 64 (8x8) for CIFAR

        This preserves student inductive bias.
        """
        B, N, D = tokens.shape
        if N == target_num_tokens:
            return tokens

        # Compute grid sizes
        H_t = int(N ** 0.5)
        H_s = int(target_num_tokens ** 0.5)

        # Reshape to spatial grid: (B, N, D) -> (B, D, H, W)
        tokens = tokens.transpose(1, 2).reshape(B, D, H_t, H_t)

        # Interpolate
        tokens = F.interpolate(tokens, size=(H_s, H_s), mode='bilinear', align_corners=False)

        # Reshape back: (B, D, H, W) -> (B, N, D)
        tokens = tokens.reshape(B, D, -1).transpose(1, 2)

        return tokens

    def train_epoch_ddp(self, train_loader, train_sampler):
        """Train one epoch with self-supervised distillation."""
        train_sampler.set_epoch(self.current_epoch)

        self.ddp_model.train()
        self.teacher_model.eval()

        # Training state
        total_loss = 0
        total_ce_loss = 0
        total_tok_loss = 0
        total_rel_loss = 0
        total_cka_loss = 0
        total_gram_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        # Stage indicator
        in_stage_a = self.current_epoch < self.ss_config.rel_warmup_epochs

        # Optional: freeze projectors for first N epochs
        if self.current_epoch < self.ss_config.projector_warmup_epochs:
            for p in self.distillation_criterion.token_rep_loss.parameters():
                p.requires_grad = False
        else:
            for p in self.distillation_criterion.token_rep_loss.parameters():
                p.requires_grad = True

        if self.is_main_process:
            stage = "A (L_tok only)" if in_stage_a else "B (L_tok + L_rel)"
            pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [{stage}]")
        else:
            pbar = train_loader

        for batch_idx, batch in enumerate(pbar):
            # Handle dual-augment batch format: (clean_imgs, student_imgs, targets)
            if self.use_dual_augment:
                clean_imgs, student_imgs, targets = batch
                clean_imgs = clean_imgs.to(self.device)
                student_imgs = student_imgs.to(self.device)
                targets = targets.to(self.device)

                # Apply MixUp/CutMix ONLY to student images
                if hasattr(self, 'mixup_alpha') and (self.mixup_alpha > 0 or self.cutmix_alpha > 0):
                    student_imgs, targets = self.apply_mixup(student_imgs, targets)

                teacher_input = clean_imgs  # Clean images for teacher
                student_input = student_imgs  # Augmented images for student
            else:
                # Standard batch format: (inputs, targets)
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                teacher_input = inputs
                student_input = inputs

            if self.use_amp:
                with autocast(**self.autocast_kwargs):
                    # Student forward with intermediates (uses student_input)
                    student_module = self.ddp_model.module if hasattr(self.ddp_model, 'module') else self.ddp_model
                    student_results = student_module.forward_with_intermediates(
                        student_input, layer_indices=self.token_layers,
                        return_cls_only=self.use_cls_only
                    )
                    student_output = student_results['output']
                    student_intermediates = student_results['intermediates']
                    student_patch_tokens = student_results['patch_tokens']

                    # Teacher forward with intermediates (uses teacher_input = clean images)
                    teacher_intermediates, teacher_patch_tokens = self.get_teacher_intermediates(
                        teacher_input, self.token_layers
                    )

                    # Compute combined loss
                    loss, loss_dict = self.distillation_criterion(
                        student_output, targets,
                        student_intermediates, teacher_intermediates,
                        student_patch_tokens, teacher_patch_tokens,
                        self.current_epoch
                    )

                    loss = loss / self.grad_accum_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        if self.config.training.gradient_clip_val > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                list(self.ddp_model.parameters()) +
                                list(self.distillation_criterion.token_rep_loss.parameters()),
                                self.config.training.gradient_clip_val
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        if self.config.training.gradient_clip_val > 0:
                            torch.nn.utils.clip_grad_norm_(
                                list(self.ddp_model.parameters()) +
                                list(self.distillation_criterion.token_rep_loss.parameters()),
                                self.config.training.gradient_clip_val
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            else:
                # Non-AMP path
                student_module = self.ddp_model.module if hasattr(self.ddp_model, 'module') else self.ddp_model
                student_results = student_module.forward_with_intermediates(
                    student_input, layer_indices=self.token_layers,
                    return_cls_only=self.use_cls_only
                )

                teacher_intermediates, teacher_patch_tokens = self.get_teacher_intermediates(
                    teacher_input, self.token_layers
                )

                loss, loss_dict = self.distillation_criterion(
                    student_results['output'], targets,
                    student_results['intermediates'], teacher_intermediates,
                    student_results['patch_tokens'], teacher_patch_tokens,
                    self.current_epoch
                )

                loss = loss / self.grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.config.training.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.ddp_model.parameters()) +
                            list(self.distillation_criterion.token_rep_loss.parameters()),
                            self.config.training.gradient_clip_val
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.grad_accum_steps
            total_ce_loss += loss_dict['ce_loss']
            total_tok_loss += loss_dict['tok_loss']
            total_rel_loss += loss_dict['rel_loss']
            total_cka_loss += loss_dict.get('cka_loss', 0.0)
            total_gram_loss += loss_dict.get('gram_loss', 0.0)

            # Accuracy
            if isinstance(student_output, tuple):
                cls_output = student_output[0]
            else:
                cls_output = student_output
            _, predicted = cls_output.max(1)

            if len(targets.shape) > 1:
                targets_idx = targets.argmax(1)
            else:
                targets_idx = targets

            correct += predicted.eq(targets_idx).sum().item()
            total += targets_idx.size(0)
            batch_count += 1

            if self.is_main_process and hasattr(pbar, 'set_postfix'):
                postfix = {
                    'Loss': f'{total_loss/batch_count:.4f}',
                    'CE': f'{total_ce_loss/batch_count:.4f}',
                    'Tok': f'{total_tok_loss/batch_count:.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                }
                if total_rel_loss > 0:
                    postfix['Rel'] = f'{total_rel_loss/batch_count:.4f}'
                if self.use_cka and total_cka_loss > 0:
                    postfix['CKA'] = f'{total_cka_loss/batch_count:.4f}'
                if self.use_gram and total_gram_loss > 0:
                    postfix['Gram'] = f'{total_gram_loss/batch_count:.4f}'
                pbar.set_postfix(postfix)

            self.global_step += 1

        # Aggregate metrics across GPUs
        metrics_tensor = torch.tensor([
            total_loss, total_ce_loss, total_tok_loss, total_rel_loss,
            total_cka_loss, total_gram_loss,
            correct, total, batch_count
        ], device=self.device)

        self.dist.all_reduce(metrics_tensor, op=self.dist.ReduceOp.SUM)

        batch_count_total = int(metrics_tensor[8].item())
        avg_loss = metrics_tensor[0].item() / batch_count_total
        avg_ce_loss = metrics_tensor[1].item() / batch_count_total
        avg_tok_loss = metrics_tensor[2].item() / batch_count_total
        avg_rel_loss = metrics_tensor[3].item() / batch_count_total
        avg_cka_loss = metrics_tensor[4].item() / batch_count_total
        avg_gram_loss = metrics_tensor[5].item() / batch_count_total
        avg_acc = 100. * metrics_tensor[6].item() / metrics_tensor[7].item()

        metrics = {
            'train_loss': avg_loss,
            'train_ce_loss': avg_ce_loss,
            'train_tok_loss': avg_tok_loss,
            'train_rel_loss': avg_rel_loss,
            'train_cka_loss': avg_cka_loss,
            'train_gram_loss': avg_gram_loss,
            'train_acc': avg_acc,
            'effective_lambda_rel': loss_dict['effective_lambda_rel'],
            'effective_lambda_cka': loss_dict.get('effective_lambda_cka', 0.0)
        }

        return metrics

    def train_ddp(self, train_loader, train_sampler, val_loader, num_epochs=None):
        """Full self-supervised distillation training loop."""
        num_epochs = num_epochs or self.config.training.num_epochs

        if self.is_main_process:
            logger.info(f"Starting CST-style self-supervised distillation for {num_epochs} epochs")
            logger.info(f"World Size: {self.world_size}")
            logger.info(f"Effective Batch Size: {self.config.data.batch_size * self.world_size}")
            logger.info(f"Stage A (L_tok only): epochs 0-{self.ss_config.rel_warmup_epochs-1}")
            logger.info(f"Stage B (L_tok + L_rel): epochs {self.ss_config.rel_warmup_epochs}+")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # LR Warmup phase
            if epoch < self.config.training.warmup_epochs:
                warmup_lr = self.config.training.learning_rate * (epoch + 1) / self.config.training.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Training
            train_metrics = self.train_epoch_ddp(train_loader, train_sampler)

            # Validation (use base class method)
            val_metrics = self.validate_ddp(val_loader)

            # Update scheduler
            if self.use_swa and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.ddp_model.module)
                self.swa_scheduler.step()
            elif self.scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            # Only rank 0 logs and saves
            if self.is_main_process:
                stage = "A" if epoch < self.ss_config.rel_warmup_epochs else "B"

                # Build loss breakdown string
                loss_parts = [
                    f"CE: {train_metrics['train_ce_loss']:.4f}",
                    f"Tok: {train_metrics['train_tok_loss']:.4f}",
                ]
                if train_metrics['train_rel_loss'] > 0:
                    loss_parts.append(f"Rel: {train_metrics['train_rel_loss']:.4f}")
                if self.use_cka and train_metrics.get('train_cka_loss', 0) > 0:
                    loss_parts.append(f"CKA: {train_metrics['train_cka_loss']:.4f}")
                if self.use_gram and train_metrics.get('train_gram_loss', 0) > 0:
                    loss_parts.append(f"Gram: {train_metrics['train_gram_loss']:.4f}")
                loss_breakdown = ", ".join(loss_parts)

                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}] [Stage {stage}] "
                    f"Loss: {train_metrics['train_loss']:.4f} "
                    f"({loss_breakdown}), "
                    f"Train Acc: {train_metrics['train_acc']:.2f}%, "
                    f"Val Acc: {val_metrics['val_acc']:.2f}%, "
                    f"LR: {current_lr:.6f}, "
                    f"Time: {epoch_time:.2f}s"
                )

                # Save metrics history
                for key, value in {**train_metrics, **val_metrics}.items():
                    self.metrics_history[key].append(value)

                # Save checkpoint if best
                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)

                # Periodic checkpoint
                if (epoch + 1) % self.config.logging.save_frequency == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_metrics)

                # Early stopping
                if self.early_stopping:
                    if self.early_stopping(val_metrics['val_loss']):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            # Synchronize all processes
            self.dist.barrier()

        # SWA finalization
        if self.use_swa and self.is_main_process:
            logger.info("Updating batch normalization statistics for SWA model")
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)

        if self.is_main_process:
            logger.info(f"Self-supervised distillation completed. Best Val Acc: {self.best_val_acc:.2f}%")

        return dict(self.metrics_history)
