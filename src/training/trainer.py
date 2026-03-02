import itertools
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from torchvision.transforms.v2 import MixUp, CutMix, RandomChoice

from src.evaluation.metrics import evaluate_model
from src.losses.combined import BASDLoss
from src.models.teacher import TeacherModel, extract_intermediates


@torch.compiler.disable
def _extract_student(
    model: nn.Module, x: torch.Tensor, layer_indices: list[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Run student forward with hooks to capture intermediates and attention."""
    hooks = []
    captured_tokens = {}
    captured_attns = {}

    for idx in layer_indices:
        block = model.blocks[idx]

        def make_token_hook(i):
            def hook(mod, inp, out):
                captured_tokens[i] = out[:, 1:, :]
            return hook
        hooks.append(block.register_forward_hook(make_token_hook(idx)))

        def make_attn_hook(i):
            def hook(mod, inp, out):
                x_in = inp[0]
                B, N, C = x_in.shape
                nh = mod.num_heads
                hd = C // nh
                qkv = mod.qkv(x_in).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                captured_attns[i] = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            return hook
        hooks.append(block.attn.register_forward_hook(make_attn_hook(idx)))

    logits = model(x)
    for h in hooks:
        h.remove()

    return logits, captured_tokens, captured_attns


class Trainer:
    def __init__(
        self,
        student_model: nn.Module,
        config,
        accelerator: Accelerator,
        teacher: TeacherModel | None = None,
    ):
        self.accelerator = accelerator
        self.device = accelerator.device
        self.config = config
        self.distill = teacher is not None

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            student_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=0.05,
            fused=True,
        )

        warmup_epochs = max(1, int(0.05 * config.training.num_epochs))
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0 / warmup_epochs,
                    total_iters=warmup_epochs,
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.training.num_epochs,
                    eta_min=1e-6,
                ),
            ],
            milestones=[warmup_epochs],
        )

        if self.distill:
            self._teacher = teacher
            student_embed_dim = student_model.embed_dim
            student_num_tokens = student_model.patch_embed.num_patches
            cross_attn_num_heads = max(1, teacher.embed_dim // 64)

            self.basd_loss = BASDLoss(
                base_criterion=self.criterion,
                student_dim=student_embed_dim,
                teacher_dim=teacher.embed_dim,
                student_depth=student_model.depth,
                num_student_tokens=student_num_tokens,
                cross_attn_num_heads=cross_attn_num_heads,
                config=config.basd,
                student_num_heads=config.model.vit.num_heads,
                teacher_num_heads=teacher.num_heads,
            ).to(self.device)

            self.token_layers = list(self.basd_loss.token_layers)

            self.optimizer.add_param_group({
                "params": list(self.basd_loss.parameters()),
                "lr": config.training.learning_rate,
            })

        self.model, self.optimizer, self.scheduler = accelerator.prepare(
            student_model, self.optimizer, self.scheduler
        )

        if self.distill:
            accelerator.register_for_checkpointing(self.basd_loss)

        self.model = torch.compile(self.model, mode="max-autotune")

        self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))
        accelerator.register_for_checkpointing(self.ema_model)

        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.metrics_history = defaultdict(list)

        self.mixup_cutmix = RandomChoice([
            MixUp(alpha=0.8, num_classes=config.model.num_classes),
            CutMix(alpha=1.0, num_classes=config.model.num_classes),
        ])

    def save_checkpoint(self, name: str, epoch: int) -> None:
        checkpoint_dir = Path(self.config.run.output_dir) / self.config.run.name / "checkpoints" / name
        self.accelerator.save_state(str(checkpoint_dir))

        custom_state = {
            "epoch": epoch,
            "best_val_acc": self.best_val_acc,
            "metrics_history": self.metrics_history,
        }
        if self.distill:
            custom_state["uwso_temperature"] = self.basd_loss.uwso_temperature
        torch.save(custom_state, checkpoint_dir / "custom_state.pth")
        print(f"event=checkpoint_saved path={checkpoint_dir} epoch={epoch + 1} name={name}")

    def save_weights(self, filename: str, epoch: int, *, ema: bool = False) -> None:
        checkpoint_dir = Path(self.config.run.output_dir) / self.config.run.name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        source = self.ema_model.module if ema else self.model
        unwrapped = self.accelerator.unwrap_model(source)
        torch.save(
            {"epoch": epoch, "model_state_dict": unwrapped.state_dict()},
            checkpoint_dir / filename,
        )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        self.accelerator.load_state(checkpoint_path)

        custom = torch.load(
            Path(checkpoint_path) / "custom_state.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.best_val_acc = custom["best_val_acc"]
        self.metrics_history = defaultdict(list, custom["metrics_history"])
        if self.distill:
            self.basd_loss.uwso_temperature = custom["uwso_temperature"]
        return custom["epoch"] + 1

    def _train_epoch_baseline(
        self,
        train_loader: torch.utils.data.DataLoader,
    ) -> dict[str, Any]:
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs = batch["pixel_values"].to(self.device)
            targets = batch["label"].to(self.device)

            inputs, mixed_targets = self.mixup_cutmix(inputs, targets)

            with self.accelerator.autocast():
                logits = self.model(inputs)
                loss = self.criterion(logits, mixed_targets)

            self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                1.0,
            )
            self.optimizer.step()
            self.ema_model.update_parameters(self.model)
            self.optimizer.zero_grad()

            n = targets.size(0)
            total_loss += loss.item() * n
            predicted = logits.argmax(1)
            correct += predicted.eq(targets).sum().item()
            total += n

        return {
            "train_total_loss": total_loss / total,
            "train_acc": 100.0 * correct / total,
            "total_samples": total,
        }

    def _train_epoch_distill(
        self,
        train_loader: torch.utils.data.DataLoader,
    ) -> dict[str, Any]:
        _LOSS_KEYS = ("ce_loss", "rsd_loss", "dsgt_loss", "attn_loss")
        loss_accum = defaultdict(float)
        correct = 0
        total = 0

        for batch in train_loader:
            clean_imgs = batch["clean"].to(self.device)
            student_imgs = batch["augmented"].to(self.device)
            targets = batch["label"].to(self.device)

            student_imgs, mixed_targets = self.mixup_cutmix(student_imgs, targets)

            with self.accelerator.autocast():
                student_logits, s_tokens, s_attns = _extract_student(
                    self.model, student_imgs, self.token_layers,
                )

                teacher_tokens, teacher_attns = extract_intermediates(self._teacher, clean_imgs)

                loss, loss_dict = self.basd_loss(
                    student_logits,
                    mixed_targets,
                    s_tokens,
                    s_attns,
                    teacher_tokens,
                    teacher_attns,
                )

            self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(
                itertools.chain(self.model.parameters(), self.basd_loss.parameters()),
                1.0,
            )
            self.optimizer.step()
            self.basd_loss.project_to_stiefel()
            self.ema_model.update_parameters(self.model)
            self.optimizer.zero_grad()

            n = targets.size(0)
            loss_accum["total_loss"] += loss.item() * n
            for key in _LOSS_KEYS:
                loss_accum[key] += loss_dict[key] * n

            predicted = student_logits.argmax(1)
            correct += predicted.eq(targets).sum().item()
            total += n

        result = {f"train_{k}": v / total for k, v in loss_accum.items()}
        result["train_acc"] = 100.0 * correct / total
        result["total_samples"] = total
        return result

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        start_epoch: int,
    ) -> dict[str, list[Any]]:
        num_epochs = self.config.training.num_epochs

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            self.model.train()
            if self.distill:
                train_metrics = self._train_epoch_distill(train_loader)
            else:
                train_metrics = self._train_epoch_baseline(train_loader)

            val_metrics = evaluate_model(
                self.model, val_loader, self.device,
                criterion=self.criterion,
                num_classes=self.config.model.num_classes,
            )

            self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]["lr"]
            throughput = train_metrics["total_samples"] / epoch_time

            if self.distill:
                print(
                    f"event=epoch_summary mode=distill epoch={epoch + 1} num_epochs={num_epochs} "
                    f"train_loss={train_metrics['train_total_loss']:.6f} "
                    f"ce={train_metrics['train_ce_loss']:.6f} "
                    f"rsd={train_metrics['train_rsd_loss']:.6f} "
                    f"dsgt={train_metrics['train_dsgt_loss']:.6f} "
                    f"attn={train_metrics['train_attn_loss']:.6f} "
                    f"train_acc={train_metrics['train_acc']:.4f} "
                    f"val_acc={val_metrics['val_acc']:.4f} "
                    f"lr={current_lr:.8f} "
                    f"epoch_time_s={epoch_time:.3f} "
                    f"throughput_img_per_sec={throughput:.2f}"
                )
            else:
                print(
                    f"event=epoch_summary mode=baseline epoch={epoch + 1} num_epochs={num_epochs} "
                    f"train_loss={train_metrics['train_total_loss']:.6f} "
                    f"train_acc={train_metrics['train_acc']:.4f} "
                    f"val_acc={val_metrics['val_acc']:.4f} "
                    f"lr={current_lr:.8f} "
                    f"epoch_time_s={epoch_time:.3f} "
                    f"throughput_img_per_sec={throughput:.2f}"
                )

            for key, value in {**train_metrics, **val_metrics}.items():
                self.metrics_history[key].append(value)
            self.metrics_history["epoch_time"].append(epoch_time)
            self.metrics_history["throughput"].append(throughput)

            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self.save_checkpoint("best_model", epoch)
                self.save_weights("best_model.pth", epoch)

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}", epoch)

        self.save_weights("ema_model.pth", epoch, ema=True)
        print(f"event=train_complete best_val_acc={self.best_val_acc:.4f}")

        return self.metrics_history
