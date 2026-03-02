import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from schedulefree import AdamWScheduleFree

from torchvision.transforms.v2 import MixUp, CutMix, RandomChoice

from src.evaluation.metrics import evaluate_model
from src.losses.combined import BASDLoss
from src.models.teacher import TeacherModel, extract_intermediates, _make_attn_capture_hook


@torch.compiler.disable
def _extract_student(
    model: nn.Module, x: torch.Tensor, layer_indices: list[int],
    *, layer_paths: list[str], attn_subpath: str | None, has_cls_token: bool,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    hooks = []
    captured_tokens = {}
    captured_attns = {}

    for idx in layer_indices:
        block = model.get_submodule(layer_paths[idx])

        def make_token_hook(i, _has_cls=has_cls_token):
            def hook(mod, inp, out):
                captured_tokens[i] = out[:, 1:, :] if _has_cls else out
            return hook
        hooks.append(block.register_forward_hook(make_token_hook(idx)))

        if attn_subpath is not None:
            attn_mod = model.get_submodule(f"{layer_paths[idx]}.{attn_subpath}")
            hooks.append(attn_mod.register_forward_hook(
                _make_attn_capture_hook(captured_attns, idx, apply_softmax=False)
            ))

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
        *,
        student_info: dict | None = None,
    ):
        self.accelerator = accelerator
        self.device = accelerator.device
        self.config = config
        self.distill = teacher is not None

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
        self.optimizer = AdamWScheduleFree(
            student_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        if self.distill:
            self._teacher = teacher
            self._student_layer_paths = student_info['layer_paths']
            self._student_attn_subpath = student_info['attn_subpath']
            self._student_has_cls = student_info['has_cls_token']

            student_heads = student_info['heads_per_layer']
            teacher_heads = teacher.heads_per_layer

            self.basd_loss = BASDLoss(
                base_criterion=self.criterion,
                student_dim=student_info['embed_dim'],
                teacher_dim=teacher.embed_dim,
                student_depth=student_info['depth'],
                num_student_tokens=student_info['num_tokens'],
                cross_attn_num_heads=teacher.heads_per_layer[0],
                config=config.basd,
                student_heads_per_layer=[
                    student_heads[min(i, len(student_heads) - 1)]
                    for i in range(config.basd.num_extraction_points)
                ],
                teacher_heads_per_layer=[
                    teacher_heads[min(i, len(teacher_heads) - 1)]
                    for i in range(config.basd.num_extraction_points)
                ],
                teacher_has_cls_token=teacher.has_cls_token,
                teacher_feature_format=teacher.feature_format,
            ).to(self.device)

            self.optimizer.add_param_group({
                "params": list(self.basd_loss.parameters()),
            })

        student_model = torch.compile(student_model, mode="max-autotune")

        self.model, self.optimizer = accelerator.prepare(
            student_model, self.optimizer
        )

        if self.distill:
            accelerator.register_for_checkpointing(self.basd_loss)

        self.best_val_acc = 0.0
        self.metrics_history = defaultdict(list)

        self.mixup_cutmix = RandomChoice([
            MixUp(alpha=1.0, num_classes=config.model.num_classes),
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
        torch.save(custom_state, checkpoint_dir / "custom_state.pth")

    def save_weights(self, filename: str, epoch: int) -> None:
        checkpoint_dir = Path(self.config.run.output_dir) / self.config.run.name / "checkpoints"
        unwrapped = self.accelerator.unwrap_model(self.model)
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
            self.optimizer.step()
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
                    self.model, student_imgs, self.basd_loss.token_layers,
                    layer_paths=self._student_layer_paths,
                    attn_subpath=self._student_attn_subpath,
                    has_cls_token=self._student_has_cls,
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
            self.optimizer.step()
            self.basd_loss.project_to_stiefel()
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
            epoch_start_time = time.time()

            self.optimizer.train()
            self.model.train()
            if self.distill:
                train_metrics = self._train_epoch_distill(train_loader)
            else:
                train_metrics = self._train_epoch_baseline(train_loader)

            self.optimizer.eval()
            val_metrics = evaluate_model(
                self.model, val_loader, self.device,
                criterion=self.criterion,
                num_classes=self.config.model.num_classes,
            )

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

            self.save_checkpoint("latest", epoch)

        self.save_weights("final_model.pth", num_epochs - 1)
        print(f"event=train_complete best_val_acc={self.best_val_acc:.4f}")

        return self.metrics_history
