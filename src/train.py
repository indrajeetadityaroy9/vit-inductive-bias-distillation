import math
from pathlib import Path

import hydra
import timm
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_dataset as hf_load_dataset
from omegaconf import DictConfig, OmegaConf, open_dict

from src.data.datasets import dataset_info, create_dataloaders, build_eval_transform
from src.evaluation.metrics import run_eval_suite, save_metrics
from src.models.teacher import TeacherModel, load_teacher, estimate_intrinsic_dim, probe_model
from src.resolvers import register_resolvers
from src.training.trainer import Trainer


def _apply_fan_in_init(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=(2.0 / m.weight.shape[1]) ** 0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            nn.init.normal_(m.weight, std=(2.0 / fan_out) ** 0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _create_student(
    model_name: str,
    *,
    num_classes: int,
    drop_path_rate: float,
    img_size: int,
    device: torch.device,
    arch_overrides: dict | None = None,
) -> nn.Module:
    kwargs = dict(
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        img_size=img_size,
    )
    if arch_overrides:
        kwargs.update(arch_overrides)
    model = timm.create_model(model_name, **kwargs)
    _apply_fan_in_init(model)
    model.set_grad_checkpointing(True)
    return model.to(device)


def _derive_from_teacher(teacher: TeacherModel, intrinsic_dim: int) -> dict:
    head_dim = teacher.embed_dim // teacher.heads_per_layer[0]
    D_s = math.ceil(intrinsic_dim / head_dim) * head_dim
    D_s = min(D_s, teacher.embed_dim)
    return {
        "embed_dim": D_s,
        "depth": teacher.depth,
        "num_heads": D_s // head_dim,
        "mlp_ratio": teacher.mlp_ratio,
    }


register_resolvers()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(config.run.seed)

    accelerator = Accelerator(mixed_precision="bf16")

    output_dir = Path(config.run.output_dir) / config.run.name
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = config.model.vit.img_size

    if config.training.mode == "baseline":
        student = _create_student(
            config.model.student_preset,
            num_classes=config.model.num_classes,
            drop_path_rate=config.model.drop_path_rate,
            img_size=img_size,
            device=accelerator.device,
        )

        train_loader, val_loader = create_dataloaders(config, view_mode="single")
        trainer = Trainer(student, config, accelerator=accelerator)
    else:
        teacher = load_teacher(
            config.basd.teacher_model_name, accelerator.device, img_size=img_size,
        )

        if teacher.feature_format == "token":
            ds_info = dataset_info(config.data.dataset)
            calib_tf = build_eval_transform(
                img_size, mean=teacher.mean, std=teacher.std,
                crop_ratio=config.data.eval_crop_ratio,
            )
            tokens_per_image = (img_size // config.model.vit.patch_size) ** 2
            num_calib = math.ceil(10 * teacher.embed_dim / tokens_per_image)

            calib_ds = hf_load_dataset(
                config.data.dataset, split=ds_info["train_split"], streaming=True,
            ).take(num_calib)
            calib_images = torch.stack([
                calib_tf(ex[ds_info["image_key"]].convert("RGB")) for ex in calib_ds
            ]).to(accelerator.device)

            intrinsic_dim = estimate_intrinsic_dim(teacher, calib_images)
            arch_overrides = _derive_from_teacher(teacher, intrinsic_dim)
            print(
                f"student_arch_derived intrinsic_dim={intrinsic_dim} "
                f"embed_dim={arch_overrides['embed_dim']} "
                f"depth={arch_overrides['depth']} num_heads={arch_overrides['num_heads']} "
                f"mlp_ratio={arch_overrides['mlp_ratio']:.1f}"
            )
        else:
            arch_overrides = None

        if arch_overrides:
            with open_dict(config):
                config.model.arch_overrides = arch_overrides

        student = _create_student(
            config.model.student_preset,
            num_classes=config.model.num_classes,
            drop_path_rate=config.model.drop_path_rate,
            img_size=img_size,
            device=accelerator.device,
            arch_overrides=arch_overrides,
        )

        student_info = probe_model(student, accelerator.device, img_size)
        print(
            f"student_probed embed_dim={student_info['embed_dim']} "
            f"depth={student_info['depth']} num_tokens={student_info['num_tokens']} "
            f"heads_per_layer={student_info['heads_per_layer']} "
            f"has_cls={student_info['has_cls_token']} "
            f"attn_subpath={student_info['attn_subpath']}"
        )

        train_loader, val_loader = create_dataloaders(
            config, view_mode="dual",
            teacher_stats=(teacher.mean, teacher.std),
        )
        trainer = Trainer(
            student, config, accelerator=accelerator,
            teacher=teacher, student_info=student_info,
        )

    OmegaConf.save(config, output_dir / "config.yaml")

    start_epoch = 0
    if config.checkpoint.resume_from:
        start_epoch = trainer.load_checkpoint(config.checkpoint.resume_from)

    trainer.train(train_loader, val_loader, start_epoch=start_epoch)

    trainer.optimizer.eval()
    model = accelerator.unwrap_model(trainer.model)
    results = run_eval_suite(
        model, config, accelerator.device,
        config_path=str(output_dir / "config.yaml"),
    )

    save_metrics(results, output_dir)


if __name__ == "__main__":
    main()
