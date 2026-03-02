from pathlib import Path

import hydra
import timm
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf

from src.evaluation.metrics import run_eval_suite, save_metrics
from src.resolvers import register_resolvers


register_resolvers()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(config.run.seed)

    accelerator = Accelerator()

    kwargs = dict(
        pretrained=False,
        num_classes=config.model.num_classes,
        img_size=config.model.vit.img_size,
        **OmegaConf.to_container(config.model.arch_overrides, resolve=True),
    )
    model = timm.create_model(config.model.student_preset, **kwargs).to(accelerator.device)

    ckpt = torch.load(config.checkpoint.path, map_location=accelerator.device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"event=checkpoint_loaded path={config.checkpoint.path} epoch={ckpt['epoch']}")

    output_dir = Path(config.run.output_dir) / config.run.name
    OmegaConf.save(config, output_dir / "config.yaml")

    results = run_eval_suite(
        model, config, accelerator.device,
        config_path=str(output_dir / "config.yaml"),
    )

    save_metrics(results, output_dir, config)


if __name__ == "__main__":
    main()
