import argparse
from pathlib import Path

import torch
from accelerate import Accelerator

from src.config import load_config
from src.evaluation.metrics import run_eval_suite, save_metrics
from src.models.deit import DeiT


def main() -> None:
    parser = argparse.ArgumentParser(description="MCSD Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_config(args.config)
    torch.manual_seed(config.run.seed)

    accelerator = Accelerator()

    model = DeiT(config.model.vit, config.model).to(accelerator.device)

    ckpt = torch.load(config.checkpoint.path, map_location=accelerator.device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"event=checkpoint_loaded path={config.checkpoint.path} epoch={ckpt['epoch']}")

    results = run_eval_suite(
        model, config, accelerator.device,
        config_path=args.config,
    )

    output_dir = Path(config.run.output_dir) / config.run.name
    metrics_path = save_metrics(results, output_dir, config)
    print(f"event=metrics_saved path={metrics_path}")


if __name__ == "__main__":
    main()
