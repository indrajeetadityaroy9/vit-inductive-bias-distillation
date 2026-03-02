import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCalibrationError,
)

from omegaconf import OmegaConf

from src.data.datasets import (
    create_eval_loader,
    dataset_info,
    get_channel_stats,
    get_subset_indices,
)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    num_classes: int,
    valid_indices: list[int] | None = None,
) -> dict[str, Any]:
    model.eval()

    acc_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro").to(device)
    acc_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5, average="micro").to(device)
    ece = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm="l1").to(device)

    total_loss = 0.0
    total = 0

    for batch in data_loader:
        inputs = batch["pixel_values"].to(device)
        targets = batch["label"].to(device)

        outputs = model(inputs)

        if valid_indices is not None:
            outputs = outputs[:, valid_indices]

        total_loss += criterion(outputs, targets).item() * inputs.size(0)
        total += targets.size(0)

        acc_top1.update(outputs, targets)
        acc_top5.update(outputs, targets)
        ece.update(outputs, targets)

    return {
        "val_acc": 100.0 * acc_top1.compute().item(),
        "val_acc_top5": 100.0 * acc_top5.compute().item(),
        "loss": total_loss / total,
        "ece": ece.compute().item(),
    }


@torch.no_grad()
def measure_efficiency(
    model: nn.Module,
    device: torch.device,
    *,
    image_size: int,
    in_channels: int = 3,
    batch_size: int = 64,
    num_warmup: int = 50,
    num_batches: int = 200,
) -> dict[str, float]:
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    param_count_m = param_count / 1e6

    dummy = torch.randn(1, in_channels, image_size, image_size, device=device)
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(dummy)
    gflops = flop_counter.get_total_flops() / 1e9

    dummy_batch = torch.randn(batch_size, in_channels, image_size, image_size, device=device)
    for _ in range(num_warmup):
        model(dummy_batch)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_batches):
        model(dummy_batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    throughput = (batch_size * num_batches) / elapsed

    return {
        "param_count": param_count,
        "param_count_m": param_count_m,
        "gflops": gflops,
        "throughput_img_per_sec": throughput,
    }


def evaluate_on_datasets(
    model: nn.Module,
    config,
    device: torch.device,
    criterion: nn.Module,
    *,
    dataset_names: list[str],
    primary_dataset: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_results = {}
    robustness_results = {}

    mean, std = get_channel_stats(primary_dataset)

    crop_ratio = config.data.eval_crop_ratio

    primary_num_classes = dataset_info(primary_dataset)["num_classes"]

    for ds_name in dataset_names:
        loader = create_eval_loader(
            ds_name,
            image_size=config.model.vit.img_size,
            batch_size=config.data.batch_size,
            mean=mean,
            std=std,
            crop_ratio=crop_ratio,
        )

        valid_indices = get_subset_indices(ds_name, primary_dataset)
        num_classes = len(valid_indices) if valid_indices is not None else primary_num_classes

        metrics = evaluate_model(
            model, loader, device, criterion,
            num_classes=num_classes, valid_indices=valid_indices,
        )

        if ds_name == primary_dataset:
            primary_results = metrics
        else:
            robustness_results[ds_name] = metrics

        print(
            f"event=eval_dataset dataset={ds_name} "
            f"top1={metrics['val_acc']:.4f} top5={metrics['val_acc_top5']:.4f} "
            f"loss={metrics['loss']:.6f} ece={metrics['ece']:.6f}"
        )

    return primary_results, robustness_results


def run_eval_suite(
    model: nn.Module,
    config,
    device: torch.device,
    *,
    config_path: str,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    datasets_to_eval = [config.data.dataset] + list(config.data.eval_datasets)

    primary_results, robustness_results = evaluate_on_datasets(
        model, config, device, criterion,
        dataset_names=datasets_to_eval,
        primary_dataset=config.data.dataset,
    )

    efficiency = measure_efficiency(
        model, device,
        image_size=config.model.vit.img_size,
    )

    print(
        f"event=eval_efficiency dataset={config.data.dataset} "
        f"params_m={efficiency['param_count_m']:.4f} gflops={efficiency['gflops']:.4f} "
        f"throughput_img_per_sec={efficiency['throughput_img_per_sec']:.2f}"
    )

    return {
        "run": {"name": config.run.name, "config": config_path},
        "primary": {
            "dataset": config.data.dataset,
            **primary_results,
        },
        "robustness": robustness_results,
        "efficiency": efficiency,
    }


def save_metrics(
    results: dict[str, Any],
    output_dir: Path,
    config,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    return metrics_path
