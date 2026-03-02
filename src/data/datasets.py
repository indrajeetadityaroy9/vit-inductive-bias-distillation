import os
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import torch
from datasets import Image, ClassLabel, load_dataset, load_dataset_builder
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    Normalize,
    RandomErasing,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToDtype,
    ToImage,
    TrivialAugmentWide,
)


_NUM_WORKERS = min(os.cpu_count(), 16)


@lru_cache(maxsize=8)
def _compute_channel_stats(
    hf_path: str, split: str, image_key: str, *, n_samples: int = 5000,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    print(f"event=computing_channel_stats dataset={hf_path} split={split} n_samples={n_samples}")
    ds = load_dataset(hf_path, split=split, streaming=True).take(n_samples)

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for example in ds:
        img = example[image_key].convert("RGB")
        arr = np.asarray(img, dtype=np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        pixel_sum += flat.sum(axis=0)
        pixel_sq_sum += (flat ** 2).sum(axis=0)
        pixel_count += flat.shape[0]

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    return tuple(mean.tolist()), tuple(std.tolist())


@lru_cache(maxsize=16)
def get_dataset_info(
    dataset_name: str, *, primary_dataset: str | None = None,
) -> dict[str, Any]:
    builder = load_dataset_builder(dataset_name)
    features = builder.info.features
    available_splits = set(builder.info.splits.keys())

    image_key = next(name for name, feat in features.items() if isinstance(feat, Image))
    label_key = next(name for name, feat in features.items() if isinstance(feat, ClassLabel))
    num_classes = features[label_key].num_classes
    class_names = features[label_key].names

    val_split_name = "validation" if "validation" in available_splits else "test"
    split_map = {"train": "train", "val": val_split_name}

    if (
        primary_dataset
        and primary_dataset != dataset_name
        and set(class_names) <= set(get_dataset_info(primary_dataset)["class_names"])
    ):
        parent = get_dataset_info(primary_dataset)
        mean, std = parent["mean"], parent["std"]
    else:
        parent = None
        mean, std = _compute_channel_stats(dataset_name, split_map["train"], image_key)

    result = {
        "image_key": image_key,
        "label_key": label_key,
        "num_classes": num_classes,
        "class_names": class_names,
        "split_map": split_map,
        "mean": mean,
        "std": std,
    }

    if parent is not None:
        result["parent_dataset"] = primary_dataset

    return result


@lru_cache(maxsize=16)
def get_subset_indices(subset_name: str, primary_dataset: str) -> tuple[int, ...]:
    subset_info = get_dataset_info(subset_name, primary_dataset=primary_dataset)
    parent_info = get_dataset_info(primary_dataset)

    parent_name_to_idx = {name: idx for idx, name in enumerate(parent_info["class_names"])}

    return tuple(sorted(parent_name_to_idx[name] for name in subset_info["class_names"]))


def _build_augmented_transform(
    image_size: int,
    *,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Compose:
    return Compose([
        RandomResizedCrop(image_size),
        RandomHorizontalFlip(),
        TrivialAugmentWide(),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean, std),
        RandomErasing(p=0.25),
    ])


def _build_eval_transform(
    image_size: int,
    *,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Compose:
    return Compose([
        Resize(image_size + 32),
        CenterCrop(image_size),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean, std),
    ])


def _single_transform(
    examples: dict,
    transform: Compose,
    image_key: str,
    label_key: str,
) -> dict:
    return {
        "pixel_values": [transform(img.convert("RGB")) for img in examples[image_key]],
        "label": examples[label_key],
    }


def _dual_transform(
    examples: dict,
    clean_tf: Compose,
    aug_tf: Compose,
    image_key: str,
    label_key: str,
) -> dict:
    images = [img.convert("RGB") for img in examples[image_key]]
    return {
        "clean": [clean_tf(img) for img in images],
        "augmented": [aug_tf(img) for img in images],
        "label": examples[label_key],
    }


def create_eval_loader(
    dataset_name: str,
    *,
    image_size: int,
    batch_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> DataLoader:
    info = get_dataset_info(dataset_name)
    transform = _build_eval_transform(image_size, mean=mean, std=std)
    image_key, label_key = info["image_key"], info["label_key"]

    ds = load_dataset(dataset_name, split=info["split_map"]["val"])
    ds.set_transform(
        lambda ex: _single_transform(ex, transform, image_key, label_key)
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )


def create_dataloaders(
    config,
    *,
    view_mode: Literal["dual", "single"],
) -> tuple[DataLoader, DataLoader]:
    info = get_dataset_info(config.data.dataset)
    mean, std = info["mean"], info["std"]
    image_size = config.model.vit.img_size
    image_key, label_key = info["image_key"], info["label_key"]

    aug_tf = _build_augmented_transform(image_size, mean=mean, std=std)

    train_ds = load_dataset(config.data.dataset, split=info["split_map"]["train"])

    if view_mode == "dual":
        eval_tf = _build_eval_transform(image_size, mean=mean, std=std)
        train_ds.set_transform(
            lambda ex: _dual_transform(ex, eval_tf, aug_tf, image_key, label_key)
        )
    else:
        train_ds.set_transform(
            lambda ex: _single_transform(ex, aug_tf, image_key, label_key)
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    val_loader = create_eval_loader(
        config.data.dataset,
        image_size=image_size,
        batch_size=config.data.batch_size,
        mean=mean,
        std=std,
    )

    return train_loader, val_loader
