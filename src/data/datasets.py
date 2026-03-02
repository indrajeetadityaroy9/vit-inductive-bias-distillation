from functools import lru_cache
from typing import Literal

import numpy as np
import torch
from datasets import ClassLabel, Image, load_dataset, load_dataset_builder
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToDtype,
    ToImage,
    TrivialAugmentWide,
)


_NUM_WORKERS = 8

@lru_cache(maxsize=8)
def dataset_info(dataset_name: str) -> dict:
    builder = load_dataset_builder(dataset_name)
    features = builder.info.features
    available_splits = set(builder.info.splits.keys())

    image_key = next(n for n, f in features.items() if isinstance(f, Image))
    label_key = next(n for n, f in features.items() if isinstance(f, ClassLabel))
    feat = features[label_key]

    train_split = "train" if "train" in available_splits else None
    eval_split = (
        "validation" if "validation" in available_splits
        else "test" if "test" in available_splits
        else "train"
    )

    return {
        "image_key": image_key,
        "label_key": label_key,
        "num_classes": feat.num_classes,
        "class_names": tuple(feat.names),
        "train_split": train_split,
        "eval_split": eval_split,
    }


@lru_cache(maxsize=4)
def get_channel_stats(dataset_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    info = dataset_info(dataset_name)
    ds = load_dataset(dataset_name, split=info["train_split"], streaming=True).take(5000)

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for example in ds:
        arr = np.asarray(example[info["image_key"]].convert("RGB"), dtype=np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        pixel_sum += flat.sum(axis=0)
        pixel_sq_sum += (flat ** 2).sum(axis=0)
        pixel_count += flat.shape[0]

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    return tuple(mean.tolist()), tuple(std.tolist())


def get_subset_indices(dataset_name: str, parent_name: str) -> tuple[int, ...] | None:
    child_names = dataset_info(dataset_name)["class_names"]
    parent_names = dataset_info(parent_name)["class_names"]
    if set(child_names) == set(parent_names):
        return None
    parent_map = {name: idx for idx, name in enumerate(parent_names)}
    return tuple(parent_map[name] for name in child_names)


def build_eval_transform(
    image_size: int,
    *,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    crop_ratio: float,
) -> Compose:
    resize_size = round(image_size / crop_ratio)
    return Compose([
        Resize(resize_size),
        CenterCrop(image_size),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean, std),
    ])


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
    ])


def _apply_transform(
    examples: dict,
    transform: Compose,
    image_key: str,
    label_key: str,
) -> dict:
    return {
        "pixel_values": [transform(img.convert("RGB")) for img in examples[image_key]],
        "label": examples[label_key],
    }


def _apply_dual_transform(
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
    crop_ratio: float,
) -> DataLoader:
    info = dataset_info(dataset_name)
    transform = build_eval_transform(image_size, mean=mean, std=std, crop_ratio=crop_ratio)
    image_key, label_key = info["image_key"], info["label_key"]

    ds = load_dataset(dataset_name, split=info["eval_split"])
    ds.set_transform(
        lambda ex: _apply_transform(ex, transform, image_key, label_key)
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
    teacher_stats: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> tuple[DataLoader, DataLoader]:
    info = dataset_info(config.data.dataset)
    mean, std = get_channel_stats(config.data.dataset)
    image_size = config.model.vit.img_size
    image_key, label_key = info["image_key"], info["label_key"]
    crop_ratio = config.data.eval_crop_ratio

    aug_tf = _build_augmented_transform(image_size, mean=mean, std=std)

    train_ds = load_dataset(config.data.dataset, split=info["train_split"])

    if view_mode == "dual":
        teacher_mean, teacher_std = teacher_stats
        clean_tf = build_eval_transform(
            image_size, mean=teacher_mean, std=teacher_std, crop_ratio=crop_ratio,
        )
        train_ds.set_transform(
            lambda ex: _apply_dual_transform(ex, clean_tf, aug_tf, image_key, label_key)
        )
    else:
        train_ds.set_transform(
            lambda ex: _apply_transform(ex, aug_tf, image_key, label_key)
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
        crop_ratio=crop_ratio,
    )

    return train_loader, val_loader
