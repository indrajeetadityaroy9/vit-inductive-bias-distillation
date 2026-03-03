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


def get_channel_stats(dataset_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    info = dataset_info(dataset_name)
    ds = load_dataset(dataset_name, split=info["train_split"], streaming=True).take(5000)

    mean = np.zeros(3, dtype=np.float64)
    m2 = np.zeros(3, dtype=np.float64)
    count = 0

    for example in ds:
        arr = np.asarray(example[info["image_key"]].convert("RGB"), dtype=np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        n = flat.shape[0]
        batch_mean = flat.mean(axis=0)
        batch_var = flat.var(axis=0)
        delta = batch_mean - mean
        new_count = count + n
        mean += delta * n / new_count
        m2 += batch_var * n + delta ** 2 * count * n / new_count
        count = new_count

    std = np.sqrt(m2 / count)
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
    ds.set_transform(lambda ex: {
        "pixel_values": [transform(img.convert("RGB")) for img in ex[image_key]],
        "label": ex[label_key],
    })

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

    aug_tf = Compose([
        RandomResizedCrop(image_size),
        RandomHorizontalFlip(),
        TrivialAugmentWide(),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean, std),
    ])

    train_ds = load_dataset(config.data.dataset, split=info["train_split"])

    if view_mode == "dual":
        teacher_mean, teacher_std = teacher_stats
        clean_tf = build_eval_transform(
            image_size, mean=teacher_mean, std=teacher_std, crop_ratio=crop_ratio,
        )
        train_ds.set_transform(lambda ex: {
            "clean": [clean_tf(img.convert("RGB")) for img in ex[image_key]],
            "augmented": [aug_tf(img.convert("RGB")) for img in ex[image_key]],
            "label": ex[label_key],
        })
    else:
        train_ds.set_transform(lambda ex: {
            "pixel_values": [aug_tf(img.convert("RGB")) for img in ex[image_key]],
            "label": ex[label_key],
        })

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
