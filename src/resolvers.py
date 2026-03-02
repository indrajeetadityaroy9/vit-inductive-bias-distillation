from omegaconf import OmegaConf

from src.data.datasets import dataset_info


def _num_classes(dataset_name: str) -> int:
    return dataset_info(dataset_name)["num_classes"]


def _label_smoothing(dataset_name: str) -> float:
    return 1.0 / dataset_info(dataset_name)["num_classes"]


def _eval_crop_ratio(img_size: int, patch_size: int) -> float:
    return img_size / (img_size + 2 * patch_size)


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("num_classes", _num_classes)
    OmegaConf.register_new_resolver("label_smoothing", _label_smoothing)
    OmegaConf.register_new_resolver("eval_crop_ratio", _eval_crop_ratio)
