from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(data: dict) -> SimpleNamespace:
    return SimpleNamespace(**{
        k: _to_namespace(v) if isinstance(v, dict) else v
        for k, v in data.items()
    })


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> SimpleNamespace:
    from src.data.datasets import get_dataset_info

    defaults_path = Path(__file__).resolve().parent.parent / "configs" / "defaults.yaml"
    with open(defaults_path) as f:
        defaults = yaml.safe_load(f)

    with open(config_path) as f:
        overrides = yaml.safe_load(f)

    presets = defaults.pop("presets")

    merged = _deep_merge(defaults, overrides)

    preset_name = merged["model"]["student_preset"]
    merged["model"]["vit"].update(presets[preset_name])

    dataset_info = get_dataset_info(merged["data"]["dataset"])
    merged["model"]["num_classes"] = dataset_info["num_classes"]

    return _to_namespace(merged)


def save_config(config: SimpleNamespace, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(_to_dict(config), f, default_flow_style=False)
