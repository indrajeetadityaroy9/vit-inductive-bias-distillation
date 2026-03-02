from typing import NamedTuple

import timm
import torch
import torch.nn as nn
from src.losses.layer_selector import marchenko_pastur_rank


class TeacherModel(NamedTuple):
    model: torch.nn.Module
    embed_dim: int
    heads_per_layer: list[int]
    depth: int
    mlp_ratio: float
    layer_paths: list[str]
    attn_subpath: str | None
    has_cls_token: bool
    feature_format: str
    mean: tuple[float, ...]
    std: tuple[float, ...]


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _extract_qk(
    mod: nn.Module, x_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, N, C = x_in.shape
    nh = mod.num_heads
    hd = C // nh

    qkv = mod.qkv(x_in).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1]


def _make_attn_capture_hook(
    capture_dict: dict, layer_idx: int, *, apply_softmax: bool = True,
):
    """Unified attention capture hook for any attention module.

    Teacher uses apply_softmax=True (probability targets).
    Student uses apply_softmax=False (logits for KL divergence).
    """
    def hook(mod, inp, out):
        x_in = inp[0]
        q, k = _extract_qk(mod, x_in)
        hd = q.shape[-1]
        attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        capture_dict[layer_idx] = attn.softmax(dim=-1) if apply_softmax else attn

    return hook


def probe_model(model: nn.Module, device: torch.device, img_size: int) -> dict:
    embed_dim = getattr(model, 'embed_dim', None) or getattr(model, 'num_features', None)

    layer_paths = []
    for name in ('blocks', 'layers', 'stages'):
        container = getattr(model, name, None)
        if isinstance(container, (nn.Sequential, nn.ModuleList)):
            layer_paths = [f"{name}.{i}" for i in range(len(container))]
            break

    attn_subpath = None
    heads_per_layer = []
    mlp_ratio = 0.0

    for path in layer_paths:
        block = model.get_submodule(path)
        block_heads = 0

        for child_name, child in block.named_children():
            if hasattr(child, 'num_heads'):
                if attn_subpath is None:
                    attn_subpath = child_name
                block_heads = child.num_heads
                break

        heads_per_layer.append(block_heads)

        if mlp_ratio == 0.0:
            for child_name, child in block.named_children():
                if hasattr(child, 'fc1'):
                    mlp_ratio = child.fc1.out_features / embed_dim
                    break
                if hasattr(child, 'mlp') and hasattr(child.mlp, 'fc1'):
                    mlp_ratio = child.mlp.fc1.out_features / embed_dim
                    break

    has_cls_token = any(n == 'cls_token' for n, _ in model.named_parameters())

    probe = torch.zeros(1, 3, img_size, img_size, device=device)
    num_tokens = 0
    with torch.no_grad():
        if layer_paths:
            captured = {}
            mod = model.get_submodule(layer_paths[-1])
            h = mod.register_forward_hook(lambda m, i, o: captured.update(out=o))
            model(probe)
            h.remove()
            out = captured['out']
        else:
            out = model.forward_features(probe)

        if out.dim() == 4:
            feature_format = "nchw" if out.shape[1] > out.shape[3] else "nhwc"
        else:
            feature_format = "token"
            num_tokens = out.shape[1] - int(has_cls_token)

    # CNN teachers: single synthetic head for uniform attention
    if feature_format != "token":
        heads_per_layer = [1]

    return {
        'embed_dim': embed_dim,
        'heads_per_layer': heads_per_layer,
        'depth': len(layer_paths),
        'mlp_ratio': mlp_ratio,
        'layer_paths': layer_paths,
        'attn_subpath': attn_subpath,
        'has_cls_token': has_cls_token,
        'feature_format': feature_format,
        'num_tokens': num_tokens,
    }


def load_teacher(model_name: str, device: torch.device,
                 img_size: int) -> TeacherModel:
    if model_name.startswith("dinov2_"):
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        cfg = model.pretrained_cfg
        mean, std = tuple(cfg['mean']), tuple(cfg['std'])

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    info = probe_model(model, device, img_size)

    print(
        f"event=teacher_loaded model={model_name} embed_dim={info['embed_dim']} "
        f"depth={info['depth']} heads_per_layer={info['heads_per_layer']} "
        f"mlp_ratio={info['mlp_ratio']:.1f} "
        f"feature_format={info['feature_format']} has_cls={info['has_cls_token']} "
        f"attn_subpath={info['attn_subpath']} "
        f"mean={mean} std={std}"
    )

    return TeacherModel(
        model=model,
        embed_dim=info['embed_dim'],
        heads_per_layer=info['heads_per_layer'],
        depth=info['depth'],
        mlp_ratio=info['mlp_ratio'],
        layer_paths=info['layer_paths'],
        attn_subpath=info['attn_subpath'],
        has_cls_token=info['has_cls_token'],
        feature_format=info['feature_format'],
        mean=mean,
        std=std,
    )


def _to_token_format(t: torch.Tensor, feature_format: str, has_cls_token: bool) -> torch.Tensor:
    if feature_format == "nhwc":
        t = t.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
    elif feature_format == "nchw":
        t = t.flatten(2).transpose(1, 2)
    if has_cls_token:
        t = t[:, 1:, :]
    return t


@torch.no_grad()
def estimate_intrinsic_dim(teacher: TeacherModel, images: torch.Tensor) -> int:
    """Estimate teacher intrinsic dimensionality via Marchenko-Pastur rank.

    Runs calibration images through the teacher, captures last-layer token
    representations, and returns the MP rank — the number of eigenvalues
    above the noise threshold lambda_plus = sigma^2 * (1 + sqrt(q))^2.
    """
    captured = {}
    mod = teacher.model.get_submodule(teacher.layer_paths[-1])
    h = mod.register_forward_hook(lambda m, i, o: captured.update(out=o))
    teacher.model(images)
    h.remove()

    tokens = _to_token_format(captured['out'], teacher.feature_format, teacher.has_cls_token)
    flat = tokens.reshape(-1, tokens.shape[-1]).float()
    return marchenko_pastur_rank(flat)


@torch.no_grad()
def extract_intermediates(
    teacher: TeacherModel, x: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    if teacher.feature_format != "token":
        features = teacher.model.forward_features(x)
        features = _to_token_format(features, teacher.feature_format, teacher.has_cls_token)
        B, N, _ = features.shape
        uniform_attn = torch.ones(
            B, 1, N + 1, N + 1, device=features.device, dtype=features.dtype,
        ) / (N + 1)
        return {0: features}, {0: uniform_attn}

    hooks = []
    captured_tokens = {}
    captured_attns = {}

    for idx, path in enumerate(teacher.layer_paths):
        module = teacher.model.get_submodule(path)

        def make_token_hook(i):
            def hook(mod, inp, out):
                captured_tokens[i] = _to_token_format(out, teacher.feature_format, teacher.has_cls_token)
            return hook
        hooks.append(module.register_forward_hook(make_token_hook(idx)))

        if teacher.attn_subpath is not None:
            attn_mod = teacher.model.get_submodule(f"{path}.{teacher.attn_subpath}")
            hooks.append(attn_mod.register_forward_hook(
                _make_attn_capture_hook(captured_attns, idx, apply_softmax=True)
            ))

    teacher.model(x)
    for h in hooks:
        h.remove()

    return captured_tokens, captured_attns
