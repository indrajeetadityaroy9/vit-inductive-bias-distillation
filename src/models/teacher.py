from typing import NamedTuple

import timm
import torch
import torch.nn as nn


class TeacherModel(NamedTuple):
    model: torch.nn.Module
    embed_dim: int
    num_heads: int
    layer_paths: list[str]
    attn_subpath: str | None
    has_cls_token: bool
    feature_format: str


def _probe_model(model: nn.Module, device: torch.device, img_size: int = 224) -> dict:
    """Dynamically discover model structure from any nn.Module."""
    embed_dim = getattr(model, 'embed_dim', None) or getattr(model, 'num_features', None)

    layer_paths = []
    for name in ('blocks', 'layers', 'stages'):
        container = getattr(model, name, None)
        if isinstance(container, (nn.Sequential, nn.ModuleList)):
            layer_paths = [f"{name}.{i}" for i in range(len(container))]
            break

    attn_subpath = None
    num_heads = 0
    if layer_paths:
        first_block = model.get_submodule(layer_paths[0])
        for child_name, child in first_block.named_children():
            if hasattr(child, 'num_heads'):
                attn_subpath = child_name
                num_heads = child.num_heads
                break
    if num_heads == 0:
        num_heads = max(1, embed_dim // 64)

    has_cls_token = any(n == 'cls_token' for n, _ in model.named_parameters())

    probe = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        if layer_paths:
            captured = {}
            mod = model.get_submodule(layer_paths[-1])
            h = mod.register_forward_hook(lambda m, i, o: captured.update(
                out=o if isinstance(o, torch.Tensor) else o[0]
            ))
            model(probe)
            h.remove()
            out = captured['out']
        else:
            out = model.forward_features(probe)

        if out.dim() == 4:
            feature_format = "nchw" if out.shape[1] > out.shape[3] else "nhwc"
        else:
            feature_format = "token"

    return {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': len(layer_paths),
        'layer_paths': layer_paths,
        'attn_subpath': attn_subpath,
        'has_cls_token': has_cls_token,
        'feature_format': feature_format,
    }


def load_teacher(model_name: str, device: torch.device, loader: str,
                 img_size: int = 224) -> TeacherModel:
    if loader == "dinov3":
        model = torch.hub.load("facebookresearch/dinov3", model_name)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    info = _probe_model(model, device, img_size)

    print(
        f"event=teacher_loaded model={model_name} embed_dim={info['embed_dim']} "
        f"num_layers={info['num_layers']} num_heads={info['num_heads']} "
        f"feature_format={info['feature_format']} has_cls={info['has_cls_token']} "
        f"attn_subpath={info['attn_subpath']}"
    )

    return TeacherModel(
        model=model,
        embed_dim=info['embed_dim'],
        num_heads=info['num_heads'],
        layer_paths=info['layer_paths'],
        attn_subpath=info['attn_subpath'],
        has_cls_token=info['has_cls_token'],
        feature_format=info['feature_format'],
    )


def _to_token_format(t: torch.Tensor, feature_format: str, has_cls_token: bool) -> torch.Tensor:
    """Reshape a captured tensor to [B, N, D] token format."""
    if feature_format == "nhwc" and t.dim() == 4:
        t = t.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
    elif feature_format == "nchw" and t.dim() == 4:
        t = t.flatten(2).transpose(1, 2)
    if has_cls_token and t.dim() == 3:
        t = t[:, 1:, :]
    return t


@torch.no_grad()
def extract_intermediates(
    teacher: TeacherModel, x: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    # CNN teachers: single-layer extraction with uniform attention.
    if teacher.feature_format != "token":
        features = teacher.model.forward_features(x)
        features = _to_token_format(features, teacher.feature_format, teacher.has_cls_token)
        B, N, _ = features.shape
        uniform_attn = torch.ones(
            B, 1, N + 1, N + 1, device=features.device, dtype=features.dtype,
        ) / (N + 1)
        return {0: features}, {0: uniform_attn}

    # ViT teachers: hook every layer for tokens and attention.
    hooks = []
    captured_tokens = {}
    captured_attns = {}

    for idx, path in enumerate(teacher.layer_paths):
        module = teacher.model.get_submodule(path)

        def make_token_hook(i):
            def hook(mod, inp, out):
                t = out if isinstance(out, torch.Tensor) else out[0]
                t = _to_token_format(t, teacher.feature_format, teacher.has_cls_token)
                captured_tokens[i] = t
            return hook
        hooks.append(module.register_forward_hook(make_token_hook(idx)))

        attn_mod = teacher.model.get_submodule(f"{path}.{teacher.attn_subpath}")

        def make_attn_hook(i):
            def hook(mod, inp, out):
                x_in = inp[0]
                B, N, C = x_in.shape
                nh = mod.num_heads
                hd = C // nh
                qkv = mod.qkv(x_in).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
                captured_attns[i] = attn.softmax(dim=-1)
            return hook
        hooks.append(attn_mod.register_forward_hook(make_attn_hook(idx)))

    teacher.model(x)
    for h in hooks:
        h.remove()

    return captured_tokens, captured_attns
