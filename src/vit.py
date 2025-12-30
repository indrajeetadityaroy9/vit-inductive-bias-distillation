"""
Data-efficient Image Transformer (DeiT) implementation with distillation support.

Based on: "Training data-efficient image transformers & distillation through attention"
(Touvron et al., 2021) https://arxiv.org/abs/2012.12877

DeiT-Tiny configuration for small datasets (MNIST, CIFAR-10):
- embed_dim: 192
- depth: 12
- num_heads: 3
- patch_size: 4 (for 28x28 and 32x32 images)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# Check PyTorch version for SDPA (Flash Attention)
PYTORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
HAS_SDPA = PYTORCH_VERSION >= (2, 0)


def interpolate_pos_embed(pos_embed, target_num_patches, num_extra_tokens=2):
    """
    Interpolate positional embeddings for different resolutions.

    This is critical for using the same model architecture across datasets
    with different image sizes (e.g., MNIST 28x28 -> 49 patches, CIFAR 32x32 -> 64 patches).

    Args:
        pos_embed: Original positional embedding (1, orig_patches + num_extra_tokens, embed_dim)
        target_num_patches: Target number of patches
        num_extra_tokens: Number of special tokens (cls + dist = 2)

    Returns:
        Interpolated positional embedding
    """
    # Separate class/dist tokens from patch embeddings
    extra_tokens = pos_embed[:, :num_extra_tokens]
    patch_pos = pos_embed[:, num_extra_tokens:]

    orig_num_patches = patch_pos.shape[1]
    if orig_num_patches == target_num_patches:
        return pos_embed

    # Reshape to 2D grid
    orig_size = int(math.sqrt(orig_num_patches))
    embed_dim = patch_pos.shape[-1]

    # SAFETY: Ensure square grids (prevents silent shape corruption)
    assert orig_size * orig_size == orig_num_patches, \
        f"Non-square patch grid: {orig_num_patches} patches (sqrt={orig_size})"

    patch_pos = patch_pos.reshape(1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)

    # Interpolate using bicubic
    target_size = int(math.sqrt(target_num_patches))

    # SAFETY: Ensure target is also square
    assert target_size * target_size == target_num_patches, \
        f"Non-square target grid: {target_num_patches} patches (sqrt={target_size})"
    patch_pos = F.interpolate(
        patch_pos,
        size=(target_size, target_size),
        mode='bicubic',
        align_corners=False
    )

    # Reshape back
    patch_pos = patch_pos.permute(0, 2, 3, 1).flatten(1, 2)
    return torch.cat([extra_tokens, patch_pos], dim=1)


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) regularization.

    Randomly drops entire residual branches during training with probability drop_prob.
    At test time, scales the output by (1 - drop_prob) for proper expectation.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Shape: (batch_size, 1, 1, ...) for proper broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using Conv2d.

    Converts input image into a sequence of patch embeddings.
    For small images (28x28, 32x32), use patch_size=4.
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial dims and transpose: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class HybridPatchEmbed(nn.Module):
    """
    Hybrid Patch Embedding with optional Conv Stem.

    Adds a 2-layer conv stem before patch projection for better local feature extraction.
    This restores some locality bias without dataset-specific branching.

    When use_conv_stem=True:
        Input -> Conv3x3 -> GELU -> Conv3x3 -> GELU -> PatchProjection -> Output
    When use_conv_stem=False:
        Equivalent to standard PatchEmbed
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192, use_conv_stem=False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.use_conv_stem = use_conv_stem

        if use_conv_stem:
            # 2-layer conv stem for local feature extraction
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
            # Patch projection after stem (input is already embed_dim channels)
            self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.stem = None
            # Standard patch projection
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Apply conv stem if enabled
        if self.stem is not None:
            x = self.stem(x)
        # Patch projection: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial dims and transpose: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module using PyTorch's scaled_dot_product_attention.

    Uses F.scaled_dot_product_attention (SDPA) which automatically dispatches to
    Flash Attention v2 on H100 GPUs for 25-40% speedup and reduced memory usage.

    Supports optional attention weight extraction for analytics via return_attention
    parameter. Note: When return_attention=True, uses manual attention computation
    instead of SDPA (slower but provides attention weights).
    """

    def __init__(self, embed_dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Storage for attention weights (used by AttentionDistanceAnalyzer)
        self.attn_weights = None

    def forward(self, x, return_attention=False):
        B, N, C = x.shape

        # QKV projection and reshape: (B, N, 3*C) -> (3, B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if return_attention:
            # Manual attention computation to extract weights
            # Cannot use SDPA as it doesn't return attention weights
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

            # Store attention weights for later extraction (B, heads, N, N)
            self.attn_weights = attn.detach()

            attn = self.attn_drop(attn)
            x = attn @ v
        elif HAS_SDPA:
            # Use SDPA for automatic Flash Attention dispatch on H100
            # This provides 25-40% speedup and reduced memory usage
            dropout_p = self.attn_drop.p if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            # Fallback for PyTorch < 2.0
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, self.attn_weights
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with GELU activation.

    Standard transformer MLP: Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block with Pre-LayerNorm.

    Architecture: LN -> MHSA -> residual -> LN -> MLP -> residual
    Includes stochastic depth (DropPath) for regularization.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeiT(nn.Module):
    """
    Data-efficient Image Transformer (DeiT) with distillation token.

    DeiT-Tiny configuration for small datasets:
    - embed_dim: 192
    - depth: 12
    - num_heads: 3
    - patch_size: 4

    Key features:
    - Distillation token for knowledge distillation from CNN teacher
    - Positional embedding interpolation for different resolutions
    - Grayscale to RGB conversion for single-channel inputs
    - Linearly scaled drop path rates across depth

    Args (from config dict):
        in_channels: Input channels (1 for MNIST, 3 for CIFAR)
        num_classes: Number of output classes
        img_size: Input image size (28 or 32)
        patch_size: Patch size (default 4)
        embed_dim: Embedding dimension (default 192)
        depth: Number of transformer blocks (default 12)
        num_heads: Number of attention heads (default 3)
        mlp_ratio: MLP expansion ratio (default 4.0)
        drop_rate: Dropout rate (default 0.0)
        attn_drop_rate: Attention dropout rate (default 0.0)
        drop_path_rate: Maximum stochastic depth rate (default 0.1)
        distillation: Whether to use distillation token (default True)
        convert_grayscale: Whether to convert 1-ch to 3-ch (default True)
    """

    def __init__(self, config):
        super().__init__()

        # Parse config
        self.in_channels = config.get('in_channels', 3)
        self.num_classes = config.get('num_classes', 10)
        self.img_size = config.get('img_size', 32)
        self.patch_size = config.get('patch_size', 4)
        self.embed_dim = config.get('embed_dim', 192)
        self.depth = config.get('depth', 12)
        self.num_heads = config.get('num_heads', 3)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.drop_rate = config.get('drop_rate', 0.0)
        self.attn_drop_rate = config.get('attn_drop_rate', 0.0)
        self.drop_path_rate = config.get('drop_path_rate', 0.1)
        self.distillation = config.get('distillation', True)
        self.convert_grayscale = config.get('convert_grayscale', True)

        # New generalized improvements
        self.use_conv_stem = config.get('use_conv_stem', False)
        self.cls_token_dropout = config.get('cls_token_dropout', 0.0)
        self.inference_mode = config.get('inference_mode', 'avg')

        # Handle grayscale to RGB conversion
        if self.convert_grayscale and self.in_channels == 1:
            self.channel_expand = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            actual_in_channels = 3
        else:
            self.channel_expand = None
            actual_in_channels = self.in_channels

        # Patch embedding (use HybridPatchEmbed for optional conv stem)
        self.patch_embed = HybridPatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=actual_in_channels,
            embed_dim=self.embed_dim,
            use_conv_stem=self.use_conv_stem
        )
        num_patches = self.patch_embed.num_patches

        # Class and distillation tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if self.distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            num_extra_tokens = 2
        else:
            self.dist_token = None
            num_extra_tokens = 1

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + num_extra_tokens, self.embed_dim)
        )
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # Transformer blocks with linearly scaled drop path
        drop_path_rates = torch.linspace(0, self.drop_path_rate, self.depth).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=drop_path_rates[i]
            )
            for i in range(self.depth)
        ])

        # Output normalization and heads
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        if self.distillation:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes)
        else:
            self.head_dist = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal distribution."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        # Initialize linear layers and layer norms
        self.apply(self._init_weights_module)

    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            # BatchNorm initialization for conv stem
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            # Fan-out initialization for conv layers
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            - Training with distillation: (cls_logits, dist_logits)
            - Inference or no distillation: based on inference_mode ('cls', 'dist', 'avg')
        """
        # Channel expansion for grayscale
        if self.channel_expand is not None:
            x = self.channel_expand(x)

        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend class token (and distillation token if enabled)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output normalization
        x = self.norm(x)

        # Class token dropout: randomly replace cls token with mean of patch tokens
        # This forces patch tokens to carry semantic information (regularization)
        if self.training and self.cls_token_dropout > 0:
            mask = torch.rand(B, device=x.device) < self.cls_token_dropout
            # Compute mean of patch tokens (skip cls and dist tokens)
            num_special_tokens = 2 if self.distillation else 1
            patch_mean = x[:, num_special_tokens:, :].mean(dim=1)  # (B, embed_dim)
            # Replace cls token where mask is True
            x = x.clone()  # Avoid in-place modification for autograd
            x[:, 0, :] = torch.where(mask.unsqueeze(-1), patch_mean, x[:, 0, :])

        # Classification
        cls_out = self.head(x[:, 0])

        if self.distillation and self.head_dist is not None:
            dist_out = self.head_dist(x[:, 1])
            if self.training:
                # Return both outputs for distillation loss
                return cls_out, dist_out
            else:
                # Inference: use configured inference mode
                if self.inference_mode == 'cls':
                    return cls_out
                elif self.inference_mode == 'dist':
                    return dist_out
                else:  # 'avg' (default)
                    return (cls_out + dist_out) / 2

        return cls_out

    def forward_features(self, x):
        """Extract features before classification head (for visualization)."""
        if self.channel_expand is not None:
            x = self.channel_expand(x)

        x = self.patch_embed(x)

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def get_classifier_outputs(self, x):
        """
        Get separate cls and dist outputs for evaluation.

        Returns:
            cls_out: Classification head output
            dist_out: Distillation head output (or None)
            avg_out: Averaged output
        """
        if self.channel_expand is not None:
            x = self.channel_expand(x)

        x = self.patch_embed(x)

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_out = self.head(x[:, 0])

        if self.distillation and self.head_dist is not None:
            dist_out = self.head_dist(x[:, 1])
            avg_out = (cls_out + dist_out) / 2
            return cls_out, dist_out, avg_out

        return cls_out, None, cls_out

    def forward_with_intermediates(self, x, layer_indices=None, return_cls_only=False):
        """
        Forward pass that returns intermediate layer outputs for distillation.

        Used for CST-style self-supervised distillation where we match
        intermediate token representations between teacher and student.

        Args:
            x: Input tensor of shape (B, C, H, W)
            layer_indices: List of layer indices to capture (0-indexed, e.g., [6, 11])
                          If None, returns only final output
            return_cls_only: If True, return only CLS token (B, 1, D) instead of patches

        Returns:
            dict with keys:
                - 'output': Final classification output(s) - tuple during training
                - 'intermediates': Dict mapping layer_idx -> (B, N_patches, embed_dim) or (B, 1, embed_dim)
                - 'patch_tokens': Final patch/CLS tokens before head
        """
        if layer_indices is None:
            layer_indices = []

        intermediates = {}

        # Channel expansion for grayscale
        if self.channel_expand is not None:
            x = self.channel_expand(x)

        # Patch embedding
        x = self.patch_embed(x)

        # Prepend tokens
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
            num_special_tokens = 2
        else:
            x = torch.cat([cls_tokens, x], dim=1)
            num_special_tokens = 1

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through transformer blocks, capturing intermediates
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in layer_indices:
                if return_cls_only:
                    # Store CLS token only (position 0) for global semantic alignment
                    intermediates[idx] = x[:, 0:1, :].clone()  # (B, 1, D)
                else:
                    # Store intermediate: only patch tokens (exclude CLS/DIST tokens)
                    intermediates[idx] = x[:, num_special_tokens:, :].clone()

        # Final normalization
        x = self.norm(x)

        # Extract patch/CLS tokens (excluding special tokens)
        if return_cls_only:
            patch_tokens = x[:, 0:1, :]  # CLS token only (B, 1, D)
        else:
            patch_tokens = x[:, num_special_tokens:, :]

        # Classification heads
        cls_out = self.head(x[:, 0])

        if self.distillation and self.head_dist is not None:
            dist_out = self.head_dist(x[:, 1])
            if self.training:
                output = (cls_out, dist_out)
            else:
                if self.inference_mode == 'cls':
                    output = cls_out
                elif self.inference_mode == 'dist':
                    output = dist_out
                else:
                    output = (cls_out + dist_out) / 2
        else:
            output = cls_out

        return {
            'output': output,
            'intermediates': intermediates,
            'patch_tokens': patch_tokens
        }

    def get_attention_weights(self, x):
        """
        Extract attention weights from all transformer blocks.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Dict mapping layer_idx -> attention weights (B, num_heads, N, N)
        """
        attention_weights = {}

        # Channel expansion for grayscale
        if self.channel_expand is not None:
            x = self.channel_expand(x)

        # Patch embedding
        x = self.patch_embed(x)

        # Prepend tokens
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through transformer blocks with attention extraction
        for idx, block in enumerate(self.blocks):
            # Forward with attention extraction
            x = block(x, return_attention=True)

            # Extract stored attention weights
            if hasattr(block.attn, 'attn_weights') and block.attn.attn_weights is not None:
                attention_weights[idx] = block.attn.attn_weights.detach()

        return attention_weights


# DeiT variant configurations
DEIT_CONFIGS = {
    'tiny': {
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
    },
    'small': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
    },
    'base': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
    },
}


def get_deit_config(variant='tiny', **kwargs):
    """Get DeiT configuration for a specific variant."""
    if variant not in DEIT_CONFIGS:
        raise ValueError(f"Unknown DeiT variant: {variant}. Choose from {list(DEIT_CONFIGS.keys())}")

    config = DEIT_CONFIGS[variant].copy()
    config.update(kwargs)
    return config
