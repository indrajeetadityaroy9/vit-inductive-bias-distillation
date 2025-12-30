"""
Teacher model architectures for knowledge distillation experiments.

This module provides alternative teacher architectures for the comparative study
on inductive bias mismatch in heterogeneous knowledge distillation:

- ResNet18CIFAR: Classic CNN with strict locality bias (control)
- ConvNeXtV2Tiny: Modern CNN with Transformer-like macro architecture (bridge)

Both models are adapted for CIFAR-10 (32x32 images).
"""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Check for timm availability (required for ConvNeXt V2)
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    logger.warning(
        "timm not installed. ConvNeXt V2 models will not be available. "
        "Install with: pip install timm>=0.9.0"
    )


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18.

    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+shortcut) -> ReLU
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection for dimension matching
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 modified for CIFAR-10 (32x32 images).

    Key modifications from ImageNet ResNet-18:
    1. First conv: 3x3 kernel with stride 1 (not 7x7 with stride 2)
    2. No max pooling after first conv (preserves spatial resolution)
    3. Better suited for small 32x32 images

    This represents the "Classic CNN" inductive bias for the distillation study.
    Target accuracy: >94% on CIFAR-10.

    Architecture:
        Input (3, 32, 32)
        -> Conv3x3 (64) -> BN -> ReLU      [32x32]
        -> Layer1: 2x BasicBlock(64)       [32x32]
        -> Layer2: 2x BasicBlock(128)      [16x16]
        -> Layer3: 2x BasicBlock(256)      [8x8]
        -> Layer4: 2x BasicBlock(512)      [4x4]
        -> AdaptiveAvgPool -> FC(10)

    Parameters: ~11.2M
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.in_channels = config.get("in_channels", 3)
        self.num_classes = config.get("num_classes", 10)
        self.in_planes = 64

        # Modified first layer for CIFAR (3x3, stride 1, no maxpool)
        self.conv1 = nn.Conv2d(
            self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        # NO MaxPool - removed for 32x32 images

        # ResNet-18 layer configuration: [2, 2, 2, 2]
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, self.num_classes)

        # Initialize weights
        self._init_weights()

        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Initialized ResNet18CIFAR: {self.num_classes} classes, "
            f"{total_params:,} parameters"
        )

    def _make_layer(
        self, block: type, planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Build a layer with multiple residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-18."""
        out = F.relu(self.bn1(self.conv1(x)))
        # No maxpool for CIFAR
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def get_feature_maps(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Extract feature maps at a specific layer for visualization."""
        layers = [
            lambda x: F.relu(self.bn1(self.conv1(x))),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]
        for i, layer in enumerate(layers):
            x = layer(x)
            if i == layer_idx:
                return x
        return x


class LayerNorm2d(nn.Module):
    """
    LayerNorm for channels-first tensors (B, C, H, W).
    Used in ConvNeXt stem.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtV2Tiny(nn.Module):
    """
    ConvNeXt V2-Tiny adapted for CIFAR-10 (32x32 images).

    Uses ImageNet-pretrained backbone with modified stem for small images.
    This represents the "Modern CNN Bridge" - an architecture that combines:
    - CNN local operations (depthwise convolutions)
    - Transformer-like macro design (stage ratios, LayerNorm, GELU)

    Key modifications:
    - Original stem: 4x4 patchify with stride 4 (for 224x224 -> 56x56)
    - Modified stem: 2x2 patchify with stride 2 (for 32x32 -> 16x16)
    - Stem weights are reinitialized (see Pre-Flight Check E, F)

    Target accuracy: >95% on CIFAR-10 with fine-tuning.

    Parameters: ~28.6M
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError(
                "timm is required for ConvNeXt V2 models. "
                "Install with: pip install timm>=0.9.0"
            )

        self.in_channels = config.get("in_channels", 3)
        self.num_classes = config.get("num_classes", 10)
        self.pretrained = config.get("pretrained", True)  # Use ImageNet weights
        self.drop_path_rate = config.get("drop_path_rate", 0.1)

        # Create ConvNeXt V2 Tiny base model with ImageNet weights
        self.model = timm.create_model(
            "convnextv2_tiny",
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            in_chans=self.in_channels,
            drop_path_rate=self.drop_path_rate,
        )

        # Get original stem output channels (96 for tiny)
        stem_out_channels = 96

        # Modify stem for 32x32 images
        # Original: 4x4 conv, stride 4 -> 56x56 from 224x224
        # Modified: 2x2 conv, stride 2 -> 16x16 from 32x32
        self.model.stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels, stem_out_channels, kernel_size=2, stride=2, padding=0
            ),
            LayerNorm2d(stem_out_channels),
        )

        # CRITICAL: Reinitialize the modified stem weights (Pre-Flight Check E)
        # Don't use mismatched pretrained weights
        nn.init.trunc_normal_(self.model.stem[0].weight, std=0.02)
        nn.init.zeros_(self.model.stem[0].bias)

        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        pretrained_str = "ImageNet-pretrained" if self.pretrained else "random init"
        logger.info(
            f"Initialized ConvNeXtV2Tiny: {self.num_classes} classes, "
            f"{total_params:,} parameters ({pretrained_str}, stem reinitialized)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvNeXt V2."""
        return self.model(x)

    def get_intermediate_features(
        self, x: torch.Tensor, stage_indices: list = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract intermediate features from specified stages.

        Args:
            x: Input tensor (B, C, H, W)
            stage_indices: List of stage indices to extract (0-3)

        Returns:
            Dict mapping stage index to feature tensor
        """
        if stage_indices is None:
            stage_indices = [0, 1, 2, 3]

        features = {}

        # Forward through stem
        x = self.model.stem(x)

        # Forward through stages
        for i, stage in enumerate(self.model.stages):
            x = stage(x)
            if i in stage_indices:
                features[i] = x.clone()

        return features


# Model registration will be done in models.py
__all__ = ["ResNet18CIFAR", "ConvNeXtV2Tiny", "HAS_TIMM"]
