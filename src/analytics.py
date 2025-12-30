"""
Research-grade analytics for knowledge distillation experiments.

Provides tools for analyzing trained models:
- HessianAnalyzer: Loss landscape curvature analysis using PyHessian
- AttentionDistanceAnalyzer: Mean attention distance per layer for ViT models
- CKAAnalyzer: Layer-wise CKA similarity between models

These analytics help understand:
1. Why CNN-distilled ViTs underperform (sharp loss landscapes, short-range attention)
2. How DINOv2 distillation improves generalization (flat landscapes, long-range attention)
3. Semantic alignment quality between teacher and student (CKA heatmaps)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HessianAnalyzer:
    """
    Loss landscape curvature analysis using PyHessian.

    Computes:
    - Trace: Sum of eigenvalues (curvature/sharpness measure)
    - Top eigenvalues: Largest eigenvalues (dominant curvature directions)
    - Eigenvalue density: Distribution of curvature

    Expected results:
    - CNN-distilled: High trace (sharp landscape, poor generalization)
    - DINO-distilled: Low trace (flat landscape, good generalization)

    Note: Disable torch.compile before running Hessian analysis (double_backward issues).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        num_samples: int = 1024,
        batch_size: int = 64,
    ):
        """
        Args:
            model: Trained model to analyze
            criterion: Loss function (e.g., CrossEntropyLoss)
            device: CUDA device
            num_samples: Number of samples for Hessian estimation
            batch_size: Batch size for Hessian computation
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.num_samples = num_samples
        self.batch_size = batch_size

        # Unwrap compiled model if necessary
        if hasattr(model, '_orig_mod'):
            logger.info("Unwrapping torch.compile model for Hessian analysis")
            self.model = model._orig_mod

        self.model.eval()
        self.model.to(device)

    def _prepare_data(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect samples from dataloader for Hessian computation."""
        inputs_list = []
        targets_list = []
        count = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[-1]  # Handle dual-augment datasets
            else:
                inputs, targets = batch

            inputs_list.append(inputs)
            targets_list.append(targets)
            count += inputs.size(0)

            if count >= self.num_samples:
                break

        inputs = torch.cat(inputs_list, dim=0)[:self.num_samples]
        targets = torch.cat(targets_list, dim=0)[:self.num_samples]

        return inputs.to(self.device), targets.to(self.device)

    def compute_trace(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Compute Hessian trace using Hutchinson's method.

        Returns:
            Dict with 'trace', 'trace_std' (standard deviation across iterations)
        """
        try:
            from pyhessian import hessian
        except ImportError:
            logger.error("pyhessian not installed. Install with: pip install pyhessian")
            return {'trace': float('nan'), 'trace_std': float('nan'), 'error': 'pyhessian not installed'}

        inputs, targets = self._prepare_data(dataloader)

        logger.info(f"Computing Hessian trace on {len(inputs)} samples...")

        # Create Hessian computer
        hessian_comp = hessian(
            self.model,
            self.criterion,
            data=(inputs, targets),
            cuda=self.device.type == 'cuda'
        )

        # Compute trace using Hutchinson's estimator
        trace, trace_std = hessian_comp.trace(maxIter=50, tol=1e-3)

        result = {
            'trace': float(np.mean(trace)),
            'trace_std': float(trace_std) if trace_std is not None else 0.0,
            'num_samples': len(inputs)
        }

        logger.info(f"Hessian trace: {result['trace']:.4f} +/- {result['trace_std']:.4f}")
        return result

    def compute_top_eigenvalues(
        self,
        dataloader: DataLoader,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Compute top eigenvalues of the Hessian.

        Args:
            dataloader: Data for Hessian computation
            top_n: Number of top eigenvalues to compute

        Returns:
            Dict with 'eigenvalues', 'eigenvalue_ratio' (max/min ratio)
        """
        try:
            from pyhessian import hessian
        except ImportError:
            logger.error("pyhessian not installed")
            return {'eigenvalues': [], 'error': 'pyhessian not installed'}

        inputs, targets = self._prepare_data(dataloader)

        logger.info(f"Computing top {top_n} Hessian eigenvalues...")

        hessian_comp = hessian(
            self.model,
            self.criterion,
            data=(inputs, targets),
            cuda=self.device.type == 'cuda'
        )

        # Compute top eigenvalues using power iteration
        top_eigenvalues, _ = hessian_comp.eigenvalues(maxIter=100, tol=1e-4, top_n=top_n)

        eigenvalues = [float(e) for e in top_eigenvalues]
        ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-10) if len(eigenvalues) > 1 else 1.0

        result = {
            'eigenvalues': eigenvalues,
            'max_eigenvalue': eigenvalues[0] if eigenvalues else 0.0,
            'eigenvalue_ratio': ratio,
            'num_samples': len(inputs)
        }

        logger.info(f"Top eigenvalues: {eigenvalues}")
        logger.info(f"Eigenvalue ratio (max/min): {ratio:.2f}")
        return result

    def run_full_analysis(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Run complete Hessian analysis."""
        results = {}

        # Trace
        trace_results = self.compute_trace(dataloader)
        results.update(trace_results)

        # Top eigenvalues
        eigen_results = self.compute_top_eigenvalues(dataloader)
        results['eigenvalues'] = eigen_results['eigenvalues']
        results['max_eigenvalue'] = eigen_results['max_eigenvalue']
        results['eigenvalue_ratio'] = eigen_results['eigenvalue_ratio']

        return results


class AttentionDistanceAnalyzer:
    """
    Mean attention distance analysis for Vision Transformers.

    Computes the mean distance between query and attended key positions
    weighted by attention scores. Higher distance = longer-range dependencies.

    Expected results:
    - CNN-distilled: Short distances (local attention patterns like CNNs)
    - DINO-distilled: Long distances (global semantic attention)

    Reference: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        img_size: int = 32,
        patch_size: int = 4,
    ):
        """
        Args:
            model: ViT model with attention weight extraction capability
            device: CUDA device
            img_size: Input image size
            patch_size: Patch size for tokenization
        """
        self.model = model
        self.device = device
        self.img_size = img_size
        self.patch_size = patch_size

        # Compute patch grid
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # Precompute distance matrix for patches
        self.distance_matrix = self._compute_distance_matrix()

        self.model.eval()
        self.model.to(device)

    def _compute_distance_matrix(self) -> torch.Tensor:
        """Compute pairwise Euclidean distances between patch positions."""
        # Create grid of patch centers
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.grid_size),
            torch.arange(self.grid_size),
            indexing='ij'
        ), dim=-1).reshape(-1, 2).float()

        # Compute pairwise distances
        # dist[i, j] = ||coords[i] - coords[j]||
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 2)
        dist = torch.sqrt((diff ** 2).sum(dim=-1))  # (N, N)

        return dist.to(self.device)

    def _extract_attention_weights(
        self,
        inputs: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Extract attention weights from all layers.

        Args:
            inputs: Input images (B, C, H, W)

        Returns:
            Dict mapping layer_idx to attention weights (B, num_heads, N, N)
        """
        attention_weights = {}

        # Check if model has get_attention_weights method
        if hasattr(self.model, 'get_attention_weights'):
            attention_weights = self.model.get_attention_weights(inputs)
        else:
            # Fallback: Use hooks to capture attention
            hooks = []
            captured = {}

            def make_hook(layer_idx):
                def hook(module, input, output):
                    # Assuming attention module stores attention weights
                    if hasattr(module, 'attn_weights'):
                        captured[layer_idx] = module.attn_weights
                return hook

            # Register hooks on attention modules
            module = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(module, 'blocks'):
                for i, block in enumerate(module.blocks):
                    if hasattr(block, 'attn'):
                        hook = block.attn.register_forward_hook(make_hook(i))
                        hooks.append(hook)

            # Forward pass
            with torch.no_grad():
                _ = self.model(inputs)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            attention_weights = captured

        return attention_weights

    def compute_mean_attention_distance(
        self,
        dataloader: DataLoader,
        num_samples: int = 512
    ) -> Dict[str, Any]:
        """
        Compute mean attention distance per layer.

        Args:
            dataloader: Data for attention analysis
            num_samples: Number of samples to analyze

        Returns:
            Dict with per-layer distances and overall mean
        """
        # Collect samples
        inputs_list = []
        count = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs_list.append(inputs)
            count += inputs.size(0)
            if count >= num_samples:
                break

        inputs = torch.cat(inputs_list, dim=0)[:num_samples].to(self.device)

        logger.info(f"Computing mean attention distance on {len(inputs)} samples...")

        # Get attention weights
        attention_weights = self._extract_attention_weights(inputs)

        if not attention_weights:
            logger.warning("No attention weights captured. Model may not support attention extraction.")
            return {'error': 'No attention weights captured'}

        # Compute mean distance per layer
        layer_distances = {}

        for layer_idx, attn in attention_weights.items():
            # attn: (B, num_heads, N, N) - includes special tokens (CLS, possibly DIST)
            # Skip special tokens for distance computation
            num_special = attn.shape[-1] - self.num_patches
            if num_special > 0:
                attn = attn[:, :, num_special:, num_special:]  # Remove special tokens

            # Normalize attention weights
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)

            # Compute weighted mean distance
            # distance_matrix: (N, N), attn: (B, H, N, N)
            distances = self.distance_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            weighted_dist = (attn * distances).sum(dim=-1)  # (B, H, N)
            mean_dist = weighted_dist.mean().item()

            layer_distances[f'layer_{layer_idx}'] = mean_dist

        # Overall mean
        overall_mean = np.mean(list(layer_distances.values()))

        result = {
            'layer_distances': layer_distances,
            'mean_distance': overall_mean,
            'num_layers': len(layer_distances),
            'num_samples': len(inputs)
        }

        logger.info(f"Mean attention distance: {overall_mean:.4f}")
        for layer, dist in layer_distances.items():
            logger.info(f"  {layer}: {dist:.4f}")

        return result


class CKAAnalyzer:
    """
    Centered Kernel Alignment (CKA) similarity analyzer.

    Computes CKA similarity between:
    - Teacher and student intermediate representations
    - Different layers within the same model

    Produces heatmaps showing semantic alignment quality.

    Expected results:
    - Good distillation: Clear diagonal pattern (layer-to-layer alignment)
    - Poor distillation: Scattered or weak correlations

    Reference: Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019.
    """

    def __init__(
        self,
        model1: nn.Module,
        model2: Optional[nn.Module],
        device: torch.device,
        kernel_type: str = 'linear',
    ):
        """
        Args:
            model1: First model (e.g., student)
            model2: Second model (e.g., teacher), or None for self-CKA
            device: CUDA device
            kernel_type: 'linear' or 'rbf' kernel
        """
        self.model1 = model1
        self.model2 = model2
        self.device = device
        self.kernel_type = kernel_type

        self.model1.eval()
        self.model1.to(device)
        if model2 is not None:
            self.model2.eval()
            self.model2.to(device)

    def _compute_gram(self, X: torch.Tensor) -> torch.Tensor:
        """Compute gram matrix from features."""
        if self.kernel_type == 'linear':
            return X @ X.T
        else:  # rbf
            sq_dist = torch.cdist(X, X, p=2) ** 2
            sigma = torch.median(sq_dist).item() + 1e-10
            return torch.exp(-sq_dist / (2 * sigma))

    def _center_gram(self, K: torch.Tensor) -> torch.Tensor:
        """Center gram matrix."""
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H

    def _hsic(self, K1: torch.Tensor, K2: torch.Tensor) -> float:
        """Compute HSIC between two gram matrices."""
        n = K1.shape[0]
        return float((K1 * K2).sum() / ((n - 1) ** 2))

    def compute_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute CKA between two feature matrices.

        Args:
            X: Features from model 1 (N, D1)
            Y: Features from model 2 (N, D2)

        Returns:
            CKA similarity value in [0, 1]
        """
        # Compute and center gram matrices
        K_X = self._center_gram(self._compute_gram(X))
        K_Y = self._center_gram(self._compute_gram(Y))

        # Compute CKA
        hsic_XY = self._hsic(K_X, K_Y)
        hsic_XX = self._hsic(K_X, K_X)
        hsic_YY = self._hsic(K_Y, K_Y)

        cka = hsic_XY / (np.sqrt(hsic_XX * hsic_YY) + 1e-10)
        return float(cka)

    def _extract_layer_features(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        layer_indices: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Extract features from specified layers."""
        features = {}

        # Try forward_with_intermediates if available (for DeiT)
        if hasattr(model, 'forward_with_intermediates'):
            with torch.no_grad():
                results = model.forward_with_intermediates(inputs, layer_indices=layer_indices)
                for idx in layer_indices:
                    if idx in results['intermediates']:
                        feat = results['intermediates'][idx]
                        # Flatten to (B, D)
                        features[idx] = feat.mean(dim=1) if feat.dim() == 3 else feat
        else:
            # Use hooks for generic models
            hooks = []
            captured = {}

            def make_hook(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    # Global average pooling if spatial
                    if output.dim() == 4:  # (B, C, H, W)
                        captured[idx] = output.mean(dim=(2, 3))
                    elif output.dim() == 3:  # (B, N, D)
                        captured[idx] = output.mean(dim=1)
                    else:
                        captured[idx] = output
                return hook

            # Get model layers
            module = model.module if hasattr(model, 'module') else model
            if hasattr(module, 'blocks'):
                for idx in layer_indices:
                    if idx < len(module.blocks):
                        hook = module.blocks[idx].register_forward_hook(make_hook(idx))
                        hooks.append(hook)
            elif hasattr(module, 'features'):
                # For CNN-like models
                for idx in layer_indices:
                    if idx < len(module.features):
                        hook = module.features[idx].register_forward_hook(make_hook(idx))
                        hooks.append(hook)

            with torch.no_grad():
                _ = model(inputs)

            for hook in hooks:
                hook.remove()

            features = captured

        return features

    def compute_layer_cka_matrix(
        self,
        dataloader: DataLoader,
        layer_indices1: List[int],
        layer_indices2: Optional[List[int]] = None,
        num_samples: int = 512
    ) -> Dict[str, Any]:
        """
        Compute CKA matrix between layers of two models.

        Args:
            dataloader: Data for feature extraction
            layer_indices1: Layer indices for model1
            layer_indices2: Layer indices for model2 (defaults to layer_indices1)
            num_samples: Number of samples to use

        Returns:
            Dict with 'cka_matrix' (2D array) and layer indices
        """
        if layer_indices2 is None:
            layer_indices2 = layer_indices1 if self.model2 is None else layer_indices1

        # Collect samples
        inputs_list = []
        count = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs_list.append(inputs)
            count += inputs.size(0)
            if count >= num_samples:
                break

        inputs = torch.cat(inputs_list, dim=0)[:num_samples].to(self.device)

        logger.info(f"Computing CKA matrix on {len(inputs)} samples...")

        # Extract features
        features1 = self._extract_layer_features(self.model1, inputs, layer_indices1)
        if self.model2 is not None:
            features2 = self._extract_layer_features(self.model2, inputs, layer_indices2)
        else:
            features2 = features1
            layer_indices2 = layer_indices1

        # Compute CKA matrix
        n_layers1 = len(layer_indices1)
        n_layers2 = len(layer_indices2)
        cka_matrix = np.zeros((n_layers1, n_layers2))

        for i, idx1 in enumerate(tqdm(layer_indices1, desc="Computing CKA")):
            if idx1 not in features1:
                continue
            feat1 = features1[idx1].reshape(len(inputs), -1)

            for j, idx2 in enumerate(layer_indices2):
                if idx2 not in features2:
                    continue
                feat2 = features2[idx2].reshape(len(inputs), -1)

                cka_matrix[i, j] = self.compute_cka(feat1, feat2)

        result = {
            'cka_matrix': cka_matrix.tolist(),
            'layer_indices_1': layer_indices1,
            'layer_indices_2': layer_indices2,
            'num_samples': len(inputs)
        }

        logger.info(f"CKA matrix computed: {n_layers1}x{n_layers2}")
        return result


class AnalyticsRunner:
    """
    Unified runner for all analytics.

    Usage:
        runner = AnalyticsRunner(model, config, device)
        results = runner.run_all(dataloader, metrics=['hessian', 'attention', 'cka'])
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device,
        teacher_model: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: Trained model to analyze
            config: Configuration object
            device: CUDA device
            teacher_model: Optional teacher for CKA comparison
        """
        self.model = model
        self.config = config
        self.device = device
        self.teacher_model = teacher_model

        # Unwrap compiled model
        if hasattr(model, '_orig_mod'):
            logger.info("Unwrapping torch.compile model for analytics")
            self.model = model._orig_mod

    def run_all(
        self,
        dataloader: DataLoader,
        metrics: List[str] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run specified analytics.

        Args:
            dataloader: Data for analysis
            metrics: List of metrics to compute ('hessian', 'attention', 'cka')
            save_path: Optional path to save results

        Returns:
            Dict with all computed metrics
        """
        if metrics is None:
            metrics = ['hessian', 'attention', 'cka']

        results = {}

        if 'hessian' in metrics:
            logger.info("=" * 60)
            logger.info("Running Hessian Analysis")
            logger.info("=" * 60)
            try:
                criterion = nn.CrossEntropyLoss()
                analyzer = HessianAnalyzer(
                    self.model, criterion, self.device,
                    num_samples=getattr(self.config, 'hessian_samples', 1024)
                )
                results['hessian'] = analyzer.run_full_analysis(dataloader)
            except Exception as e:
                logger.error(f"Hessian analysis failed: {e}")
                results['hessian'] = {'error': str(e)}

        if 'attention' in metrics:
            logger.info("=" * 60)
            logger.info("Running Attention Distance Analysis")
            logger.info("=" * 60)
            try:
                img_size = getattr(self.config.vit, 'img_size', 32) if hasattr(self.config, 'vit') else 32
                patch_size = getattr(self.config.vit, 'patch_size', 4) if hasattr(self.config, 'vit') else 4
                analyzer = AttentionDistanceAnalyzer(
                    self.model, self.device,
                    img_size=img_size, patch_size=patch_size
                )
                results['attention'] = analyzer.compute_mean_attention_distance(dataloader)
            except Exception as e:
                logger.error(f"Attention analysis failed: {e}")
                results['attention'] = {'error': str(e)}

        if 'cka' in metrics:
            logger.info("=" * 60)
            logger.info("Running CKA Analysis")
            logger.info("=" * 60)
            try:
                analyzer = CKAAnalyzer(
                    self.model, self.teacher_model, self.device,
                    kernel_type=getattr(self.config, 'cka_kernel', 'linear')
                )
                # Default layer indices for DeiT-Tiny (12 layers)
                layer_indices = list(range(12))
                results['cka'] = analyzer.compute_layer_cka_matrix(
                    dataloader, layer_indices
                )
            except Exception as e:
                logger.error(f"CKA analysis failed: {e}")
                results['cka'] = {'error': str(e)}

        # Save results
        if save_path is not None:
            import json
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Analytics results saved to {save_path}")

        return results


# Export public API
__all__ = [
    'HessianAnalyzer',
    'AttentionDistanceAnalyzer',
    'CKAAnalyzer',
    'AnalyticsRunner',
]
