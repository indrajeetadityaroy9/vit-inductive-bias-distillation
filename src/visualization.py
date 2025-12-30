import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
import cv2

logger = logging.getLogger(__name__)

class FeatureMapVisualizer:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = {}
        self.gradients = {}

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output, name):
            self.activations[name] = output.detach()

        def backward_hook(module, grad_input, grad_output, name):
            self.gradients[name] = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(
                    lambda m, i, o, n=name: forward_hook(m, i, o, n)
                )
                module.register_backward_hook(
                    lambda m, gi, go, n=name: backward_hook(m, gi, go, n)
                )

    @torch.no_grad()
    def visualize_feature_maps(self, image, layer_name,
                              n_features=32, save_path=None):
        self.model.eval()

        image = image.to(self.device)
        _ = self.model(image)

        if layer_name not in self.activations:
            logger.error(f"Layer {layer_name} not found in activations")
            return

        activation = self.activations[layer_name].squeeze(0)
        n_features = min(n_features, activation.shape[0])

        n_cols = 8
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i in range(n_features):
            feature_map = activation[i].cpu().numpy()

            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i}', fontsize=8)

        for i in range(n_features, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Feature Maps from {layer_name}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature maps saved to {save_path}")

        plt.show()

    def visualize_filters(self, layer_name, save_path=None):

        layer = None
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                layer = module
                break

        if layer is None:
            logger.error(f"Layer {layer_name} not found or not a Conv2d layer")
            return

        filters = layer.weight.data.cpu().numpy()
        n_filters = min(32, filters.shape[0])
        n_channels = filters.shape[1]

        if n_channels > 1:
            filters = filters.mean(axis=1)
        else:
            filters = filters.squeeze(1)

        n_cols = 8
        n_rows = (n_filters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i in range(n_filters):
            filter_img = filters[i]

            vmin, vmax = filter_img.min(), filter_img.max()
            filter_img = (filter_img - vmin) / (vmax - vmin + 1e-8)

            axes[i].imshow(filter_img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i}', fontsize=8)

        for i in range(n_filters, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Filters from {layer_name}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Filters saved to {save_path}")

        plt.show()

class GradCAM:

    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        layer_found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                layer_found = True
                break

        if not layer_found:
            logger.warning(f"GradCAM target layer '{self.target_layer}' not found in model. "
                          f"Available layers: {[n for n, _ in self.model.named_modules() if n]}")

    def generate_heatmap(self, image, class_idx=None):
        self.model.eval()

        image = image.to(self.device)
        image.requires_grad = True

        output = self.model(image)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        self.model.zero_grad()

        output[0, class_idx].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)

        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    def visualize(self, image, original_image,
                 class_idx=None, alpha=0.5,
                 save_path=None):

        heatmap = self.generate_heatmap(image, class_idx)

        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        if original_image.max() <= 1:
            original_image = (original_image * 255).astype(np.uint8)

        superimposed = heatmap * alpha + original_image * (1 - alpha)
        superimposed = superimposed.astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(heatmap)
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')

        axes[2].imshow(superimposed)
        axes[2].set_title('GradCAM Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"GradCAM visualization saved to {save_path}")

        plt.show()

class TrainingVisualizer:

    @staticmethod
    def plot_training_history(metrics_history, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        if 'train_loss' in metrics_history:
            axes[0, 0].plot(metrics_history['train_loss'], label='Train Loss')
            if 'val_loss' in metrics_history:
                axes[0, 0].plot(metrics_history['val_loss'], label='Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        if 'train_acc' in metrics_history:
            axes[0, 1].plot(metrics_history['train_acc'], label='Train Acc')
            if 'val_acc' in metrics_history:
                axes[0, 1].plot(metrics_history['val_acc'], label='Val Acc')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        if 'learning_rate' in metrics_history:
            axes[1, 0].plot(metrics_history['learning_rate'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')

        if 'train_acc' in metrics_history and 'val_acc' in metrics_history:
            train_acc = np.array(metrics_history['train_acc'])
            val_acc = np.array(metrics_history['val_acc'])
            gap = train_acc - val_acc
            axes[1, 1].plot(gap)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gap (%)')
            axes[1, 1].set_title('Train-Val Accuracy Gap')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

        plt.suptitle('Training History')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_learning_rate_finder(lrs, losses,
                                 save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(lrs, losses)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder')
        ax.grid(True, alpha=0.3)

        min_loss_idx = np.argmin(losses)
        suggested_lr = lrs[min_loss_idx] / 10

        ax.axvline(x=suggested_lr, color='r', linestyle='--',
                  label=f'Suggested LR: {suggested_lr:.2e}')
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Learning rate finder plot saved to {save_path}")

        plt.show()

        return suggested_lr


class AnalyticsVisualizer:
    """
    Visualization tools for research-grade analytics.

    Provides publication-quality plots for:
    - CKA similarity heatmaps
    - Attention distance profiles
    - Hessian trace comparisons
    - Experiment result summaries
    """

    @staticmethod
    def plot_cka_heatmap(
        cka_matrix,
        layer_indices_x=None,
        layer_indices_y=None,
        title="CKA Similarity",
        xlabel="Teacher Layer",
        ylabel="Student Layer",
        save_path=None,
        figsize=(10, 8),
        cmap='viridis'
    ):
        """
        Plot CKA similarity heatmap.

        Args:
            cka_matrix: 2D numpy array of CKA values
            layer_indices_x: Labels for x-axis (teacher layers)
            layer_indices_y: Labels for y-axis (student layers)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save figure
            figsize: Figure size tuple
            cmap: Colormap name
        """
        cka_matrix = np.array(cka_matrix)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(cka_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('CKA Similarity', fontsize=12)

        # Set axis labels
        if layer_indices_x is not None:
            ax.set_xticks(range(len(layer_indices_x)))
            ax.set_xticklabels([f'L{i}' for i in layer_indices_x], fontsize=9)
        if layer_indices_y is not None:
            ax.set_yticks(range(len(layer_indices_y)))
            ax.set_yticklabels([f'L{i}' for i in layer_indices_y], fontsize=9)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)

        # Add value annotations for small matrices
        if cka_matrix.shape[0] <= 12 and cka_matrix.shape[1] <= 12:
            for i in range(cka_matrix.shape[0]):
                for j in range(cka_matrix.shape[1]):
                    value = cka_matrix[i, j]
                    color = 'white' if value < 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=color, fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"CKA heatmap saved to {save_path}")

        plt.show()
        return fig

    @staticmethod
    def plot_attention_distances(
        layer_distances,
        title="Mean Attention Distance per Layer",
        save_path=None,
        figsize=(12, 6),
        color='steelblue',
        comparison_data=None
    ):
        """
        Plot mean attention distance per layer.

        Args:
            layer_distances: Dict mapping layer names to distances
            title: Plot title
            save_path: Optional path to save figure
            figsize: Figure size tuple
            color: Bar color
            comparison_data: Optional dict for comparison (e.g., CNN-distilled vs DINO-distilled)
        """
        fig, ax = plt.subplots(figsize=figsize)

        layers = list(layer_distances.keys())
        distances = list(layer_distances.values())

        x = np.arange(len(layers))
        width = 0.35 if comparison_data else 0.7

        bars1 = ax.bar(x - width/2 if comparison_data else x, distances, width,
                      label='Model', color=color, alpha=0.8)

        if comparison_data:
            comp_distances = [comparison_data.get(l, 0) for l in layers]
            bars2 = ax.bar(x + width/2, comp_distances, width,
                          label='Comparison', color='coral', alpha=0.8)

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Mean Attention Distance (patches)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)

        if comparison_data:
            ax.legend()

        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention distance plot saved to {save_path}")

        plt.show()
        return fig

    @staticmethod
    def plot_hessian_comparison(
        results,
        metric='trace',
        title="Hessian Trace Comparison",
        save_path=None,
        figsize=(10, 6)
    ):
        """
        Plot Hessian metric comparison across experiments.

        Args:
            results: Dict mapping experiment names to Hessian results
            metric: 'trace' or 'max_eigenvalue'
            title: Plot title
            save_path: Optional path to save figure
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)

        experiments = list(results.keys())
        values = [results[exp].get(metric, 0) for exp in experiments]
        errors = [results[exp].get(f'{metric}_std', 0) for exp in experiments]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(experiments)))

        bars = ax.bar(experiments, values, yerr=errors, capsize=5,
                     color=colors, alpha=0.8, edgecolor='black')

        ax.set_ylabel(f'Hessian {metric.replace("_", " ").title()}', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x labels if many experiments
        if len(experiments) > 4:
            plt.xticks(rotation=45, ha='right')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hessian comparison plot saved to {save_path}")

        plt.show()
        return fig

    @staticmethod
    def plot_experiment_summary(
        results,
        metrics=['accuracy', 'hessian_trace', 'mean_attention_distance'],
        title="Experiment Comparison Summary",
        save_path=None,
        figsize=(14, 5)
    ):
        """
        Create multi-panel summary of experiment results.

        Args:
            results: Dict mapping experiment names to result dicts
            metrics: List of metrics to plot
            title: Overall title
            save_path: Optional path to save figure
            figsize: Figure size tuple
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]

        experiments = list(results.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

        metric_labels = {
            'accuracy': 'Test Accuracy (%)',
            'hessian_trace': 'Hessian Trace',
            'mean_attention_distance': 'Mean Attn Distance',
            'cka_diagonal_mean': 'CKA Diagonal Mean'
        }

        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = []

            for exp in experiments:
                if metric in results[exp]:
                    values.append(results[exp][metric])
                elif metric == 'hessian_trace' and 'hessian' in results[exp]:
                    values.append(results[exp]['hessian'].get('trace', 0))
                elif metric == 'mean_attention_distance' and 'attention' in results[exp]:
                    values.append(results[exp]['attention'].get('mean_distance', 0))
                else:
                    values.append(0)

            bars = ax.bar(experiments, values, color=colors, alpha=0.8, edgecolor='black')
            ax.set_ylabel(metric_labels.get(metric, metric.replace('_', ' ').title()))
            ax.set_title(metric_labels.get(metric, metric.replace('_', ' ').title()))
            ax.grid(True, alpha=0.3, axis='y')

            if len(experiments) > 3:
                ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)

            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Experiment summary saved to {save_path}")

        plt.show()
        return fig

    @staticmethod
    def plot_distillation_loss_curves(
        metrics_history,
        title="Distillation Training Curves",
        save_path=None,
        figsize=(14, 10)
    ):
        """
        Plot detailed distillation training curves.

        Args:
            metrics_history: Dict with training metrics over epochs
            title: Plot title
            save_path: Optional path to save figure
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Total loss
        if 'train_loss' in metrics_history:
            axes[0, 0].plot(metrics_history['train_loss'], label='Train', color='blue')
            if 'val_loss' in metrics_history:
                axes[0, 0].plot(metrics_history['val_loss'], label='Val', color='orange')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        if 'train_acc' in metrics_history:
            axes[0, 1].plot(metrics_history['train_acc'], label='Train', color='blue')
            if 'val_acc' in metrics_history:
                axes[0, 1].plot(metrics_history['val_acc'], label='Val', color='orange')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # CE Loss
        if 'train_ce_loss' in metrics_history:
            axes[0, 2].plot(metrics_history['train_ce_loss'], label='CE Loss', color='green')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].set_title('Classification Loss')
            axes[0, 2].grid(True, alpha=0.3)

        # Token Loss
        if 'train_tok_loss' in metrics_history:
            axes[1, 0].plot(metrics_history['train_tok_loss'], label='Token Loss', color='purple')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Token Representation Loss')
            axes[1, 0].grid(True, alpha=0.3)

        # CKA/Correlation Loss
        if 'train_cka_loss' in metrics_history:
            axes[1, 1].plot(metrics_history['train_cka_loss'], label='CKA Loss', color='red')
        if 'train_rel_loss' in metrics_history:
            axes[1, 1].plot(metrics_history['train_rel_loss'], label='Correlation Loss',
                          color='coral', linestyle='--')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Structural Losses')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Lambda schedule
        if 'effective_lambda_rel' in metrics_history or 'effective_lambda_cka' in metrics_history:
            if 'effective_lambda_rel' in metrics_history:
                axes[1, 2].plot(metrics_history['effective_lambda_rel'],
                               label='Lambda Rel', color='coral')
            if 'effective_lambda_cka' in metrics_history:
                axes[1, 2].plot(metrics_history['effective_lambda_cka'],
                               label='Lambda CKA', color='red')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Lambda')
            axes[1, 2].set_title('Loss Weight Schedule')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distillation curves saved to {save_path}")

        plt.show()
        return fig
