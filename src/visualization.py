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
