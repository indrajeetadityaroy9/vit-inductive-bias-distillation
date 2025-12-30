#!/usr/bin/env python3
"""
Generate comparison plots for PhD portfolio analytics.

Creates side-by-side visualizations comparing:
- EXP-1 (ResNet-18 distillation)
- EXP-2 (ConvNeXt V2 distillation)
- EXP-3 (DINOv2 CKA distillation)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_cka_matrix(results_path):
    """Load CKA matrix from analytics results."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return np.array(data['cka']['cka_matrix'])


def plot_cka_comparison(output_dir):
    """Create side-by-side CKA heatmaps for all experiments."""

    # Load CKA matrices
    exp1_cka = load_cka_matrix('outputs/analytics/exp1_resnet/analytics_results.json')
    exp2_cka = load_cka_matrix('outputs/analytics/exp2_convnext/analytics_results.json')
    exp3_cka = load_cka_matrix('outputs/analytics/exp3_dino/analytics_results.json')

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot settings
    vmin, vmax = 0.2, 1.0
    cmap = 'viridis'

    experiments = [
        ('EXP-1: ResNet-18 Teacher\n(Test Acc: 90.05%)', exp1_cka),
        ('EXP-2: ConvNeXt V2 Teacher\n(Test Acc: 90.11%)', exp2_cka),
        ('EXP-3: DINOv2 + CKA\n(Test Acc: 89.69%)', exp3_cka),
    ]

    for ax, (title, cka_matrix) in zip(axes, experiments):
        im = ax.imshow(cka_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel('Layer Index', fontsize=10)
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))

        # Add diagonal mean annotation
        diag_mean = np.mean(np.diag(cka_matrix))
        off_diag_mean = np.mean(cka_matrix[~np.eye(12, dtype=bool)])
        ax.text(0.02, 0.98, f'Diag: {diag_mean:.3f}\nOff-diag: {off_diag_mean:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label='CKA Similarity')

    plt.suptitle('Layer-wise CKA Self-Similarity: Impact of Distillation Method',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = Path(output_dir) / 'cka_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_comparison(output_dir):
    """Create bar chart comparing test accuracies."""

    experiments = ['Baseline\n(No Distill)', 'EXP-1\nResNet-18', 'EXP-2\nConvNeXt V2',
                   'EXP-3\nDINOv2+CKA']
    accuracies = [86.02, 90.05, 90.11, 89.69]
    colors = ['#808080', '#1f77b4', '#2ca02c', '#ff7f0e']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(experiments, accuracies, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(84, 92)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Comparison of Knowledge Distillation Methods on CIFAR-10\nDeiT-Tiny Student',
                 fontsize=14, fontweight='bold')

    # Add horizontal line for baseline
    ax.axhline(y=86.02, color='gray', linestyle='--', alpha=0.7, label='Baseline')
    ax.axhline(y=90.11, color='green', linestyle=':', alpha=0.7, label='Best (ConvNeXt)')

    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    output_path = Path(output_dir) / 'accuracy_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_distillation_efficiency(output_dir):
    """Create scatter plot showing distillation efficiency."""

    # Data: (Teacher Accuracy, Student Accuracy, Method)
    data = [
        (95.10, 90.05, 'ResNet-18\n(Classic CNN)', '#1f77b4'),
        (93.10, 90.11, 'ConvNeXt V2\n(Modern CNN)', '#2ca02c'),
        (None, 89.69, 'DINOv2\n(Label-Free)', '#ff7f0e'),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot supervised teachers
    for teacher_acc, student_acc, label, color in data:
        if teacher_acc is not None:
            efficiency = student_acc / teacher_acc * 100
            ax.scatter(teacher_acc, student_acc, s=200, c=color, edgecolors='black',
                      linewidth=2, zorder=5, label=f'{label} ({efficiency:.1f}% transfer)')

    # Add diagonal line (perfect transfer)
    ax.plot([88, 98], [88, 98], 'k--', alpha=0.3, label='Perfect Transfer')

    # Add DINOv2 point separately (no teacher accuracy)
    ax.axhline(y=89.69, color='#ff7f0e', linestyle=':', alpha=0.7)
    ax.annotate('DINOv2 (Label-Free): 89.69%', xy=(90, 89.69), fontsize=10,
                xytext=(90, 90.2), arrowprops=dict(arrowstyle='->', color='#ff7f0e'))

    ax.set_xlim(88, 98)
    ax.set_ylim(88, 92)
    ax.set_xlabel('Teacher Accuracy (%)', fontsize=12)
    ax.set_ylabel('Student Accuracy (%)', fontsize=12)
    ax.set_title('Knowledge Transfer Efficiency: Teacher vs Student Performance',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    output_path = Path(output_dir) / 'transfer_efficiency.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_attention_comparison(output_dir):
    """Create attention distance comparison plot."""

    # Load attention results
    with open('outputs/analytics/exp1_resnet/analytics_results.json') as f:
        exp1 = json.load(f)
    with open('outputs/analytics/exp2_convnext/analytics_results.json') as f:
        exp2 = json.load(f)
    with open('outputs/analytics/exp3_dino/analytics_results.json') as f:
        exp3 = json.load(f)

    layers = list(range(12))

    # Extract attention distances
    exp1_dist = [exp1['attention']['layer_distances'][f'layer_{i}'] for i in layers]
    exp2_dist = [exp2['attention']['layer_distances'][f'layer_{i}'] for i in layers]
    exp3_dist = [exp3['attention']['layer_distances'][f'layer_{i}'] for i in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, exp1_dist, 'o-', label=f'EXP-1: ResNet-18 (mean={np.mean(exp1_dist):.2f})',
            linewidth=2, markersize=8, color='#1f77b4')
    ax.plot(layers, exp2_dist, 's-', label=f'EXP-2: ConvNeXt V2 (mean={np.mean(exp2_dist):.2f})',
            linewidth=2, markersize=8, color='#2ca02c')
    ax.plot(layers, exp3_dist, '^-', label=f'EXP-3: DINOv2 (mean={np.mean(exp3_dist):.2f})',
            linewidth=2, markersize=8, color='#ff7f0e')

    ax.set_xlabel('Transformer Layer', fontsize=12)
    ax.set_ylabel('Mean Attention Distance (patches)', fontsize=12)
    ax.set_title('Attention Range Analysis: Structural Distillation Preserves Long-Range Dependencies',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(layers)
    ax.set_ylim(1.5, 4.5)

    # Add annotation
    ax.annotate('DINOv2 maintains\nlonger-range attention',
                xy=(6, 3.77), fontsize=10, ha='center',
                xytext=(6, 4.2),
                arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5))

    output_path = Path(output_dir) / 'attention_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path('outputs/analytics')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating PhD portfolio visualizations...")
    print("=" * 60)

    plot_cka_comparison(output_dir)
    plot_accuracy_comparison(output_dir)
    plot_distillation_efficiency(output_dir)
    plot_attention_comparison(output_dir)

    print("=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
