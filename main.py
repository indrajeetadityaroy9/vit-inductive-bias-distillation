"""
Unified CLI entry point for dataset-adaptive image classification.

Supports:
- Standard training (AdaptiveCNN, ViT)
- Knowledge distillation (CNN → DeiT)
- Self-supervised distillation (DINOv2 → DeiT)
- Evaluation and analytics
"""
import argparse
import logging
import os
import random
import socket
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.config import (
    ConfigManager,
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    LoggingConfig,
    ViTConfig,
    DistillationConfig,
    SelfSupervisedDistillationConfig,
    setup_logging
)
from src.models import ModelFactory
from src.datasets import DatasetManager, preprocess_image
from src.evaluation import ModelEvaluator, TestTimeAugmentation
from src.training import DDPTrainer
from src.distillation import DistillationTrainer, SelfSupervisedDistillationTrainer, load_dino_teacher
from src.visualization import FeatureMapVisualizer, GradCAM, TrainingVisualizer
from src.analytics import AnalyticsRunner, AnalyticsVisualizer

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def find_free_port() -> int:
    """Find a free port for DDP communication."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_ddp_environment(rank: int, world_size: int, port: Optional[int] = None) -> None:
    """Set up DDP environment variables and initialize process group."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(port or 29500))
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )


def setup_directories(config: Config, world_size: int, is_main_process: bool) -> Tuple[Path, Path]:
    """Create output and checkpoint directories."""
    if world_size > 1:
        config.experiment_name = f"{config.experiment_name}_ddp_{world_size}gpu"

    output_dir = Path(config.output_dir) / config.experiment_name
    checkpoints_dir = output_dir / "checkpoints"

    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    dist.barrier()
    return output_dir, checkpoints_dir


def setup_logging_for_rank(config: Config, rank: int, world_size: int, mode_name: str) -> logging.Logger:
    """Set up logging based on process rank."""
    is_main_process = (rank == 0)

    if is_main_process:
        setup_logging(config.logging)
        log = logging.getLogger(__name__)
        log.info("=" * 60)
        log.info(f"{mode_name} (DDP)")
        log.info("=" * 60)
        log.info(f"Experiment: {config.experiment_name}")
        log.info(f"Dataset: {config.data.dataset}")
        log.info(f"World Size: {world_size}")
        log.info(f"GPUs: {world_size} x {torch.cuda.get_device_name(0)}")
        log.info("=" * 60)
    else:
        logging.basicConfig(level=logging.ERROR)
        log = logging.getLogger(__name__)

    return log


def build_model_config(config: Config) -> dict:
    """Build merged model configuration (model + vit configs)."""
    model_config = config.model.__dict__.copy()
    if hasattr(config, 'vit') and config.vit is not None:
        vit_dict = config.vit.__dict__ if hasattr(config.vit, '__dict__') else config.vit
        model_config.update(vit_dict)
    return model_config


def create_distributed_dataloaders(
    config: Config,
    world_size: int,
    rank: int,
    use_dual_augment: bool = False
) -> Tuple[DataLoader, DistributedSampler, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with distributed samplers."""
    # Get datasets
    if use_dual_augment:
        train_dataset = DatasetManager.get_dual_augment_dataset(config)
    else:
        train_dataset = DatasetManager.get_dataset(config, is_train=True)

    val_dataset = DatasetManager.get_dataset(config, is_train=False)
    test_dataset = DatasetManager.get_dataset(config, is_train=False)

    # Create samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # DataLoader kwargs
    loader_kwargs = {
        'batch_size': config.data.batch_size,
        'num_workers': config.data.num_workers,
        'pin_memory': config.data.pin_memory,
        'persistent_workers': config.data.persistent_workers and config.data.num_workers > 0,
        'prefetch_factor': config.data.prefetch_factor if config.data.num_workers > 0 else 2
    }

    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    return train_loader, train_sampler, val_loader, test_loader


def evaluate_and_save_results(
    trainer,
    test_loader: DataLoader,
    dataset_info: dict,
    config: Config,
    output_dir: Path,
    metrics_history: dict,
    world_size: int,
    mode_name: str,
    extra_metrics_info: Optional[Dict[str, Any]] = None
) -> dict:
    """Evaluate model and save results (only on main process)."""
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("EVALUATING ON TEST SET")
    log.info("=" * 60)

    # Get unwrapped model for evaluation
    model = trainer.ddp_model.module if hasattr(trainer.ddp_model, 'module') else trainer.ddp_model
    device = trainer.device

    evaluator = ModelEvaluator(model, device, dataset_info.get('classes'))
    test_metrics = evaluator.evaluate(test_loader)
    evaluator.print_summary()

    # Save config
    ConfigManager.save_config(config, output_dir / 'config.yaml')

    # Save training history
    if metrics_history:
        TrainingVisualizer.plot_training_history(
            metrics_history,
            save_path=output_dir / 'training_history.png'
        )

    # Save confusion matrix
    evaluator.plot_confusion_matrix(
        save_path=output_dir / 'confusion_matrix.png',
        normalize=True
    )

    # Save metrics file
    metrics_file = output_dir / 'test_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"{mode_name} RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Configuration: {world_size} GPUs with DDP\n")
        f.write(f"Effective Batch Size: {config.data.batch_size * world_size}\n")

        # Write extra info if provided
        if extra_metrics_info:
            f.write("\n")
            for key, value in extra_metrics_info.items():
                f.write(f"{key}: {value}\n")

        f.write(f"\nAccuracy:          {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision (macro): {test_metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (macro):    {test_metrics['recall_macro']:.4f}\n")
        f.write(f"F1 Score (macro):  {test_metrics['f1_macro']:.4f}\n")
        f.write(f"Loss:              {test_metrics.get('loss', 'N/A')}\n")

        if 'auc_macro' in test_metrics:
            f.write(f"\nAUC Macro: {test_metrics['auc_macro']:.4f}\n")
            f.write(f"AUC Weighted: {test_metrics['auc_weighted']:.4f}\n")

    log.info("=" * 60)
    log.info(f"{mode_name} COMPLETED")
    log.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    log.info(f"Outputs saved to: {output_dir}")
    log.info("=" * 60)

    return {
        'experiment_name': config.experiment_name,
        'test_accuracy': test_metrics['accuracy'],
        'output_dir': str(output_dir),
        'num_gpus': world_size
    }


def add_safe_globals_for_checkpoints():
    """Add safe globals for torch.load with weights_only=False."""
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig,
            ViTConfig, DistillationConfig, SelfSupervisedDistillationConfig
        ])


# =============================================================================
# Unified DDP Worker
# =============================================================================

def unified_ddp_worker(
    rank: int,
    world_size: int,
    config_path: str,
    mode: str,
    port: int
) -> Optional[dict]:
    """
    Unified DDP training worker that handles all training modes.

    Args:
        rank: Process rank
        world_size: Total number of processes
        config_path: Path to configuration file
        mode: Training mode ('standard', 'distill', 'ss_distill')
        port: Port for DDP communication
    """
    setup_ddp_environment(rank, world_size, port)

    result = None
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        is_main_process = (rank == 0)

        # Load config
        config = ConfigManager.load_config(config_path)

        # Mode-specific names
        mode_names = {
            'standard': 'PyTorch DDP Training',
            'distill': 'DeiT Distillation Training',
            'ss_distill': 'Self-Supervised Token Distillation (CST-Style) Training'
        }
        mode_name = mode_names[mode]

        # Setup directories and logging
        output_dir, checkpoints_dir = setup_directories(config, world_size, is_main_process)
        log = setup_logging_for_rank(config, rank, world_size, mode_name)

        set_seed(config.seed + rank)

        # Get dataset info
        dataset_info = DatasetManager.get_dataset_info(config)
        config.model.in_channels = dataset_info['in_channels']
        config.model.num_classes = dataset_info['num_classes']
        config.model.dataset = config.data.dataset

        # Build model config
        model_config = build_model_config(config)

        # Mode-specific setup
        if mode == 'standard':
            result = _run_standard_training(
                rank, world_size, device, config, model_config, dataset_info,
                output_dir, checkpoints_dir, is_main_process, log, mode_name
            )
        elif mode == 'distill':
            result = _run_distillation_training(
                rank, world_size, device, config, model_config, dataset_info,
                output_dir, checkpoints_dir, is_main_process, log, mode_name
            )
        elif mode == 'ss_distill':
            result = _run_ss_distillation_training(
                rank, world_size, device, config, model_config, dataset_info,
                output_dir, checkpoints_dir, is_main_process, log, mode_name
            )

    finally:
        dist.destroy_process_group()

    return result


def _run_standard_training(
    rank, world_size, device, config, model_config, dataset_info,
    output_dir, checkpoints_dir, is_main_process, log, mode_name
):
    """Run standard DDP training."""
    # Create model
    model = ModelFactory.create_model(config.model.model_type, model_config)
    model = model.to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if is_main_process:
        total_params = sum(p.numel() for p in model.module.parameters())
        trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        log.info(f"Total parameters: {total_params:,}")
        log.info(f"Trainable parameters: {trainable_params:,}")
        log.info(f"Effective batch size: {config.data.batch_size * world_size}")

    # Create dataloaders
    train_loader, train_sampler, val_loader, test_loader = create_distributed_dataloaders(
        config, world_size, rank
    )

    if is_main_process:
        log.info(f"Train samples: {len(train_loader.dataset)} ({len(train_loader.dataset)//world_size} per GPU)")

    # Create trainer
    trainer = DDPTrainer(model, config, device, rank, world_size)

    # Check for existing checkpoint
    checkpoint_path = checkpoints_dir / f"best_model_{config.data.dataset}_ddp.pth"
    if is_main_process and checkpoint_path.exists():
        try:
            trainer.load_checkpoint(checkpoint_path)
            log.info(f"Resumed from epoch {trainer.current_epoch}")
        except Exception as e:
            log.warning(f"Failed to load checkpoint: {e}")

    dist.barrier()

    # Train
    if is_main_process:
        log.info("=" * 60)
        log.info("STARTING DDP TRAINING")
        log.info("=" * 60)

    metrics_history = trainer.train_ddp(train_loader, train_sampler, val_loader)

    # Evaluate (main process only)
    if is_main_process:
        return evaluate_and_save_results(
            trainer, test_loader, dataset_info, config, output_dir,
            metrics_history, world_size, mode_name
        )
    return None


def _run_distillation_training(
    rank, world_size, device, config, model_config, dataset_info,
    output_dir, checkpoints_dir, is_main_process, log, mode_name
):
    """Run CNN→DeiT distillation training."""
    # Create student model (DeiT)
    student_model = ModelFactory.create_model('deit', model_config)
    student_model = student_model.to(device)

    if is_main_process:
        total_params = sum(p.numel() for p in student_model.parameters())
        log.info(f"Student model (DeiT) created with {total_params:,} parameters")

    # Load teacher model
    teacher_checkpoint_path = Path(config.distillation.teacher_checkpoint)
    if not teacher_checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")

    teacher_config = config.model.__dict__.copy()
    teacher_model = ModelFactory.create_model(config.distillation.teacher_model_type, teacher_config)
    teacher_model = teacher_model.to(device)

    add_safe_globals_for_checkpoints()
    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    if 'model_state_dict' not in teacher_checkpoint:
        raise KeyError(f"Teacher checkpoint missing 'model_state_dict'. Keys: {list(teacher_checkpoint.keys())}")
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.eval()

    if is_main_process:
        teacher_acc = teacher_checkpoint.get('best_val_acc', 'N/A')
        log.info(f"Teacher model loaded from: {teacher_checkpoint_path}")
        log.info(f"Teacher validation accuracy: {teacher_acc}")

    # Wrap student with DDP
    student_model = DDP(student_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Create dataloaders
    train_loader, train_sampler, val_loader, test_loader = create_distributed_dataloaders(
        config, world_size, rank
    )

    if is_main_process:
        log.info(f"Train samples: {len(train_loader.dataset)} ({len(train_loader.dataset)//world_size} per GPU)")
        log.info(f"Effective batch size: {config.data.batch_size * world_size}")

    # Create distillation trainer
    trainer = DistillationTrainer(student_model, teacher_model, config, device, rank, world_size)

    dist.barrier()

    if is_main_process:
        log.info("=" * 60)
        log.info("STARTING DISTILLATION TRAINING")
        log.info("=" * 60)

    # Train
    metrics_history = trainer.train_ddp(train_loader, train_sampler, val_loader)

    # Evaluate (main process only)
    if is_main_process:
        extra_info = {
            'Teacher': config.distillation.teacher_model_type,
            'Distillation Type': config.distillation.distillation_type,
            'Alpha': config.distillation.alpha
        }
        return evaluate_and_save_results(
            trainer, test_loader, dataset_info, config, output_dir,
            metrics_history, world_size, mode_name, extra_info
        )
    return None


def _run_ss_distillation_training(
    rank, world_size, device, config, model_config, dataset_info,
    output_dir, checkpoints_dir, is_main_process, log, mode_name
):
    """Run DINOv2→DeiT self-supervised distillation training."""
    # Validate config
    if config.ss_distillation is None:
        raise ValueError("ss_distillation config is required for train-ss-distill command")

    # Create student model (DeiT)
    student_model = ModelFactory.create_model('deit', model_config)
    student_model = student_model.to(device)

    if is_main_process:
        total_params = sum(p.numel() for p in student_model.parameters())
        log.info(f"Student model (DeiT) created with {total_params:,} parameters")
        log.info(f"Loading {config.ss_distillation.teacher_type} teacher: {config.ss_distillation.teacher_model_name}")

    # Load DINO/DINOv2 teacher
    teacher_model, teacher_embed_dim = load_dino_teacher(
        config.ss_distillation.teacher_type,
        config.ss_distillation.teacher_model_name,
        device
    )

    if is_main_process:
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        log.info(f"Teacher model loaded with {teacher_params:,} parameters (frozen)")
        log.info(f"Teacher embedding dim: {teacher_embed_dim}")

    # Wrap student with DDP
    student_model = DDP(student_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Create dataloaders (with dual-augment if enabled)
    use_dual_augment = getattr(config.ss_distillation, 'use_dual_augment', False)
    if is_main_process and use_dual_augment:
        log.info("Using dual-path augmentation (clean for teacher, augmented for student)")

    train_loader, train_sampler, val_loader, test_loader = create_distributed_dataloaders(
        config, world_size, rank, use_dual_augment=use_dual_augment
    )

    if is_main_process:
        log.info(f"Train samples: {len(train_loader.dataset)} ({len(train_loader.dataset)//world_size} per GPU)")
        log.info(f"Effective batch size: {config.data.batch_size * world_size}")
        log.info(f"Token layers for distillation: {config.ss_distillation.token_layers}")
        log.info(f"Lambda tok: {config.ss_distillation.lambda_tok}, Lambda rel: {config.ss_distillation.lambda_rel}")
        log.info(f"L_rel warmup epochs: {config.ss_distillation.rel_warmup_epochs}")

    # Create trainer
    trainer = SelfSupervisedDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        teacher_embed_dim=teacher_embed_dim,
        config=config,
        device=device,
        rank=rank,
        world_size=world_size
    )

    dist.barrier()

    if is_main_process:
        log.info("=" * 60)
        log.info("STARTING SELF-SUPERVISED DISTILLATION TRAINING")
        log.info("=" * 60)

    # Train
    metrics_history = trainer.train_ddp(train_loader, train_sampler, val_loader)

    # Evaluate (main process only)
    if is_main_process:
        extra_info = {
            'Teacher': f"{config.ss_distillation.teacher_type} ({config.ss_distillation.teacher_model_name})",
            'Token Layers': str(config.ss_distillation.token_layers),
            'Lambda Tok': config.ss_distillation.lambda_tok,
            'Lambda Rel': config.ss_distillation.lambda_rel,
            'L_rel Warmup': f"{config.ss_distillation.rel_warmup_epochs} epochs"
        }
        return evaluate_and_save_results(
            trainer, test_loader, dataset_info, config, output_dir,
            metrics_history, world_size, mode_name, extra_info
        )
    return None


# =============================================================================
# Unified Local Training Entry Point
# =============================================================================

def train_locally(config_path: str, num_gpus: int = 1, mode: str = 'standard') -> int:
    """
    Train on local GPUs using DDP for multi-GPU.

    Args:
        config_path: Path to configuration file
        num_gpus: Number of GPUs to use
        mode: Training mode ('standard', 'distill', 'ss_distill')
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA GPUs available. Please check your GPU setup.")

    if num_gpus > available_gpus:
        print(f"Requested {num_gpus} GPUs, but only {available_gpus} available. Using {available_gpus}.")
        num_gpus = available_gpus

    mode_names = {
        'standard': 'PyTorch DDP Training',
        'distill': 'DeiT Distillation Training',
        'ss_distill': 'Self-Supervised Token Distillation (CST-Style) Training'
    }

    # Find free port for this training run
    port = find_free_port()

    print(f"\n{'='*60}")
    print(mode_names[mode])
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"GPUs: {num_gpus} x {torch.cuda.get_device_name(0)}")
    print(f"Mode: {'DDP Multi-GPU' if num_gpus > 1 else 'Single GPU'}")
    print(f"Port: {port}")
    print(f"{'='*60}\n")

    if num_gpus == 1:
        result = unified_ddp_worker(0, 1, config_path, mode, port)
    else:
        mp.spawn(
            unified_ddp_worker,
            args=(num_gpus, config_path, mode, port),
            nprocs=num_gpus,
            join=True
        )
        result = {"status": "completed", "num_gpus": num_gpus}

    if result and isinstance(result, dict):
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        if 'test_accuracy' in result:
            print(f"Experiment: {result['experiment_name']}")
            print(f"Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"GPUs Used: {result['num_gpus']}")
            print(f"Output Directory: {result['output_dir']}")
        else:
            print(f"Training completed with {result.get('num_gpus', num_gpus)} GPUs")
        print(f"{'='*60}\n")

    return 0


# =============================================================================
# Evaluation and Analysis Functions
# =============================================================================

def evaluate_model(config_path: str, checkpoint_path: str) -> None:
    """Evaluate a trained model on the test set."""
    config = ConfigManager.load_config(config_path)
    setup_logging(config.logging)

    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    dataset_info = DatasetManager.get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']
    config.model.dataset = config.data.dataset

    model_config = build_model_config(config)
    model = ModelFactory.create_model(config.model.model_type, model_config)
    model = model.to(device)

    add_safe_globals_for_checkpoints()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {checkpoint_path}")

    _, _, test_loader = DatasetManager.create_data_loaders(config)

    evaluator = ModelEvaluator(model, device, dataset_info.get('classes'))
    evaluator.evaluate(test_loader)
    evaluator.print_summary()

    output_dir = Path(config.output_dir) / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator.plot_confusion_matrix(save_path=output_dir / 'confusion_matrix.png', normalize=True)
    evaluator.plot_roc_curves(save_path=output_dir / 'roc_curves.png')
    evaluator.plot_class_distribution(save_path=output_dir / 'class_distribution.png')

    misclassified = evaluator.get_misclassified_samples(n_samples=20)
    logger.info(f"Top misclassified samples: {misclassified[:5]}")


def test_single_image(config_path: str, checkpoint_path: str, image_path: str, use_tta: bool = False) -> None:
    """Test model on a single image with visualization."""
    config = ConfigManager.load_config(config_path)
    setup_logging(config.logging)

    device = torch.device(config.device)

    dataset_info = DatasetManager.get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']
    config.model.dataset = config.data.dataset

    model_config = build_model_config(config)
    model = ModelFactory.create_model(config.model.model_type, model_config)
    model = model.to(device)

    add_safe_globals_for_checkpoints()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image = preprocess_image(image_path, config)
    image = image.to(device)

    if use_tta:
        tta = TestTimeAugmentation(model, device, n_augmentations=10)
        output = tta.predict(image, [])
    else:
        with torch.no_grad():
            output = model(image)
            output = torch.softmax(output, dim=1)

    pred_prob, pred_class = output.max(1)
    class_names = dataset_info.get('classes', [str(i) for i in range(dataset_info['num_classes'])])

    print("\nPrediction Results:")
    print(f"Predicted Class: {class_names[pred_class.item()]}")
    print(f"Confidence: {pred_prob.item():.4f}")

    top5_probs, top5_classes = output.topk(5, dim=1)
    print("\nTop-5 Predictions:")
    for i in range(5):
        class_idx = top5_classes[0, i].item()
        prob = top5_probs[0, i].item()
        print(f"{i+1}. {class_names[class_idx]}: {prob:.4f}")

    # Visualizations
    output_dir = Path('outputs') / 'inference'
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = FeatureMapVisualizer(model, device)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            visualizer.visualize_feature_maps(
                image, name, n_features=32,
                save_path=output_dir / f'feature_maps_{name.replace("/", "_")}.png'
            )
            break

    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_name = name

    if last_conv_name:
        original_img = Image.open(image_path)
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        original_img = np.array(original_img)

        gradcam = GradCAM(model, last_conv_name, device)
        gradcam.visualize(image, original_img, class_idx=pred_class.item(), save_path=output_dir / 'gradcam.png')


def analyze_model(
    config_path: str,
    checkpoint_path: str,
    metrics: str = 'all',
    output_dir: Optional[str] = None,
    num_samples: int = 1024
) -> dict:
    """
    Run research-grade analytics on a trained model.

    Computes:
    - Hessian trace and top eigenvalues (loss landscape curvature)
    - Mean attention distance per layer (for ViT models)
    - CKA similarity matrix (layer-wise representations)
    """
    import json

    config = ConfigManager.load_config(config_path)
    setup_logging(config.logging)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running analytics on device: {device}")

    dataset_info = DatasetManager.get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']
    config.model.dataset = config.data.dataset

    model_config = build_model_config(config)
    model = ModelFactory.create_model(config.model.model_type, model_config)
    model = model.to(device)

    add_safe_globals_for_checkpoints()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")

    # Unwrap compiled model if necessary
    if hasattr(model, '_orig_mod'):
        logger.info("Unwrapping torch.compile model for analytics")
        model = model._orig_mod

    # Parse metrics
    if metrics.lower() == 'all':
        metric_list = ['hessian', 'attention', 'cka']
    else:
        metric_list = [m.strip().lower() for m in metrics.split(',')]

    logger.info(f"Analytics metrics to compute: {metric_list}")

    # Create data loader
    test_dataset = DatasetManager.get_dataset(config, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    # Set up output directory
    out_path = Path(output_dir) if output_dir else Path(config.output_dir) / 'analytics'
    out_path.mkdir(parents=True, exist_ok=True)

    # Analytics config
    class AnalyticsConfig:
        def __init__(self, cfg, n_samples):
            self.hessian_samples = n_samples
            self.cka_kernel = 'linear'
            self.vit = cfg.vit if hasattr(cfg, 'vit') else None

    analytics_config = AnalyticsConfig(config, num_samples)

    # Run analytics
    runner = AnalyticsRunner(model=model, config=analytics_config, device=device, teacher_model=None)
    results = runner.run_all(
        dataloader=test_loader,
        metrics=metric_list,
        save_path=out_path / 'analytics_results.json'
    )

    # Generate visualizations
    logger.info("=" * 60)
    logger.info("Generating Analytics Visualizations")
    logger.info("=" * 60)

    if 'cka' in results and 'cka_matrix' in results['cka']:
        AnalyticsVisualizer.plot_cka_heatmap(
            cka_matrix=results['cka']['cka_matrix'],
            layer_indices_x=results['cka'].get('layer_indices_1'),
            layer_indices_y=results['cka'].get('layer_indices_2'),
            title=f"CKA Similarity - {config.experiment_name}",
            save_path=out_path / 'cka_heatmap.png'
        )

    if 'attention' in results and 'layer_distances' in results['attention']:
        AnalyticsVisualizer.plot_attention_distances(
            layer_distances=results['attention']['layer_distances'],
            title=f"Mean Attention Distance - {config.experiment_name}",
            save_path=out_path / 'attention_distances.png'
        )

    # Print summary
    logger.info("=" * 60)
    logger.info("ANALYTICS SUMMARY")
    logger.info("=" * 60)

    if 'hessian' in results:
        hessian = results['hessian']
        if 'trace' in hessian:
            logger.info(f"Hessian Trace: {hessian['trace']:.4f} +/- {hessian.get('trace_std', 0):.4f}")
        if 'max_eigenvalue' in hessian:
            logger.info(f"Max Eigenvalue: {hessian['max_eigenvalue']:.4f}")

    if 'attention' in results and 'mean_distance' in results['attention']:
        logger.info(f"Mean Attention Distance: {results['attention']['mean_distance']:.4f}")

    if 'cka' in results and 'cka_matrix' in results['cka']:
        cka_matrix = np.array(results['cka']['cka_matrix'])
        if cka_matrix.shape[0] == cka_matrix.shape[1]:
            diagonal_mean = np.diag(cka_matrix).mean()
            logger.info(f"CKA Diagonal Mean: {diagonal_mean:.4f}")

    logger.info("=" * 60)
    logger.info(f"Analytics results saved to: {out_path}")
    logger.info("=" * 60)

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Adaptive CNN Training System',
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model on local GPUs')
    train_parser.add_argument('config', type=str, help='Path to configuration file')
    train_parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model locally')
    eval_parser.add_argument('config', type=str, help='Path to configuration file')
    eval_parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test on single image with visualization')
    test_parser.add_argument('config', type=str, help='Path to configuration file')
    test_parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    test_parser.add_argument('image', type=str, help='Path to input image')
    test_parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')

    # Distillation command
    distill_parser = subparsers.add_parser('train-distill', help='Train DeiT with knowledge distillation')
    distill_parser.add_argument('config', type=str, help='Path to DeiT config file')
    distill_parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')

    # Self-supervised distillation command
    ss_distill_parser = subparsers.add_parser(
        'train-ss-distill',
        help='Train DeiT with self-supervised (CST-style) token distillation from DINO/DINOv2'
    )
    ss_distill_parser.add_argument('config', type=str, help='Path to config file with ss_distillation section')
    ss_distill_parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Run research-grade analytics on a trained model (Hessian, Attention Distance, CKA)'
    )
    analyze_parser.add_argument('config', type=str, help='Path to configuration file')
    analyze_parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    analyze_parser.add_argument('--metrics', type=str, default='all',
                               help='Comma-separated metrics: hessian,attention,cka or all (default: all)')
    analyze_parser.add_argument('--output-dir', type=str, default=None,
                               help='Output directory for results (default: outputs/analytics)')
    analyze_parser.add_argument('--num-samples', type=int, default=1024,
                               help='Number of samples for Hessian analysis (default: 1024)')

    args = parser.parse_args()

    if args.command == 'train':
        return train_locally(args.config, args.num_gpus, mode='standard')
    elif args.command == 'train-distill':
        return train_locally(args.config, args.num_gpus, mode='distill')
    elif args.command == 'train-ss-distill':
        return train_locally(args.config, args.num_gpus, mode='ss_distill')
    elif args.command == 'evaluate':
        evaluate_model(args.config, args.checkpoint)
    elif args.command == 'test':
        test_single_image(args.config, args.checkpoint, args.image, args.tta)
    elif args.command == 'analyze':
        analyze_model(
            args.config,
            args.checkpoint,
            metrics=args.metrics,
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )
    else:
        parser.print_help()

    return 0


if __name__ == '__main__':
    exit(main())
