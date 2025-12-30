import argparse
import logging
import os
import random
from pathlib import Path

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
from src.visualization import FeatureMapVisualizer, GradCAM, TrainingVisualizer, AnalyticsVisualizer
from src.analytics import AnalyticsRunner

logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def train_worker(
    rank: int,
    world_size: int,
    config_path: str
):
    """DDP training worker for a single GPU process."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    result = None
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        is_main_process = (rank == 0)

        config = ConfigManager.load_config(config_path)

        if world_size > 1:
            config.experiment_name = f"{config.experiment_name}_ddp_{world_size}gpu"

        output_dir = Path(config.output_dir) / config.experiment_name
        checkpoints_dir = output_dir / "checkpoints"

        if is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

        dist.barrier()

        if is_main_process:
            setup_logging(config.logging)
            logger = logging.getLogger(__name__)
            logger.info("=" * 60)
            logger.info("PyTorch Distributed Data Parallel (DDP) Training")
            logger.info("=" * 60)
            logger.info(f"Experiment: {config.experiment_name}")
            logger.info(f"Dataset: {config.data.dataset}")
            logger.info(f"World Size: {world_size}")
            logger.info(f"GPUs: {world_size} x {torch.cuda.get_device_name(0)}")
            logger.info("Backend: NCCL")
            logger.info("=" * 60)
        else:
            logging.basicConfig(level=logging.ERROR)
            logger = logging.getLogger(__name__)

        set_seed(config.seed + rank)

        if is_main_process:
            logger.info(f"Process {rank}: Using device cuda:{rank}")
            logger.info(f"GPU Name: {torch.cuda.get_device_name(rank)}")
            logger.info(f"Total GPUs Available: {torch.cuda.device_count()}")

        dataset_info = DatasetManager.get_dataset_info(config)
        config.model.in_channels = dataset_info['in_channels']
        config.model.num_classes = dataset_info['num_classes']
        config.model.dataset = config.data.dataset

        # Build model config (merge vit config for DeiT models)
        model_config = config.model.__dict__.copy()
        if config.model.model_type == 'deit' and hasattr(config, 'vit') and config.vit is not None:
            model_config.update(config.vit.__dict__ if hasattr(config.vit, '__dict__') else config.vit)

        model = ModelFactory.create_model(config.model.model_type, model_config)
        model = model.to(device)

        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )

        if is_main_process:
            total_params = sum(p.numel() for p in model.module.parameters())
            trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
            logger.info("Model wrapped with DistributedDataParallel")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Effective batch size: {config.data.batch_size * world_size}")

        if is_main_process:
            logger.info("Creating distributed data loaders...")

        train_dataset = DatasetManager.get_dataset(config, is_train=True)
        val_dataset = DatasetManager.get_dataset(config, is_train=False)
        test_dataset = DatasetManager.get_dataset(config, is_train=False)

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

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            sampler=train_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else 2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            sampler=val_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else 2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory
        )

        if is_main_process:
            logger.info(f"Train samples: {len(train_dataset)} ({len(train_dataset)//world_size} per GPU)")
            logger.info(f"Val samples: {len(val_dataset)} ({len(val_dataset)//world_size} per GPU)")
            logger.info(f"Test samples: {len(test_dataset)}")

        ddp_trainer = DDPTrainer(model, config, device, rank, world_size)

        checkpoint_path = checkpoints_dir / f"best_model_{config.data.dataset}_ddp.pth"

        if is_main_process:
            if checkpoint_path.exists():
                logger.info(f"Found checkpoint: {checkpoint_path}")
                try:
                    ddp_trainer.load_checkpoint(checkpoint_path)
                    logger.info(f"Resumed from epoch {ddp_trainer.current_epoch}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}")
            else:
                logger.info("No checkpoint found. Starting fresh training.")

        dist.barrier()

        if is_main_process:
            logger.info("=" * 60)
            logger.info("STARTING DDP TRAINING")
            logger.info("=" * 60)

        metrics_history = ddp_trainer.train_ddp(train_loader, train_sampler, val_loader)

        if is_main_process:
            logger.info("=" * 60)
            logger.info("EVALUATING ON TEST SET")
            logger.info("=" * 60)

            evaluator = ModelEvaluator(ddp_trainer.model, device, dataset_info.get('classes'))
            test_metrics = evaluator.evaluate(test_loader)
            evaluator.print_summary()

            ConfigManager.save_config(config, output_dir / 'config.yaml')

            if metrics_history:
                TrainingVisualizer.plot_training_history(
                    metrics_history,
                    save_path=output_dir / 'training_history.png'
                )

            evaluator.plot_confusion_matrix(
                save_path=output_dir / 'confusion_matrix.png',
                normalize=True
            )

            metrics_file = output_dir / 'test_metrics.txt'
            with open(metrics_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("FINAL TEST METRICS (DDP Training)\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Configuration: {world_size} GPUs with DDP\n")
                f.write(f"Effective Batch Size: {config.data.batch_size * world_size}\n\n")
                f.write(f"Accuracy:          {test_metrics['accuracy']:.4f}\n")
                f.write(f"Precision (macro): {test_metrics['precision_macro']:.4f}\n")
                f.write(f"Recall (macro):    {test_metrics['recall_macro']:.4f}\n")
                f.write(f"F1 Score (macro):  {test_metrics['f1_macro']:.4f}\n")
                f.write(f"Loss:              {test_metrics.get('loss', 'N/A')}\n")

                if 'auc_macro' in test_metrics:
                    f.write(f"\nAUC Macro: {test_metrics['auc_macro']:.4f}\n")
                    f.write(f"AUC Weighted: {test_metrics['auc_weighted']:.4f}\n")

            logger.info("=" * 60)
            logger.info("DDP TRAINING COMPLETED SUCCESSFULLY")
            logger.info(f"Experiment: {config.experiment_name}")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Outputs saved to: {output_dir}")
            logger.info("=" * 60)

            result = {
                'experiment_name': config.experiment_name,
                'test_accuracy': test_metrics['accuracy'],
                'output_dir': str(output_dir),
                'num_gpus': world_size
            }

    finally:
        dist.destroy_process_group()

    return result


def train_locally(config_path: str, num_gpus: int = 1):
    """Train on local GPUs using DDP for multi-GPU."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA GPUs available. Please check your GPU setup.")

    if num_gpus > available_gpus:
        print(f"Requested {num_gpus} GPUs, but only {available_gpus} available. Using {available_gpus}.")
        num_gpus = available_gpus

    print(f"\n{'='*60}")
    print("PyTorch Distributed Data Parallel (DDP) Training")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"GPUs: {num_gpus} x {torch.cuda.get_device_name(0)}")
    print(f"Mode: {'DDP Multi-GPU' if num_gpus > 1 else 'Single GPU'}")
    print(f"{'='*60}\n")

    if num_gpus == 1:
        result = train_worker(0, 1, config_path)
    else:
        mp.spawn(train_worker, args=(num_gpus, config_path), nprocs=num_gpus, join=True)
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
            print(f"DDP Training completed with {result.get('num_gpus', num_gpus)} GPUs")
        print(f"{'='*60}\n")

    return 0


def train_distill_worker(
    rank: int,
    world_size: int,
    config_path: str
):
    """DDP training worker for DeiT distillation training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29501')  # Use env or default
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    result = None
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        is_main_process = (rank == 0)

        config = ConfigManager.load_config(config_path)

        if world_size > 1:
            config.experiment_name = f"{config.experiment_name}_ddp_{world_size}gpu"

        output_dir = Path(config.output_dir) / config.experiment_name
        checkpoints_dir = output_dir / "checkpoints"

        if is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

        dist.barrier()

        if is_main_process:
            setup_logging(config.logging)
            logger = logging.getLogger(__name__)
            logger.info("=" * 60)
            logger.info("DeiT Distillation Training (DDP)")
            logger.info("=" * 60)
            logger.info(f"Experiment: {config.experiment_name}")
            logger.info(f"Dataset: {config.data.dataset}")
            logger.info(f"World Size: {world_size}")
            logger.info(f"GPUs: {world_size} x {torch.cuda.get_device_name(0)}")
            logger.info("=" * 60)
        else:
            logging.basicConfig(level=logging.ERROR)
            logger = logging.getLogger(__name__)

        set_seed(config.seed + rank)

        # Get dataset info
        dataset_info = DatasetManager.get_dataset_info(config)
        config.model.in_channels = dataset_info['in_channels']
        config.model.num_classes = dataset_info['num_classes']
        config.model.dataset = config.data.dataset

        # Build student model config (merge model and vit configs)
        student_config = config.model.__dict__.copy()
        if config.vit is not None:
            student_config.update(config.vit.__dict__)

        # Create student model (DeiT)
        student_model = ModelFactory.create_model('deit', student_config)
        student_model = student_model.to(device)

        if is_main_process:
            total_params = sum(p.numel() for p in student_model.parameters())
            logger.info(f"Student model (DeiT) created with {total_params:,} parameters")

        # Load teacher model
        teacher_checkpoint_path = Path(config.distillation.teacher_checkpoint)
        if not teacher_checkpoint_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")

        # Create teacher model (AdaptiveCNN)
        teacher_config = config.model.__dict__.copy()
        teacher_model = ModelFactory.create_model(
            config.distillation.teacher_model_type,
            teacher_config
        )
        teacher_model = teacher_model.to(device)

        # Load teacher weights
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([
                Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig,
                ViTConfig, DistillationConfig
            ])
        teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
        if 'model_state_dict' not in teacher_checkpoint:
            raise KeyError(
                f"Teacher checkpoint at {teacher_checkpoint_path} is missing 'model_state_dict' key. "
                f"Available keys: {list(teacher_checkpoint.keys())}"
            )
        teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
        teacher_model.eval()

        if is_main_process:
            teacher_acc = teacher_checkpoint.get('best_val_acc', 'N/A')
            logger.info(f"Teacher model loaded from: {teacher_checkpoint_path}")
            logger.info(f"Teacher validation accuracy: {teacher_acc}")
            if isinstance(teacher_acc, (int, float)) and teacher_acc > 99.7:
                logger.warning("Teacher accuracy > 99.7%. Consider using lower alpha to prevent dominance.")

        # Wrap student with DDP
        # DeiT has a distillation token that may not be used in all loss calculations
        student_model = DDP(
            student_model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True  # Required for DeiT distillation token
        )

        # Create data loaders
        train_dataset = DatasetManager.get_dataset(config, is_train=True)
        val_dataset = DatasetManager.get_dataset(config, is_train=False)
        test_dataset = DatasetManager.get_dataset(config, is_train=False)

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

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            sampler=train_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else 2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            sampler=val_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else 2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory
        )

        if is_main_process:
            logger.info(f"Train samples: {len(train_dataset)} ({len(train_dataset)//world_size} per GPU)")
            logger.info(f"Effective batch size: {config.data.batch_size * world_size}")

        # Create distillation trainer
        distill_trainer = DistillationTrainer(
            student_model, teacher_model, config, device, rank, world_size
        )

        dist.barrier()

        if is_main_process:
            logger.info("=" * 60)
            logger.info("STARTING DISTILLATION TRAINING")
            logger.info("=" * 60)

        # Train
        metrics_history = distill_trainer.train_ddp(train_loader, train_sampler, val_loader)

        # Evaluate on test set
        if is_main_process:
            logger.info("=" * 60)
            logger.info("EVALUATING ON TEST SET")
            logger.info("=" * 60)

            # Use unwrapped model for evaluation (consistent with regular training)
            eval_model = distill_trainer.ddp_model.module if hasattr(distill_trainer.ddp_model, 'module') else distill_trainer.ddp_model
            evaluator = ModelEvaluator(eval_model, device, dataset_info.get('classes'))
            test_metrics = evaluator.evaluate(test_loader)
            evaluator.print_summary()

            # Save config and visualizations
            ConfigManager.save_config(config, output_dir / 'config.yaml')

            if metrics_history:
                TrainingVisualizer.plot_training_history(
                    metrics_history,
                    save_path=output_dir / 'training_history.png'
                )

            evaluator.plot_confusion_matrix(
                save_path=output_dir / 'confusion_matrix.png',
                normalize=True
            )

            # Save metrics
            metrics_file = output_dir / 'test_metrics.txt'
            with open(metrics_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("DEIT DISTILLATION TRAINING RESULTS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Configuration: {world_size} GPUs with DDP\n")
                f.write(f"Teacher: {config.distillation.teacher_model_type}\n")
                f.write(f"Distillation Type: {config.distillation.distillation_type}\n")
                f.write(f"Alpha: {config.distillation.alpha}\n\n")
                f.write(f"Accuracy:          {test_metrics['accuracy']:.4f}\n")
                f.write(f"Precision (macro): {test_metrics['precision_macro']:.4f}\n")
                f.write(f"Recall (macro):    {test_metrics['recall_macro']:.4f}\n")
                f.write(f"F1 Score (macro):  {test_metrics['f1_macro']:.4f}\n")

            logger.info("=" * 60)
            logger.info("DISTILLATION TRAINING COMPLETED")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Outputs saved to: {output_dir}")
            logger.info("=" * 60)

            result = {
                'experiment_name': config.experiment_name,
                'test_accuracy': test_metrics['accuracy'],
                'output_dir': str(output_dir),
                'num_gpus': world_size
            }

    finally:
        dist.destroy_process_group()

    return result


def train_distillation_locally(config_path: str, num_gpus: int = 1):
    """Train DeiT with distillation on local GPUs using DDP."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA GPUs available.")

    if num_gpus > available_gpus:
        print(f"Requested {num_gpus} GPUs, but only {available_gpus} available. Using {available_gpus}.")
        num_gpus = available_gpus

    print(f"\n{'='*60}")
    print("DeiT Distillation Training (DDP)")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"GPUs: {num_gpus} x {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    if num_gpus == 1:
        result = train_distill_worker(0, 1, config_path)
    else:
        mp.spawn(train_distill_worker, args=(num_gpus, config_path), nprocs=num_gpus, join=True)
        result = {"status": "completed", "num_gpus": num_gpus}

    if result and isinstance(result, dict):
        print(f"\n{'='*60}")
        print("DISTILLATION TRAINING COMPLETED")
        print(f"{'='*60}")
        if 'test_accuracy' in result:
            print(f"Experiment: {result['experiment_name']}")
            print(f"Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"GPUs Used: {result['num_gpus']}")
            print(f"Output Directory: {result['output_dir']}")
        print(f"{'='*60}\n")

    return 0


def train_ss_distill_worker(
    rank: int,
    world_size: int,
    config_path: str
):
    """DDP training worker for self-supervised (CST-style) distillation training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29502'  # Different port from other training modes
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    result = None
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        is_main_process = (rank == 0)

        config = ConfigManager.load_config(config_path)

        if world_size > 1:
            config.experiment_name = f"{config.experiment_name}_ddp_{world_size}gpu"

        output_dir = Path(config.output_dir) / config.experiment_name
        checkpoints_dir = output_dir / "checkpoints"

        if is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

        dist.barrier()

        if is_main_process:
            setup_logging(config.logging)
            logger = logging.getLogger(__name__)
            logger.info("=" * 60)
            logger.info("Self-Supervised Token Distillation (CST-Style) Training")
            logger.info("=" * 60)
            logger.info(f"Experiment: {config.experiment_name}")
            logger.info(f"Dataset: {config.data.dataset}")
            logger.info(f"World Size: {world_size}")
            logger.info(f"GPUs: {world_size} x {torch.cuda.get_device_name(0)}")
            logger.info("=" * 60)
        else:
            logging.basicConfig(level=logging.ERROR)
            logger = logging.getLogger(__name__)

        set_seed(config.seed + rank)

        # Get dataset info
        dataset_info = DatasetManager.get_dataset_info(config)
        config.model.in_channels = dataset_info['in_channels']
        config.model.num_classes = dataset_info['num_classes']
        config.model.dataset = config.data.dataset

        # Validate ss_distillation config exists
        if config.ss_distillation is None:
            raise ValueError("ss_distillation config is required for train-ss-distill command")

        # Build student model config (merge model and vit configs)
        student_config = config.model.__dict__.copy()
        if config.vit is not None:
            student_config.update(config.vit.__dict__)

        # Create student model (DeiT)
        student_model = ModelFactory.create_model('deit', student_config)
        student_model = student_model.to(device)

        if is_main_process:
            total_params = sum(p.numel() for p in student_model.parameters())
            logger.info(f"Student model (DeiT) created with {total_params:,} parameters")

        # Load self-supervised teacher (DINO/DINOv2)
        if is_main_process:
            logger.info(f"Loading {config.ss_distillation.teacher_type} teacher: {config.ss_distillation.teacher_model_name}")

        teacher_model, teacher_embed_dim = load_dino_teacher(
            config.ss_distillation.teacher_type,
            config.ss_distillation.teacher_model_name,
            device
        )

        if is_main_process:
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            logger.info(f"Teacher model loaded with {teacher_params:,} parameters (frozen)")
            logger.info(f"Teacher embedding dim: {teacher_embed_dim}")

        # Wrap student with DDP
        student_model = DDP(
            student_model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True  # Required for DeiT distillation token
        )

        # Create data loaders
        # Use dual-augment loaders if enabled (clean for teacher, augmented for student)
        use_dual_augment = getattr(config.ss_distillation, 'use_dual_augment', False)

        if use_dual_augment:
            if is_main_process:
                logger.info("Using dual-path augmentation (clean for teacher, augmented for student)")
            train_dataset = DatasetManager.get_dual_augment_dataset(config)
        else:
            train_dataset = DatasetManager.get_dataset(config, is_train=True)

        val_dataset = DatasetManager.get_dataset(config, is_train=False)
        test_dataset = DatasetManager.get_dataset(config, is_train=False)

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

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            sampler=train_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else 2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            sampler=val_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else 2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory
        )

        if is_main_process:
            logger.info(f"Train samples: {len(train_dataset)} ({len(train_dataset)//world_size} per GPU)")
            logger.info(f"Effective batch size: {config.data.batch_size * world_size}")
            logger.info(f"Token layers for distillation: {config.ss_distillation.token_layers}")
            logger.info(f"Lambda tok: {config.ss_distillation.lambda_tok}, Lambda rel: {config.ss_distillation.lambda_rel}")
            logger.info(f"L_rel warmup epochs: {config.ss_distillation.rel_warmup_epochs}")

        # Create self-supervised distillation trainer
        ss_distill_trainer = SelfSupervisedDistillationTrainer(
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
            logger.info("=" * 60)
            logger.info("STARTING SELF-SUPERVISED DISTILLATION TRAINING")
            logger.info("=" * 60)

        # Train
        metrics_history = ss_distill_trainer.train_ddp(train_loader, train_sampler, val_loader)

        # Evaluate on test set
        if is_main_process:
            logger.info("=" * 60)
            logger.info("EVALUATING ON TEST SET")
            logger.info("=" * 60)

            # Use unwrapped model for evaluation
            eval_model = ss_distill_trainer.ddp_model.module if hasattr(ss_distill_trainer.ddp_model, 'module') else ss_distill_trainer.ddp_model
            evaluator = ModelEvaluator(eval_model, device, dataset_info.get('classes'))
            test_metrics = evaluator.evaluate(test_loader)
            evaluator.print_summary()

            # Save config and visualizations
            ConfigManager.save_config(config, output_dir / 'config.yaml')

            if metrics_history:
                TrainingVisualizer.plot_training_history(
                    metrics_history,
                    save_path=output_dir / 'training_history.png'
                )

            evaluator.plot_confusion_matrix(
                save_path=output_dir / 'confusion_matrix.png',
                normalize=True
            )

            # Save metrics
            metrics_file = output_dir / 'test_metrics.txt'
            with open(metrics_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("SELF-SUPERVISED TOKEN DISTILLATION (CST-STYLE) RESULTS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Configuration: {world_size} GPUs with DDP\n")
                f.write(f"Teacher: {config.ss_distillation.teacher_type} ({config.ss_distillation.teacher_model_name})\n")
                f.write(f"Token Layers: {config.ss_distillation.token_layers}\n")
                f.write(f"Lambda Tok: {config.ss_distillation.lambda_tok}\n")
                f.write(f"Lambda Rel: {config.ss_distillation.lambda_rel}\n")
                f.write(f"L_rel Warmup: {config.ss_distillation.rel_warmup_epochs} epochs\n\n")
                f.write(f"Accuracy:          {test_metrics['accuracy']:.4f}\n")
                f.write(f"Precision (macro): {test_metrics['precision_macro']:.4f}\n")
                f.write(f"Recall (macro):    {test_metrics['recall_macro']:.4f}\n")
                f.write(f"F1 Score (macro):  {test_metrics['f1_macro']:.4f}\n")

            logger.info("=" * 60)
            logger.info("SELF-SUPERVISED DISTILLATION TRAINING COMPLETED")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Outputs saved to: {output_dir}")
            logger.info("=" * 60)

            result = {
                'experiment_name': config.experiment_name,
                'test_accuracy': test_metrics['accuracy'],
                'output_dir': str(output_dir),
                'num_gpus': world_size
            }

    finally:
        dist.destroy_process_group()

    return result


def train_ss_distillation_locally(config_path: str, num_gpus: int = 1):
    """Train DeiT with self-supervised (CST-style) distillation on local GPUs using DDP."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA GPUs available.")

    if num_gpus > available_gpus:
        print(f"Requested {num_gpus} GPUs, but only {available_gpus} available. Using {available_gpus}.")
        num_gpus = available_gpus

    print(f"\n{'='*60}")
    print("Self-Supervised Token Distillation (CST-Style) Training")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"GPUs: {num_gpus} x {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    if num_gpus == 1:
        result = train_ss_distill_worker(0, 1, config_path)
    else:
        mp.spawn(train_ss_distill_worker, args=(num_gpus, config_path), nprocs=num_gpus, join=True)
        result = {"status": "completed", "num_gpus": num_gpus}

    if result and isinstance(result, dict):
        print(f"\n{'='*60}")
        print("SELF-SUPERVISED DISTILLATION TRAINING COMPLETED")
        print(f"{'='*60}")
        if 'test_accuracy' in result:
            print(f"Experiment: {result['experiment_name']}")
            print(f"Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"GPUs Used: {result['num_gpus']}")
            print(f"Output Directory: {result['output_dir']}")
        print(f"{'='*60}\n")

    return 0


def evaluate_model(config_path, checkpoint_path):
    """Evaluate a trained model on the test set."""
    config = ConfigManager.load_config(config_path)
    setup_logging(config.logging)

    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    dataset_info = DatasetManager.get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']
    config.model.dataset = config.data.dataset

    # Build model config (merge model and vit configs for DeiT)
    model_config = config.model.__dict__.copy()
    if hasattr(config, 'vit') and config.vit is not None:
        model_config.update(config.vit.__dict__)

    model = ModelFactory.create_model(config.model.model_type, model_config)
    model = model.to(device)

    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            Config,
            DataConfig,
            ModelConfig,
            TrainingConfig,
            LoggingConfig
        ])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {checkpoint_path}")

    _, _, test_loader = DatasetManager.create_data_loaders(config)

    evaluator = ModelEvaluator(model, device, dataset_info.get('classes'))
    evaluator.evaluate(test_loader)

    evaluator.print_summary()

    output_dir = Path(config.output_dir) / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator.plot_confusion_matrix(
        save_path=output_dir / 'confusion_matrix.png',
        normalize=True
    )

    evaluator.plot_roc_curves(save_path=output_dir / 'roc_curves.png')

    evaluator.plot_class_distribution(save_path=output_dir / 'class_distribution.png')

    misclassified = evaluator.get_misclassified_samples(n_samples=20)
    logger.info(f"Top misclassified samples: {misclassified[:5]}")


def test_single_image(config_path, checkpoint_path, image_path, use_tta=False):
    """Test model on a single image with visualization."""
    config = ConfigManager.load_config(config_path)
    setup_logging(config.logging)

    device = torch.device(config.device)

    dataset_info = DatasetManager.get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']
    config.model.dataset = config.data.dataset

    # Build model config (merge model and vit configs for DeiT)
    model_config = config.model.__dict__.copy()
    if hasattr(config, 'vit') and config.vit is not None:
        model_config.update(config.vit.__dict__)

    model = ModelFactory.create_model(config.model.model_type, model_config)
    model = model.to(device)

    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            Config,
            DataConfig,
            ModelConfig,
            TrainingConfig,
            LoggingConfig
        ])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image = preprocess_image(image_path, config)
    image = image.to(device)

    if use_tta:
        tta = TestTimeAugmentation(model, device, n_augmentations=10)
        transforms = []
        output = tta.predict(image, transforms)
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
        gradcam.visualize(
            image, original_img,
            class_idx=pred_class.item(),
            save_path=output_dir / 'gradcam.png'
        )


def analyze_model(
    config_path: str,
    checkpoint_path: str,
    metrics: str = 'all',
    output_dir: str = None,
    num_samples: int = 1024
):
    """
    Run research-grade analytics on a trained model.

    Computes:
    - Hessian trace and top eigenvalues (loss landscape curvature)
    - Mean attention distance per layer (for ViT models)
    - CKA similarity matrix (layer-wise representations)

    Args:
        config_path: Path to model configuration file
        checkpoint_path: Path to model checkpoint
        metrics: Comma-separated list of metrics ('hessian', 'attention', 'cka', or 'all')
        output_dir: Directory to save results (default: outputs/analytics)
        num_samples: Number of samples for analysis
    """
    import json

    config = ConfigManager.load_config(config_path)
    setup_logging(config.logging)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running analytics on device: {device}")

    # Get dataset info
    dataset_info = DatasetManager.get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']
    config.model.dataset = config.data.dataset

    # Build model config (merge model and vit configs for DeiT)
    model_config = config.model.__dict__.copy()
    if hasattr(config, 'vit') and config.vit is not None:
        model_config.update(config.vit.__dict__)

    # Create model
    model = ModelFactory.create_model(config.model.model_type, model_config)
    model = model.to(device)

    # Load checkpoint
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig,
            ViTConfig, DistillationConfig, SelfSupervisedDistillationConfig
        ])

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")

    # Unwrap compiled model if necessary (CRITICAL for Hessian)
    if hasattr(model, '_orig_mod'):
        logger.info("Unwrapping torch.compile model for analytics")
        model = model._orig_mod

    # Parse metrics
    if metrics.lower() == 'all':
        metric_list = ['hessian', 'attention', 'cka']
    else:
        metric_list = [m.strip().lower() for m in metrics.split(',')]

    logger.info(f"Analytics metrics to compute: {metric_list}")

    # Create data loader for analytics
    test_dataset = DatasetManager.get_dataset(config, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,  # Smaller batch for Hessian stability
        shuffle=True,  # Random samples
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    # Set up output directory
    if output_dir is None:
        output_dir = Path(config.output_dir) / 'analytics'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add config parameters for analytics
    class AnalyticsConfig:
        def __init__(self, config, num_samples):
            self.hessian_samples = num_samples
            self.cka_kernel = 'linear'
            # Copy vit config if available
            if hasattr(config, 'vit') and config.vit is not None:
                self.vit = config.vit
            else:
                self.vit = None

    analytics_config = AnalyticsConfig(config, num_samples)

    # Run analytics
    runner = AnalyticsRunner(
        model=model,
        config=analytics_config,
        device=device,
        teacher_model=None  # Could add teacher for comparison
    )

    results = runner.run_all(
        dataloader=test_loader,
        metrics=metric_list,
        save_path=output_dir / 'analytics_results.json'
    )

    # Generate visualizations
    logger.info("=" * 60)
    logger.info("Generating Analytics Visualizations")
    logger.info("=" * 60)

    # CKA Heatmap
    if 'cka' in results and 'cka_matrix' in results['cka']:
        AnalyticsVisualizer.plot_cka_heatmap(
            cka_matrix=results['cka']['cka_matrix'],
            layer_indices_x=results['cka'].get('layer_indices_1'),
            layer_indices_y=results['cka'].get('layer_indices_2'),
            title=f"CKA Similarity - {config.experiment_name}",
            save_path=output_dir / 'cka_heatmap.png'
        )

    # Attention Distance
    if 'attention' in results and 'layer_distances' in results['attention']:
        AnalyticsVisualizer.plot_attention_distances(
            layer_distances=results['attention']['layer_distances'],
            title=f"Mean Attention Distance - {config.experiment_name}",
            save_path=output_dir / 'attention_distances.png'
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
        if 'eigenvalue_ratio' in hessian:
            logger.info(f"Eigenvalue Ratio: {hessian['eigenvalue_ratio']:.2f}")

    if 'attention' in results and 'mean_distance' in results['attention']:
        logger.info(f"Mean Attention Distance: {results['attention']['mean_distance']:.4f}")

    if 'cka' in results and 'cka_matrix' in results['cka']:
        cka_matrix = np.array(results['cka']['cka_matrix'])
        diagonal_mean = np.diag(cka_matrix).mean() if cka_matrix.shape[0] == cka_matrix.shape[1] else 0
        logger.info(f"CKA Diagonal Mean: {diagonal_mean:.4f}")

    logger.info("=" * 60)
    logger.info(f"Analytics results saved to: {output_dir}")
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Adaptive CNN Training System',
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    train_parser = subparsers.add_parser('train', help='Train a model on local GPUs')
    train_parser.add_argument('config', type=str, help='Path to configuration file')
    train_parser.add_argument('--num-gpus', type=int, default=1,
                             help='Number of GPUs to use (default: 1)')

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model locally')
    eval_parser.add_argument('config', type=str, help='Path to configuration file')
    eval_parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')

    test_parser = subparsers.add_parser('test', help='Test on single image with visualization')
    test_parser.add_argument('config', type=str, help='Path to configuration file')
    test_parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    test_parser.add_argument('image', type=str, help='Path to input image')
    test_parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')

    distill_parser = subparsers.add_parser('train-distill', help='Train DeiT with knowledge distillation')
    distill_parser.add_argument('config', type=str, help='Path to DeiT config file')
    distill_parser.add_argument('--num-gpus', type=int, default=1,
                                help='Number of GPUs to use (default: 1)')

    ss_distill_parser = subparsers.add_parser(
        'train-ss-distill',
        help='Train DeiT with self-supervised (CST-style) token distillation from DINO/DINOv2'
    )
    ss_distill_parser.add_argument('config', type=str, help='Path to config file with ss_distillation section')
    ss_distill_parser.add_argument('--num-gpus', type=int, default=1,
                                   help='Number of GPUs to use (default: 1)')

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
        return train_locally(args.config, args.num_gpus)
    elif args.command == 'train-distill':
        return train_distillation_locally(args.config, args.num_gpus)
    elif args.command == 'train-ss-distill':
        return train_ss_distillation_locally(args.config, args.num_gpus)
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
