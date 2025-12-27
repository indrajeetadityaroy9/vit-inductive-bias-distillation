import json
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataConfig:
    def __init__(self, dataset=None, batch_size=64, num_workers=4, pin_memory=True,
                 persistent_workers=True, prefetch_factor=2, augmentation=None,
                 normalization=None, data_path="./data"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.augmentation = augmentation if augmentation is not None else {}
        self.normalization = normalization if normalization is not None else {}
        self.data_path = data_path

class ModelConfig:
    def __init__(self, model_type="adaptive_cnn", in_channels=1, num_classes=10,
                 dropout=0.5, use_se=True, architecture=None, classifier_layers=None):
        self.model_type = model_type
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_se = use_se
        self.architecture = architecture if architecture is not None else []
        self.classifier_layers = classifier_layers if classifier_layers is not None else []

class TrainingConfig:
    def __init__(self, num_epochs=10, learning_rate=0.001, weight_decay=0.0005,
                 optimizer="adamw", scheduler="cosine", warmup_epochs=5,
                 gradient_clip_val=1.0, gradient_accumulation_steps=1,
                 use_amp=True, amp_backend="native", early_stopping=True,
                 early_stopping_patience=10, early_stopping_min_delta=0.001,
                 lr_scheduler_params=None, label_smoothing=0.1, use_swa=True,
                 swa_start_epoch=0.75, swa_lr=0.0005,
                 # H100 optimization flags
                 use_bf16=True, use_compile=True, compile_mode='max-autotune',
                 use_fused_optimizer=True, use_tf32=True):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.gradient_clip_val = gradient_clip_val
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.amp_backend = amp_backend
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.lr_scheduler_params = lr_scheduler_params if lr_scheduler_params is not None else {}
        self.label_smoothing = label_smoothing
        self.use_swa = use_swa
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        # H100 optimization flags
        self.use_bf16 = use_bf16  # Use BF16 instead of FP16 for better numerical stability
        self.use_compile = use_compile  # Enable torch.compile for graph optimization
        self.compile_mode = compile_mode  # 'max-autotune', 'reduce-overhead', or 'default'
        self.use_fused_optimizer = use_fused_optimizer  # Use fused AdamW/SGD kernels
        self.use_tf32 = use_tf32  # Enable TF32 for matmul operations

class LoggingConfig:
    def __init__(self, log_level="INFO", log_dir="./logs", wandb=False,
                 wandb_project="adaptive-cnn", wandb_entity=None,
                 save_frequency=5, track_grad_norm=True):
        self.log_level = log_level
        self.log_dir = log_dir
        self.wandb = wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.save_frequency = save_frequency
        self.track_grad_norm = track_grad_norm


class ViTConfig:
    """Vision Transformer (DeiT) specific configuration."""

    def __init__(self, variant='tiny', img_size=32, patch_size=4,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 distillation=True, convert_grayscale=True,
                 # New generalized improvements
                 use_conv_stem=False, cls_token_dropout=0.0, inference_mode='avg'):
        self.variant = variant
        self.img_size = img_size
        self.patch_size = patch_size

        # Set defaults based on variant if not explicitly provided
        variant_configs = {
            'tiny': {'embed_dim': 192, 'num_heads': 3},
            'small': {'embed_dim': 384, 'num_heads': 6},
            'base': {'embed_dim': 768, 'num_heads': 12},
        }
        if variant in variant_configs:
            self.embed_dim = embed_dim if embed_dim != 192 else variant_configs[variant]['embed_dim']
            self.num_heads = num_heads if num_heads != 3 else variant_configs[variant]['num_heads']
        else:
            self.embed_dim = embed_dim
            self.num_heads = num_heads

        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.distillation = distillation
        self.convert_grayscale = convert_grayscale

        # New generalized improvements
        self.use_conv_stem = use_conv_stem  # Enable hybrid conv stem for better local features
        self.cls_token_dropout = cls_token_dropout  # Probability of replacing cls token with patch mean
        self.inference_mode = inference_mode  # 'cls', 'dist', or 'avg' for inference


class DistillationConfig:
    """Knowledge distillation configuration."""

    def __init__(self, teacher_checkpoint=None, teacher_model_type='adaptive_cnn',
                 distillation_type='hard', alpha=0.5, tau=3.0,
                 distillation_warmup_epochs=0,
                 # Alpha scheduling for distillation
                 alpha_schedule='constant', alpha_start=0.0, alpha_end=0.5):
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_model_type = teacher_model_type
        self.distillation_type = distillation_type  # 'hard' or 'soft'
        self.alpha = alpha  # Weight for distillation loss (used when alpha_schedule='constant')
        self.tau = tau  # Temperature for soft distillation
        self.distillation_warmup_epochs = distillation_warmup_epochs  # Epochs without distillation

        # Alpha scheduling: 'constant', 'linear', or 'cosine'
        self.alpha_schedule = alpha_schedule
        self.alpha_start = alpha_start  # Starting alpha (after warmup)
        self.alpha_end = alpha_end  # Ending alpha (at final epoch)


class SelfSupervisedDistillationConfig:
    """
    Self-supervised token correlation distillation configuration (CST-style).

    Uses a pretrained self-supervised ViT (DINO/DINOv2) as teacher instead of
    a weaker CNN, distilling token representations and correlations.
    """

    def __init__(
        self,
        # Teacher model settings
        teacher_type='dinov2',              # 'dino' or 'dinov2'
        teacher_model_name='dinov2_vits14', # Model name for torch.hub
        teacher_embed_dim=384,              # Teacher embedding dimension
        # Token representation distillation (L_tok) - PRIMARY SIGNAL
        token_layers=None,                  # Layer indices to extract [6, 11] default
        projection_dim=256,                 # Dimension for alignment projectors
        lambda_tok=1.0,                     # Weight for token representation loss
        token_loss_type='cosine',           # 'cosine' or 'mse'
        # Token correlation distillation (L_rel) - LIGHTWEIGHT REGULARIZER
        lambda_rel=0.1,                     # Weight for correlation loss (keep small)
        correlation_temperature=0.1,        # Temperature for softening correlations
        correlation_loss_type='kl',         # 'kl' (stable) or 'frobenius'
        use_pooled_correlation=True,        # Use patch-mean pooling to avoid O(N²)
        # Staged training (essential for stability)
        rel_warmup_epochs=10,               # Epochs before enabling L_rel
        projector_warmup_epochs=0,          # Epochs to freeze projectors (optional)
    ):
        self.teacher_type = teacher_type
        self.teacher_model_name = teacher_model_name
        self.teacher_embed_dim = teacher_embed_dim

        # Token representation distillation
        self.token_layers = token_layers if token_layers is not None else [6, 11]
        self.projection_dim = projection_dim
        self.lambda_tok = lambda_tok
        self.token_loss_type = token_loss_type

        # Token correlation distillation
        self.lambda_rel = lambda_rel
        self.correlation_temperature = correlation_temperature
        self.correlation_loss_type = correlation_loss_type
        self.use_pooled_correlation = use_pooled_correlation

        # Staged training
        self.rel_warmup_epochs = rel_warmup_epochs
        self.projector_warmup_epochs = projector_warmup_epochs


class Config:
    def __init__(self, data=None, model=None, training=None, logging=None,
                 vit=None, distillation=None, ss_distillation=None,
                 experiment_name="default", seed=42, device="cuda", output_dir="./outputs"):
        self.data = data
        self.model = model
        self.training = training
        self.logging = logging
        self.vit = vit  # ViT-specific configuration
        self.distillation = distillation  # Knowledge distillation configuration
        self.ss_distillation = ss_distillation  # Self-supervised distillation (CST-style)
        self.experiment_name = experiment_name
        self.seed = seed
        self.device = device
        self.output_dir = output_dir

class ConfigManager:

    @staticmethod
    def load_config(config_path):
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                raw_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        # Parse optional vit and distillation configs
        vit_config = None
        if 'vit' in raw_config:
            vit_config = ViTConfig(**raw_config['vit'])

        distillation_config = None
        if 'distillation' in raw_config:
            distillation_config = DistillationConfig(**raw_config['distillation'])

        ss_distillation_config = None
        if 'ss_distillation' in raw_config:
            ss_distillation_config = SelfSupervisedDistillationConfig(**raw_config['ss_distillation'])

        config = Config(
            data=DataConfig(**raw_config.get('data', {})),
            model=ModelConfig(**raw_config.get('model', {})),
            training=TrainingConfig(**raw_config.get('training', {})),
            logging=LoggingConfig(**raw_config.get('logging', {})),
            vit=vit_config,
            distillation=distillation_config,
            ss_distillation=ss_distillation_config,
            **{k: v for k, v in raw_config.items()
               if k not in ['data', 'model', 'training', 'logging', 'vit', 'distillation', 'ss_distillation']}
        )

        ConfigManager.validate_config(config)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    @staticmethod
    def validate_config(config):

        valid_datasets = ['mnist', 'cifar', 'custom']
        if config.data.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {config.data.dataset}")

        # Validate model type
        valid_model_types = ['adaptive_cnn', 'deit']
        if config.model.model_type not in valid_model_types:
            raise ValueError(f"Invalid model_type: {config.model.model_type}")

        if config.data.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if config.model.dropout < 0 or config.model.dropout > 1:
            raise ValueError("Dropout must be between 0 and 1")

        if config.training.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        if config.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        valid_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
        if config.training.optimizer.lower() not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {config.training.optimizer}")

        valid_schedulers = ['step', 'cosine', 'plateau', 'exponential', 'cyclic']
        if config.training.scheduler.lower() not in valid_schedulers:
            raise ValueError(f"Invalid scheduler: {config.training.scheduler}")

    @staticmethod
    def save_config(config, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            'data': config.data.__dict__,
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'logging': config.logging.__dict__,
            'experiment_name': config.experiment_name,
            'seed': config.seed,
            'device': config.device,
            'output_dir': config.output_dir
        }

        # Include optional vit and distillation configs
        if config.vit is not None:
            config_dict['vit'] = config.vit.__dict__
        if config.distillation is not None:
            config_dict['distillation'] = config.distillation.__dict__
        if config.ss_distillation is not None:
            config_dict['ss_distillation'] = config.ss_distillation.__dict__

        if save_path.suffix in ['.yml', '.yaml']:
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

        logger.info(f"Saved configuration to {save_path}")

    @staticmethod
    def get_default_config(dataset):
        if dataset == 'mnist':
            return Config(
                data=DataConfig(
                    dataset='mnist',
                    batch_size=64,
                    augmentation={},
                    normalization={'mean': [0.1307], 'std': [0.3081]}
                ),
                model=ModelConfig(
                    in_channels=1,
                    num_classes=10,
                    dropout=0.5
                ),
                training=TrainingConfig(
                    num_epochs=10,
                    learning_rate=0.001,
                    weight_decay=0.0005,
                    lr_scheduler_params={'step_size': 5, 'gamma': 0.1}
                ),
                logging=LoggingConfig()
            )
        elif dataset == 'cifar':
            return Config(
                data=DataConfig(
                    dataset='cifar',
                    batch_size=128,
                    augmentation={
                        'random_crop': True,
                        'random_flip': True,
                        'color_jitter': True,
                        'cutout': True
                    },
                    normalization={
                        'mean': [0.4914, 0.4822, 0.4465],
                        'std': [0.2470, 0.2435, 0.2616]
                    }
                ),
                model=ModelConfig(
                    in_channels=3,
                    num_classes=10,
                    dropout=0.5
                ),
                training=TrainingConfig(
                    num_epochs=200,
                    learning_rate=0.1,
                    weight_decay=0.0005,
                    optimizer='sgd',
                    scheduler='cosine',
                    warmup_epochs=10,
                    lr_scheduler_params={'T_max': 200, 'eta_min': 0.0001}
                ),
                logging=LoggingConfig()
            )
        else:
            raise ValueError(f"No default config for dataset: {dataset}")

def setup_logging(config):
    log_level = getattr(logging, config.log_level.upper())

    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(config.log_dir) / 'training.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("Logging configured successfully")
