import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import numpy as np
import logging
from PIL import Image, ImageOps
import random

logger = logging.getLogger(__name__)

class Cutout:

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class MixingDataset(Dataset):
    """
    Unified MixUp/CutMix augmentation wrapper.

    Combines the functionality of MixUpDataset and CutMixDataset into
    a single class controlled by the mix_type parameter.
    """

    def __init__(self, dataset, mix_type='mixup', alpha=1.0, num_classes=10):
        """
        Args:
            dataset: Base dataset to wrap
            mix_type: 'mixup' or 'cutmix'
            alpha: Beta distribution parameter for mixing ratio
            num_classes: Number of classes for one-hot encoding
        """
        self.dataset = dataset
        self.mix_type = mix_type
        self.alpha = alpha
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, target1 = self.dataset[idx]

        idx2 = random.randint(0, len(self.dataset) - 1)
        img2, target2 = self.dataset[idx2]

        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1

        if self.mix_type == 'cutmix':
            img, lam = self._apply_cutmix(img1, img2, lam)
        else:  # mixup
            img = lam * img1 + (1 - lam) * img2

        # One-hot encode targets
        target1_oh = torch.zeros(self.num_classes)
        target2_oh = torch.zeros(self.num_classes)
        target1_oh[target1] = 1
        target2_oh[target2] = 1

        target = lam * target1_oh + (1 - lam) * target2_oh

        return img, target

    def _apply_cutmix(self, img1, img2, lam):
        """Apply CutMix augmentation and return adjusted lambda."""
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(img1.size(), lam)
        img = img1.clone()
        img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img1.size()[-1] * img1.size()[-2]))
        return img, lam

    def _rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix."""
        W = size[2]
        H = size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


# Backward compatibility aliases
MixUpDataset = lambda dataset, alpha=1.0, num_classes=10: MixingDataset(dataset, 'mixup', alpha, num_classes)
CutMixDataset = lambda dataset, alpha=1.0, num_classes=10: MixingDataset(dataset, 'cutmix', alpha, num_classes)


class DualAugmentDataset(Dataset):
    """
    Dual-path augmentation dataset for self-supervised distillation.

    Returns (clean_img, augmented_img, target) tuple where:
    - clean_img: Only normalized (for teacher like DINOv2)
    - augmented_img: Full augmentation pipeline (for student like DeiT)
    - target: Integer class label (MixUp/CutMix applied in training loop)

    This avoids the "MixUp Trap" where self-supervised teachers (trained on
    natural images) degrade when fed heavily augmented inputs.

    Usage in training loop:
        for batch in loader:
            clean_imgs, student_imgs, targets = batch
            # Resize clean_imgs to 224x224 for DINOv2
            clean_224 = F.interpolate(clean_imgs, (224, 224), mode='bicubic')
            with torch.no_grad():
                teacher_features = teacher(clean_224)
            # Apply MixUp ONLY to student_imgs and targets (in loop, not here)
            if mixup_fn:
                student_imgs, targets = mixup_fn(student_imgs, targets)
            student_out = student(student_imgs)
    """

    def __init__(self, base_dataset, clean_transform, student_transform):
        """
        Args:
            base_dataset: Raw dataset (e.g., CIFAR10 with transform=None)
            clean_transform: Minimal transform for teacher (resize, to_tensor, normalize)
            student_transform: Full augmentation for student (crop, flip, cutout, etc.)
        """
        self.base_dataset = base_dataset
        self.clean_transform = clean_transform
        self.student_transform = student_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get raw image and target from base dataset
        # Note: base_dataset should have transform=None or return PIL images
        img, target = self.base_dataset[idx]

        # If img is already a tensor (base dataset has transform), we need raw
        if torch.is_tensor(img):
            raise ValueError(
                "DualAugmentDataset requires base_dataset with transform=None. "
                "Got tensor instead of PIL Image."
            )

        # Apply different transforms for teacher (clean) and student (augmented)
        clean_img = self.clean_transform(img)
        augmented_img = self.student_transform(img)

        return clean_img, augmented_img, target


class DatasetManager:

    @staticmethod
    def get_transforms(config, is_train=True):
        dataset = config.data.dataset
        aug_config = config.data.augmentation if is_train else {}

        transform_list = []

        if dataset == 'mnist':
            if is_train and aug_config:
                if aug_config.get('random_rotation'):
                    transform_list.append(transforms.RandomRotation(10))
                if aug_config.get('random_affine'):
                    transform_list.append(transforms.RandomAffine(
                        degrees=0, translate=(0.1, 0.1)
                    ))
                # RandAugment for MNIST (must be before ToTensor)
                if aug_config.get('randaugment'):
                    n_ops = aug_config.get('randaugment_n', 2)
                    magnitude = aug_config.get('randaugment_m', 9)
                    transform_list.append(transforms.RandAugment(num_ops=n_ops, magnitude=magnitude))

            transform_list.extend([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.data.normalization['mean'],
                    std=config.data.normalization['std']
                )
            ])

            if is_train and aug_config.get('cutout'):
                transform_list.append(Cutout(n_holes=1, length=8))

        elif dataset == 'cifar':
            if is_train and aug_config:
                if aug_config.get('random_crop'):
                    transform_list.append(transforms.RandomCrop(32, padding=4))
                if aug_config.get('random_flip'):
                    transform_list.append(transforms.RandomHorizontalFlip())
                if aug_config.get('color_jitter'):
                    transform_list.append(transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                    ))
                if aug_config.get('auto_augment'):
                    transform_list.append(transforms.AutoAugment(
                        transforms.AutoAugmentPolicy.CIFAR10
                    ))
                # RandAugment for CIFAR (must be before ToTensor)
                if aug_config.get('randaugment'):
                    n_ops = aug_config.get('randaugment_n', 2)
                    magnitude = aug_config.get('randaugment_m', 9)
                    transform_list.append(transforms.RandAugment(num_ops=n_ops, magnitude=magnitude))

            transform_list.extend([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.data.normalization['mean'],
                    std=config.data.normalization['std']
                )
            ])

            if is_train and aug_config.get('cutout'):
                transform_list.append(Cutout(n_holes=1, length=16))

        else:

            if is_train and aug_config:
                if aug_config.get('random_crop'):
                    size = aug_config.get('image_size', 224)
                    transform_list.append(transforms.RandomResizedCrop(size))
                if aug_config.get('random_flip'):
                    transform_list.append(transforms.RandomHorizontalFlip())
                if aug_config.get('color_jitter'):
                    transform_list.append(transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ))
                # RandAugment for custom datasets (must be before ToTensor)
                if aug_config.get('randaugment'):
                    n_ops = aug_config.get('randaugment_n', 2)
                    magnitude = aug_config.get('randaugment_m', 9)
                    transform_list.append(transforms.RandAugment(num_ops=n_ops, magnitude=magnitude))

            size = aug_config.get('image_size', 224)
            transform_list.extend([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.data.normalization.get('mean', [0.485, 0.456, 0.406]),
                    std=config.data.normalization.get('std', [0.229, 0.224, 0.225])
                )
            ])

        return transforms.Compose(transform_list)

    @staticmethod
    def get_dataset(config, is_train=True):
        dataset_name = config.data.dataset
        transform = DatasetManager.get_transforms(config, is_train)

        if dataset_name == 'mnist':
            dataset = datasets.MNIST(
                root=config.data.data_path,
                train=is_train,
                download=True,
                transform=transform
            )
        elif dataset_name == 'cifar':
            dataset = datasets.CIFAR10(
                root=config.data.data_path,
                train=is_train,
                download=True,
                transform=transform
            )
        elif dataset_name == 'fashion_mnist':
            dataset = datasets.FashionMNIST(
                root=config.data.data_path,
                train=is_train,
                download=True,
                transform=transform
            )
        elif dataset_name == 'svhn':
            split = 'train' if is_train else 'test'
            dataset = datasets.SVHN(
                root=config.data.data_path,
                split=split,
                download=True,
                transform=transform
            )
        else:

            from torchvision.datasets import ImageFolder
            data_dir = f"{config.data.data_path}/{dataset_name}"
            split = 'train' if is_train else 'test'
            dataset = ImageFolder(
                root=f"{data_dir}/{split}",
                transform=transform
            )

        if is_train:
            # Determine num_classes for MixUp/CutMix
            num_classes = config.model.num_classes if hasattr(config, 'model') and hasattr(config.model, 'num_classes') else 10
            if config.data.augmentation.get('mixup'):
                dataset = MixUpDataset(dataset, alpha=config.data.augmentation.get('mixup_alpha', 1.0), num_classes=num_classes)
            elif config.data.augmentation.get('cutmix'):
                dataset = CutMixDataset(dataset, alpha=config.data.augmentation.get('cutmix_alpha', 1.0), num_classes=num_classes)

        return dataset

    @staticmethod
    def create_data_loaders(config, val_split=0.1):

        full_train_dataset = DatasetManager.get_dataset(config, is_train=True)

        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed)
        )

        test_dataset = DatasetManager.get_dataset(config, is_train=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else 2,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
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

        logger.info(f"Created data loaders - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    @staticmethod
    def get_dataset_info(config):
        dataset = DatasetManager.get_dataset(config, is_train=False)
        sample, _ = dataset[0]

        if isinstance(sample, (list, tuple)):
            sample = sample[0]

        if not torch.is_tensor(sample):
            sample = transforms.ToTensor()(sample)

        if sample.dim() == 2:
            sample = sample.unsqueeze(0)

        in_channels = sample.shape[0]
        image_size = sample.shape[-1]

        if hasattr(dataset, 'classes') and dataset.classes:
            classes = list(dataset.classes)
        elif hasattr(dataset, 'class_to_idx') and dataset.class_to_idx:
            classes = [name for name, _ in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]
        else:
            raw_targets = None
            if hasattr(dataset, 'targets'):
                raw_targets = dataset.targets
            elif hasattr(dataset, 'labels'):
                raw_targets = dataset.labels

            if raw_targets is None:
                num_classes = getattr(config.model, 'num_classes', None)
                if num_classes is None:
                    raise ValueError(f"Unable to determine class count for dataset {config.data.dataset}")
                classes = [str(i) for i in range(num_classes)]
            else:
                if isinstance(raw_targets, torch.Tensor):
                    raw_targets = raw_targets.tolist()
                elif isinstance(raw_targets, np.ndarray):
                    raw_targets = raw_targets.tolist()
                num_classes = len(set(raw_targets))
                classes = [str(i) for i in range(num_classes)]

        num_classes = len(classes)

        return {
            'num_classes': num_classes,
            'in_channels': in_channels,
            'image_size': image_size,
            'classes': classes
        }

    @staticmethod
    def get_dual_augment_dataset(config):
        """
        Create a DualAugmentDataset for self-supervised distillation.

        Returns dataset that yields (clean_img, augmented_img, target) tuples.
        - clean_img: Minimal preprocessing for teacher (DINOv2)
        - augmented_img: Full augmentation for student (DeiT)
        - target: Integer class label (MixUp applied in training loop)

        Args:
            config: Configuration object

        Returns:
            DualAugmentDataset instance
        """
        dataset_name = config.data.dataset

        # Get raw dataset without transforms
        if dataset_name == 'mnist':
            base_dataset = datasets.MNIST(
                root=config.data.data_path,
                train=True,
                download=True,
                transform=None  # Raw PIL images
            )
        elif dataset_name == 'cifar':
            base_dataset = datasets.CIFAR10(
                root=config.data.data_path,
                train=True,
                download=True,
                transform=None  # Raw PIL images
            )
        elif dataset_name == 'fashion_mnist':
            base_dataset = datasets.FashionMNIST(
                root=config.data.data_path,
                train=True,
                download=True,
                transform=None
            )
        elif dataset_name == 'svhn':
            base_dataset = datasets.SVHN(
                root=config.data.data_path,
                split='train',
                download=True,
                transform=None
            )
        else:
            from torchvision.datasets import ImageFolder
            data_dir = f"{config.data.data_path}/{dataset_name}"
            base_dataset = ImageFolder(
                root=f"{data_dir}/train",
                transform=None
            )

        # Get transforms - clean uses is_train=False for minimal preprocessing
        clean_transform = DatasetManager.get_transforms(config, is_train=False)
        student_transform = DatasetManager.get_transforms(config, is_train=True)

        # Create dual augment dataset
        dual_dataset = DualAugmentDataset(
            base_dataset=base_dataset,
            clean_transform=clean_transform,
            student_transform=student_transform
        )

        logger.info(
            f"Created DualAugmentDataset for {dataset_name}: "
            f"{len(dual_dataset)} samples (clean + augmented paths)"
        )

        return dual_dataset

    @staticmethod
    def create_dual_augment_loaders(config, val_split=0.1):
        """
        Create data loaders for dual-path augmentation training.

        Returns train_loader that yields (clean_imgs, augmented_imgs, targets).

        Args:
            config: Configuration object
            val_split: Fraction for validation set

        Returns:
            train_loader, val_loader, test_loader
        """
        # Get dual augment training dataset
        full_train_dataset = DatasetManager.get_dual_augment_dataset(config)

        # Split into train/val
        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed)
        )

        # Test dataset uses standard single-path transform
        test_dataset = DatasetManager.get_dataset(config, is_train=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else 2,
            drop_last=True
        )

        # Validation uses clean transform only (single path)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
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

        logger.info(
            f"Created dual-augment data loaders - Train: {len(train_dataset)}, "
            f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

        return train_loader, val_loader, test_loader


def preprocess_image(image_path, config):
    image = Image.open(image_path)

    dataset_info = DatasetManager.get_dataset_info(config)
    if dataset_info['in_channels'] == 1:
        image = image.convert('L')

        if config.data.dataset == 'mnist':
            image_np = np.array(image)
            if image_np.mean() > 127:
                image = ImageOps.invert(image)
    else:
        image = image.convert('RGB')

    transform = DatasetManager.get_transforms(config, is_train=False)
    image = transform(image).unsqueeze(0)

    return image
