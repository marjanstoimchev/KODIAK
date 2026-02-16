# =============================================================================
# Classification DataModule
# =============================================================================
# PyTorch Lightning DataModule for downstream classification tasks.
# Supports train/val/test splits with smart HuggingFace split detection.
# =============================================================================

import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import functional as TF

from .transforms import ClassificationTrainTransform, ClassificationValTransform
from .collate import classification_collate
from ..utils import SamplerType, make_sampler

logger = logging.getLogger("kodiak.data.classification")


class TransformDataset(Dataset):
    """
    Wraps a base dataset and applies transforms for classification.
    """
    def __init__(self, base_subset, transform: Optional[nn.Module]):
        self.base = base_subset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        if isinstance(item, dict):
            image = item.get("image")
            meta = {k: v for k, v in item.items() if k != "image"}
        else:
            image = item
            meta = {}

        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)  # (C,H,W) in [0,1]

        if self.transform is not None:
            image = self.transform(image)

        return {"images": image, **meta}


class ClassificationDataModule(pl.LightningDataModule):
    """
    DataModule for downstream classification tasks.

    Features:
    - Smart train/val/test splitting based on available HuggingFace splits
    - Classification transforms (resize, augmentation, normalize)
    - Support for HuggingFace datasets and custom datasets
    - DINOv3-style samplers (DISTRIBUTED by default)

    Split Logic:
    - If dataset has train+val+test: use all as-is
    - If dataset has train+test: use test as-is, split val from train
    - If dataset has only train: split into train/val/test

    Usage:
        datamodule = ClassificationDataModule(
            dataset_type="huggingface",
            hf_dataset_name="cansa/Describable-Textures-Dataset-DTD",
            batch_size=64,
        )
    """

    def __init__(
        self,
        dataset_type: str = 'huggingface',
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        # Split configuration
        train_split: float = 0.7,
        val_split: float = 0.1,
        test_split: float = 0.2,
        # Augmentation config
        image_size: int = 256,  # DINOv3 official default
        normalize_mean: list = None,
        normalize_std: list = None,
        # Sampler config
        sampler_type: SamplerType = SamplerType.DISTRIBUTED,
        # Seed for reproducible splits
        seed: int = 42,
        # Custom dataset args
        csv_path: Optional[str] = None,
        root_dir: Optional[str] = None,
        magnification: Optional[str] = None,
        root_path: Optional[str] = None,
        # HuggingFace dataset args
        hf_dataset_name: Optional[str] = None,
        hf_split: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['batch_size'])

        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = bool(persistent_workers and num_workers > 0)

        # Split config
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        # Dataset args
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.magnification = magnification
        self.root_path = root_path
        self.hf_dataset_name = hf_dataset_name
        self.hf_split = hf_split
        self.hf_cache_dir = hf_cache_dir

        # Sampler config
        self.sampler_type = sampler_type
        self.seed = seed

        # Augmentation config
        mean = normalize_mean or [0.485, 0.456, 0.406]
        std = normalize_std or [0.229, 0.224, 0.225]

        self.train_transform = ClassificationTrainTransform(
            image_size=image_size,
            mean=mean,
            std=std
        )
        self.val_transform = ClassificationValTransform(
            image_size=image_size,
            mean=mean,
            std=std
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _create_base_dataset(self) -> Dataset:
        """Create the base dataset using the DatasetRegistry factory."""
        from ..utils import DatasetRegistry

        kwargs = {}

        if self.dataset_type == 'pancreatic':
            if self.root_dir:
                logger.info(f"Loading pancreatic dataset from folder: {self.root_dir}")
                kwargs = {'root_dir': self.root_dir}
            else:
                logger.info(f"Loading pancreatic dataset from CSV: {self.csv_path}")
                kwargs = {
                    'csv_file': self.csv_path,
                    'magnification_filter': self.magnification,
                    'root_path': self.root_path
                }

        elif self.dataset_type == 'huggingface':
            logger.info(f"Loading HuggingFace dataset: {self.hf_dataset_name}")
            kwargs = {
                'name': self.hf_dataset_name,
                'split': self.hf_split,
                'cache_dir': self.hf_cache_dir
            }

        return DatasetRegistry.create(self.dataset_type, **kwargs)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets with smart splitting."""
        if self.dataset_type == 'huggingface' and self.hf_dataset_name:
            self._setup_huggingface()
        else:
            self._setup_simple()

    def _setup_huggingface(self):
        """Smart splitting for HuggingFace datasets based on available splits."""
        from ..datasets.huggingface.dataset import get_available_splits, HuggingFaceDataset

        splits_info = get_available_splits(self.hf_dataset_name, self.hf_cache_dir)
        has_train = splits_info['train']
        has_val = splits_info['val']
        has_test = splits_info['test']
        val_key = splits_info['val_key']

        logger.info(f"Dataset {self.hf_dataset_name} splits: {splits_info['available']}")

        if has_train and has_val and has_test:
            # Case 1: All splits available - use as-is
            logger.info("Using existing train/val/test splits")
            train_ds = HuggingFaceDataset(self.hf_dataset_name, split='train', cache_dir=self.hf_cache_dir)
            val_ds = HuggingFaceDataset(self.hf_dataset_name, split=val_key, cache_dir=self.hf_cache_dir)
            test_ds = HuggingFaceDataset(self.hf_dataset_name, split='test', cache_dir=self.hf_cache_dir)

            self.train_dataset = TransformDataset(train_ds, self.train_transform)
            self.val_dataset = TransformDataset(val_ds, self.val_transform)
            self.test_dataset = TransformDataset(test_ds, self.val_transform)

        elif has_train and has_test:
            # Case 2: train+test available - split val from train
            logger.info(f"Splitting val from train (val_split={self.val_split})")
            train_ds = HuggingFaceDataset(self.hf_dataset_name, split='train', cache_dir=self.hf_cache_dir)
            test_ds = HuggingFaceDataset(self.hf_dataset_name, split='test', cache_dir=self.hf_cache_dir)

            total_train = len(train_ds)
            val_size = int(self.val_split * total_train)
            train_size = total_train - val_size

            train_base, val_base = random_split(
                train_ds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.train_dataset = TransformDataset(train_base, self.train_transform)
            self.val_dataset = TransformDataset(val_base, self.val_transform)
            self.test_dataset = TransformDataset(test_ds, self.val_transform)

        else:
            # Case 3: Only train available - split into train/val/test
            logger.info(f"Splitting train into train/val/test ({self.train_split}/{self.val_split}/{self.test_split})")
            train_ds = HuggingFaceDataset(self.hf_dataset_name, split='train', cache_dir=self.hf_cache_dir)

            total_size = len(train_ds)
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size

            train_base, val_base, test_base = random_split(
                train_ds,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.train_dataset = TransformDataset(train_base, self.train_transform)
            self.val_dataset = TransformDataset(val_base, self.val_transform)
            self.test_dataset = TransformDataset(test_base, self.val_transform)

        logger.info(f"Classification: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")

    def _setup_simple(self):
        """Simple splitting for non-HuggingFace datasets."""
        full_dataset = self._create_base_dataset()
        total_size = len(full_dataset)

        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size

        train_base, val_base, test_base = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.train_dataset = TransformDataset(train_base, self.train_transform)
        self.val_dataset = TransformDataset(val_base, self.val_transform)
        self.test_dataset = TransformDataset(test_base, self.val_transform)

        logger.info(f"Classification: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")

    def train_dataloader(self):
        # Build sampler (DISTRIBUTED by default)
        sampler = make_sampler(
            dataset=self.train_dataset,
            sampler_type=self.sampler_type,
            shuffle=True,
            seed=self.seed,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            collate_fn=classification_collate
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=classification_collate
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=classification_collate
        )
