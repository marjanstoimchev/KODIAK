# =============================================================================
# Pretraining DataModule
# =============================================================================
# PyTorch Lightning DataModule for self-supervised pretraining (KODIAK).
# Uses DINOv3-style augmentations and masking.
# =============================================================================

import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .transforms import DINOv3PretrainingTransform, MultiCropPretrainingTransform
from .collate import KodiakCollate, MultiCropKodiakCollate
from ..utils import MaskingGenerator, SamplerType, make_sampler

logger = logging.getLogger("kodiak.data.pretraining")


class TransformDataset(Dataset):
    """
    Wraps a base dataset and applies transforms.

    Handles two modes:
    1. Multi-Crop Pretraining: Transform returns dict with global_crops + local_crops
    2. Standard Pretraining: Transform returns dict with 'teacher_crop' and 'student_crop'
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
            result = self.transform(image)

            # Handle multi-crop pretraining (for MultiCropPrototypeCLSLoss)
            if isinstance(result, dict) and "global_crops" in result:
                return {
                    "global_crops": result["global_crops"],  # List of 2 tensors
                    "local_crops": result.get("local_crops", []),  # List of N tensors
                    **meta
                }
            # Handle standard cross-view pretraining (no local crops)
            elif isinstance(result, dict) and "teacher_crop" in result:
                return {
                    "teacher_image": result["teacher_crop"],
                    "student_image": result["student_crop"],
                    **meta
                }
            else:
                return {"images": result, **meta}

        return {"images": image, **meta}


class PretrainingDataModule(pl.LightningDataModule):
    """
    DataModule for self-supervised pretraining with KODIAK.

    Features:
    - DINOv3-style augmentations (2 global crops with different distortions)
    - Optional multi-crop training (2 global + N local crops)
    - DINOv3-style masking with variable mask ratios
    - Support for HuggingFace datasets and custom datasets
    - DINOv3-style samplers (DISTRIBUTED by default)

    Usage:
        datamodule = PretrainingDataModule(
            dataset_type="huggingface",
            hf_dataset_name="cansa/Describable-Textures-Dataset-DTD",
            batch_size=64,
            multi_crop=True,
        )
    """

    def __init__(
        self,
        dataset_type: str = 'huggingface',
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        # Augmentation config
        augmentation: Optional[dict] = None,
        # Multi-crop config
        multi_crop: bool = False,
        local_crops_number: int = 8,
        local_crops_size: int = 96,
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        # Masking config
        mask_ratio_tuple: Tuple[float, float] = (0.1, 0.5),
        mask_probability: float = 0.5,
        num_masks: int = 2,
        # Sampler config
        sampler_type: SamplerType = SamplerType.DISTRIBUTED,
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

        # Augmentation config
        aug = augmentation or {}
        self.img_size = int(aug.get("global_crops_size", 256))

        # Multi-crop config
        self.multi_crop = multi_crop
        self.local_crops_number = local_crops_number
        self.local_crops_size = local_crops_size
        self.local_crops_scale = local_crops_scale

        # Build transforms
        if multi_crop:
            multicrop_aug = dict(aug)
            multicrop_aug["local_crops_number"] = local_crops_number
            multicrop_aug["local_crops_size"] = local_crops_size
            multicrop_aug["local_crops_scale"] = list(local_crops_scale)
            self.train_transform = MultiCropPretrainingTransform(multicrop_aug)
        else:
            self.train_transform = DINOv3PretrainingTransform(aug)

        # Mask Generator Setup
        self.patch_size = 16
        self.grid_size = self.img_size // self.patch_size
        self.num_patches_total = self.grid_size ** 2

        self.mask_generator = MaskingGenerator(
            input_size=(self.grid_size, self.grid_size),
            max_num_patches=None,
            min_num_patches=4,
            min_aspect=0.3,
            max_aspect=None
        )

        self.mask_ratio_tuple = mask_ratio_tuple
        self.mask_probability = mask_probability
        self.num_masks = num_masks

        self.train_dataset = None

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
        """Setup datasets for pretraining."""
        if self.dataset_type == 'huggingface' and self.hf_dataset_name:
            self._setup_huggingface()
        else:
            self._setup_simple()

    def _setup_huggingface(self):
        """Setup HuggingFace dataset for pretraining.

        If hf_split is None or 'all', uses ALL available splits (recommended for small datasets).
        If hf_split is set to a specific split (e.g., 'train'), uses only that split.
        """
        from ..datasets.huggingface.dataset import get_available_splits, HuggingFaceDataset

        splits_info = get_available_splits(self.hf_dataset_name, self.hf_cache_dir)

        logger.info(f"Pretraining setup for {self.hf_dataset_name}")
        logger.info(f"  Available splits: {splits_info['available']}")
        logger.info(f"  Requested split: {self.hf_split or 'all (default)'}")

        # Determine which split(s) to use
        if self.hf_split and self.hf_split.lower() != 'all':
            # Use specific split requested
            logger.info(f"Using ONLY '{self.hf_split}' split for pretraining")
            train_ds = HuggingFaceDataset(self.hf_dataset_name, split=self.hf_split, cache_dir=self.hf_cache_dir)
            self.train_dataset = TransformDataset(train_ds, self.train_transform)
            logger.info(f"Pretraining: {len(self.train_dataset)} samples from '{self.hf_split}' split")
        else:
            # Use ALL splits for pretraining (recommended for small datasets)
            # Self-supervised learning doesn't use labels, so we can use all data
            logger.info("Using ALL splits for pretraining (self-supervised, no labels needed)")
            full_ds = HuggingFaceDataset(self.hf_dataset_name, split=None, cache_dir=self.hf_cache_dir)
            self.train_dataset = TransformDataset(full_ds, self.train_transform)
            logger.info(f"Pretraining: {len(self.train_dataset)} samples from ALL splits")

    def _setup_simple(self):
        """Setup non-HuggingFace dataset."""
        full_dataset = self._create_base_dataset()
        self.train_dataset = TransformDataset(full_dataset, self.train_transform)
        logger.info(f"Pretraining: train={len(self.train_dataset)} samples")

    def train_dataloader(self):
        # Build collate function
        if self.multi_crop:
            collate_fn = MultiCropKodiakCollate(
                mask_generator=self.mask_generator,
                mask_ratio_tuple=self.mask_ratio_tuple,
                mask_probability=self.mask_probability,
                num_patches_total=self.num_patches_total,
            )
        else:
            collate_fn = KodiakCollate(
                mask_generator=self.mask_generator,
                mask_ratio_tuple=self.mask_ratio_tuple,
                mask_probability=self.mask_probability,
                num_masks=self.num_masks,
                num_patches_total=self.num_patches_total
            )

        # Build sampler (DISTRIBUTED by default)
        sampler = make_sampler(
            dataset=self.train_dataset,
            sampler_type=self.sampler_type,
            shuffle=True,
            seed=42,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),  # Only shuffle if no sampler
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        """No validation dataloader for pretraining (DINOv3 style)."""
        return None
