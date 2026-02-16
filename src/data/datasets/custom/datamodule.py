# custom/datamodule.py

from typing import Optional, Any, Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import functional as TF
from kornia.augmentation import AugmentationSequential, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, \
    ColorJitter, RandomGrayscale, RandomGaussianBlur, Normalize
from kornia.constants import DataKey

from .dataset import CustomPatchDataset


class KorniaAugModule(nn.Module):
    """
    Kornia augmentation pipeline.

    Expects CHW or BCHW, always returns CHW (for a single image).
    We enforce 3xH'xW' with explicit resize & normalization.
    """
    def __init__(self, aug_cfg: dict):
        super().__init__()

        image_size = int(aug_cfg.get("global_crop_size", 256))
        scale = tuple(aug_cfg.get("global_crop_scale", [0.4, 1.0]))
        ratio = tuple(aug_cfg.get("global_crop_ratio", [0.75, 1.33]))
        color_jitter_prob = float(aug_cfg.get("color_jitter_prob", 0.6))
        grayscale_prob = float(aug_cfg.get("grayscale_prob", 0.1))
        gaussian_blur_prob = float(aug_cfg.get("gaussian_blur_prob", 0.5))
        allow_vertical = bool(aug_cfg.get("allow_vertical_flip", True))

        mean = aug_cfg.get("normalize_mean", [0.485, 0.456, 0.406])
        std  = aug_cfg.get("normalize_std",  [0.229, 0.224, 0.225])

        aug_list = []

        # RandomResizedCrop to force square image_size x image_size
        aug_list.append(
            RandomResizedCrop(
                size=(image_size, image_size),
                scale=scale,
                ratio=ratio,
                resample="bilinear",
                align_corners=False,
            )
        )
        aug_list.append(RandomHorizontalFlip(p=0.5))
        if allow_vertical:
            aug_list.append(RandomVerticalFlip(p=0.5))

        if color_jitter_prob > 0.0:
            aug_list.append(
                ColorJitter(
                    p=color_jitter_prob,
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                )
            )

        if grayscale_prob > 0.0:
            aug_list.append(RandomGrayscale(p=grayscale_prob))

        if gaussian_blur_prob > 0.0:
            aug_list.append(
                RandomGaussianBlur(
                    kernel_size=(3, 3),
                    sigma=(0.1, 2.0),
                    p=gaussian_blur_prob,
                )
            )

        # Normalization at the end
        aug_list.append(
            Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        )

        self.aug = AugmentationSequential(
            *aug_list,
            data_keys=[DataKey.INPUT],
        )

        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: CHW or BCHW tensor in [0, 1] or arbitrary float.
        Returns CHW.
        """
        # Ensure CHW
        if x.dim() == 2:
            # Grayscale HW -> CHW (3-channel)
            x = x.unsqueeze(0).repeat(3, 1, 1)
        elif x.dim() == 3:
            # CHW as is
            pass
        elif x.dim() == 4:
            # If someone passes BCHW from outside, just take first
            x = x[0]
        else:
            raise ValueError(f"Unexpected input dim {x.dim()} for KorniaAugModule.")

        # Make sure we have 3 channels
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        elif x.size(0) > 3:
            # just in case some dataset has extra channels, keep first 3
            x = x[:3]

        # Ensure float32
        x = x.to(torch.float32)

        # Add batch dimension
        x = x.unsqueeze(0)  # 1, C, H, W

        # Run Kornia pipeline: output BCHW
        x = self.aug(x)

        # Remove batch dimension
        x = x.squeeze(0)    # C, H, W

        return x


class TransformDataset(torch.utils.data.Dataset):
    """
    Applies Kornia transforms and returns dict: {"images": Tensor[C,H,W], "magnification": str, ...}
    """
    def __init__(self, base_subset, transform: Optional[nn.Module]):
        self.base = base_subset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Convert numpy integers to Python int
        idx = int(idx)
        item = self.base[idx]
        if isinstance(item, dict):
            image = item.get("image")
            meta = {k: v for k, v in item.items() if k != "image"}
        else:
            image = item
            meta = {}

        # Convert PIL -> tensor CHW
        if not isinstance(image, torch.Tensor):
            # assumes PIL.Image or numpy-like
            image = TF.to_tensor(image)  # [0,1] float, C,H,W

        if self.transform is not None:
            image = self.transform(image)  # KorniaAugModule -> C,H,W

        return {"images": image, **meta}


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_split: float = 0.9,
        val_split: float = 0.1,
        magnification: Optional[str] = None,
        augmentation: Optional[dict] = None,
        root_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.csv_path = csv_path
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = bool(persistent_workers and num_workers > 0)
        self.train_split = train_split
        self.val_split = val_split
        self.magnification = magnification

        aug = augmentation or {}
        self.train_transform = KorniaAugModule(aug)
        # For val: we usually want deterministic center-ish crop + normalize; reuse Kornia with narrow scale
        val_aug = dict(aug)
        val_aug["global_crop_scale"] = [1.0, 1.0]
        self.val_transform = KorniaAugModule(val_aug)

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        full_dataset = CustomPatchDataset(
            csv_file=self.csv_path,
            magnification_filter=self.magnification,
            root_path=self.root_path,
        )

        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = total_size - train_size

        train_base, val_base = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_dataset = TransformDataset(train_base, self.train_transform)
        self.val_dataset   = TransformDataset(val_base,   self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
