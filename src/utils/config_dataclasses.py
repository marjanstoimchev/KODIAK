"""
Configuration Dataclasses

Structured configuration objects following Interface Segregation Principle.
Breaks down large parameter lists into focused, cohesive configuration objects.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Attributes:
        img_size: Input image size (square images)
        patch_size: Vision Transformer patch size
        embed_dim: Embedding dimension for ViT
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio for MLP hidden dimension
        num_storage_tokens: Number of storage tokens (register tokens)
    """
    img_size: int = 256  # DINOv3 official default
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_storage_tokens: int = 4

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by patch_size ({self.patch_size})"
            )
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")

    @property
    def num_patches(self) -> int:
        """Calculate number of patches."""
        return (self.img_size // self.patch_size) ** 2


@dataclass
class KodiakHeadConfig:
    """
    KODIAK model head configuration (prototypes, projector, decoder).

    Attributes:
        num_prototypes: Number of prototype vectors for clustering
        projector_dim: Dimension of projection head
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Number of decoder transformer blocks
        decoder_num_heads: Number of attention heads in decoder
    """
    num_prototypes: int = 4096
    projector_dim: int = 256
    decoder_embed_dim: int = 192
    decoder_depth: int = 4
    decoder_num_heads: int = 6

    def __post_init__(self):
        """Validate configuration."""
        if self.num_prototypes <= 0:
            raise ValueError(f"num_prototypes must be positive, got {self.num_prototypes}")
        if self.projector_dim <= 0:
            raise ValueError(f"projector_dim must be positive, got {self.projector_dim}")


@dataclass
class LossConfig:
    """
    Loss function configuration.

    Attributes:
        teacher_temp: Temperature for teacher softmax in optimal transport
        center_momentum: Momentum for centering operation
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations
        koleo_loss_weight: Weight for KoLeo loss (collapse prevention)
        koleo_distributed: Whether to use distributed KoLeo
        prototype_cls_loss_weight: Weight for multi-crop prototype CLS loss
        prototype_cls_student_temp: Student temperature for prototype CLS loss
        prototype_cls_teacher_temp: Teacher temperature for prototype CLS loss
        n_local_crops: Number of local crops (for multi-crop training)
    """
    teacher_temp: float = 0.04
    center_momentum: float = 0.9
    sinkhorn_iters: int = 3
    # KoLeo loss parameters
    koleo_loss_weight: float = 0.1
    koleo_distributed: bool = False
    # Multi-Crop Prototype CLS Loss
    prototype_cls_loss_weight: float = 1.0
    prototype_cls_student_temp: float = 0.1
    prototype_cls_teacher_temp: float = 0.04
    n_local_crops: int = 8  # DINOv3 default

    def __post_init__(self):
        """Validate configuration."""
        if self.teacher_temp <= 0:
            raise ValueError(f"teacher_temp must be positive, got {self.teacher_temp}")
        if not 0 <= self.center_momentum <= 1:
            raise ValueError(f"center_momentum must be in [0, 1], got {self.center_momentum}")
        if self.sinkhorn_iters <= 0:
            raise ValueError(f"sinkhorn_iters must be positive, got {self.sinkhorn_iters}")
        if self.koleo_loss_weight < 0:
            raise ValueError(f"koleo_loss_weight must be non-negative, got {self.koleo_loss_weight}")
        if self.prototype_cls_loss_weight < 0:
            raise ValueError(f"prototype_cls_loss_weight must be non-negative, got {self.prototype_cls_loss_weight}")
        if self.prototype_cls_student_temp <= 0:
            raise ValueError(f"prototype_cls_student_temp must be positive, got {self.prototype_cls_student_temp}")
        if self.prototype_cls_teacher_temp <= 0:
            raise ValueError(f"prototype_cls_teacher_temp must be positive, got {self.prototype_cls_teacher_temp}")
        if self.n_local_crops < 0:
            raise ValueError(f"n_local_crops must be non-negative, got {self.n_local_crops}")


@dataclass
class TrainingConfig:
    """
    Training hyperparameters (DINOv3-aligned).

    Attributes:
        lr: Base learning rate (DINOv3 default: 1e-3)
        min_lr: Minimum learning rate at end of cosine decay (DINOv3 default: 1e-6)
        weight_decay: Initial weight decay for AdamW (DINOv3 default: 0.04)
        weight_decay_end: Final weight decay after cosine schedule (DINOv3 default: 0.4)
        warmup_epochs: Number of warmup epochs (DINOv3 default: 10)
        freeze_last_layer_epochs: Epochs to freeze last layer at start (DINOv3 default: 1)
        max_epochs: Maximum number of training epochs
        freeze_backbone: Whether to freeze backbone weights
        backbone_lr_scale: Learning rate scale for backbone (relative to head)
        ema_momentum: Initial EMA momentum for teacher (DINOv3 default: 0.992)
        ema_momentum_end: Final EMA momentum (DINOv3 default: 1.0)
        layerwise_decay: Layerwise LR decay rate (DINOv3 default: 0.9)
        patch_embed_lr_mult: LR multiplier for patch embedding (DINOv3 default: 0.2)
        lr_scaling: LR scaling rule ('none', 'sqrt_wrt_1024')
    """
    lr: float = 1e-3  # DINOv3 default
    min_lr: float = 1e-6  # DINOv3 default
    weight_decay: float = 0.04  # DINOv3 default
    weight_decay_end: float = 0.4  # DINOv3 cosine schedule endpoint
    warmup_epochs: int = 10  # DINOv3 default
    freeze_last_layer_epochs: int = 1  # DINOv3 default (freeze at start)
    max_epochs: int = 100
    freeze_backbone: bool = False
    backbone_lr_scale: float = 1.0  # 1.0 for pretraining, 0.1 for finetuning
    ema_momentum: float = 0.992  # DINOv3 default start value
    ema_momentum_end: float = 1.0  # DINOv3 default end value
    layerwise_decay: float = 0.9  # DINOv3 default
    patch_embed_lr_mult: float = 0.2  # DINOv3 default
    lr_scaling: str = "sqrt_wrt_1024"  # DINOv3 scaling rule

    def __post_init__(self):
        """Validate configuration."""
        if self.lr <= 0 or self.lr >= 1:
            raise ValueError(f"lr must be in (0, 1), got {self.lr}")
        if self.min_lr < 0 or self.min_lr >= self.lr:
            raise ValueError(f"min_lr must be in [0, lr), got {self.min_lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.weight_decay_end < 0:
            raise ValueError(f"weight_decay_end must be non-negative, got {self.weight_decay_end}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {self.warmup_epochs}")
        if self.freeze_last_layer_epochs < 0:
            raise ValueError(f"freeze_last_layer_epochs must be non-negative, got {self.freeze_last_layer_epochs}")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.backbone_lr_scale <= 0 or self.backbone_lr_scale > 1:
            raise ValueError(f"backbone_lr_scale must be in (0, 1], got {self.backbone_lr_scale}")
        if not 0 <= self.ema_momentum <= 1:
            raise ValueError(f"ema_momentum must be in [0, 1], got {self.ema_momentum}")
        if not 0 <= self.ema_momentum_end <= 1:
            raise ValueError(f"ema_momentum_end must be in [0, 1], got {self.ema_momentum_end}")
        if not 0 < self.layerwise_decay <= 1:
            raise ValueError(f"layerwise_decay must be in (0, 1], got {self.layerwise_decay}")
        if self.patch_embed_lr_mult <= 0:
            raise ValueError(f"patch_embed_lr_mult must be positive, got {self.patch_embed_lr_mult}")
        if self.lr_scaling not in ("none", "sqrt_wrt_1024"):
            raise ValueError(f"lr_scaling must be 'none' or 'sqrt_wrt_1024', got {self.lr_scaling}")


@dataclass
class DataConfig:
    """
    Data loading and augmentation configuration.

    Attributes:
        dataset_type: Type of dataset ('pancreatic', 'huggingface', etc.)
        batch_size: Batch size for training
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        multi_crop: Whether to use multi-crop training (for MultiCropPrototypeCLSLoss)
    """
    dataset_type: str = "pancreatic"
    batch_size: int = 128
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    train_split: float = 0.9
    val_split: float = 0.1
    # Multi-crop pretraining (for MultiCropPrototypeCLSLoss)
    multi_crop: bool = False

    # Dataset-specific parameters
    root_dir: Optional[str] = None
    csv_path: Optional[str] = None
    magnification: Optional[str] = None
    root_path: Optional[str] = None
    hf_dataset_name: Optional[str] = None
    hf_split: Optional[str] = None
    hf_cache_dir: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        if not 0 < self.train_split <= 1:
            raise ValueError(f"train_split must be in (0, 1], got {self.train_split}")
        if not 0 <= self.val_split < 1:
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")
        if self.train_split + self.val_split > 1:
            raise ValueError(
                f"train_split + val_split must be <= 1, got {self.train_split + self.val_split}"
            )

        # Validate dataset-specific requirements
        if self.dataset_type == "pancreatic":
            if self.root_dir is None and self.csv_path is None:
                raise ValueError("root_dir or csv_path is required when dataset_type='pancreatic'")
            if self.root_dir and not Path(self.root_dir).is_dir():
                raise FileNotFoundError(f"root_dir does not exist: {self.root_dir}")
            if self.csv_path and not self.root_dir and not Path(self.csv_path).exists():
                raise FileNotFoundError(f"csv_path does not exist: {self.csv_path}")

        elif self.dataset_type == "huggingface":
            if self.hf_dataset_name is None:
                raise ValueError(
                    "hf_dataset_name is required when dataset_type='huggingface'\n"
                    "Example: hf_dataset_name='timm/oxford-iiit-pet'"
                )


@dataclass
class AugmentationConfig:
    """
    Image augmentation configuration.

    Attributes:
        global_crops_size: Size of global crops
        global_crops_scale: Scale range for random resized crop
        horizontal_flips: Whether to apply random horizontal flips
        normalize_mean: Mean for normalization
        normalize_std: Standard deviation for normalization
        # Multi-crop settings (for MultiCropPrototypeCLSLoss)
        local_crops_number: Number of local crops (DINOv3 default: 8)
        local_crops_size: Size of local crops (DINOv3 default: 96)
        local_crops_scale: Scale range for local crops (DINOv3 default: 0.05-0.4)
    """
    global_crops_size: int = 256  # DINOv3 official default
    global_crops_scale: Tuple[float, float] = (0.32, 1.0)
    horizontal_flips: bool = True
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Multi-crop settings (for MultiCropPrototypeCLSLoss)
    local_crops_number: int = 8  # DINOv3 default
    local_crops_size: int = 112  # DINOv3 official default
    local_crops_scale: Tuple[float, float] = (0.05, 0.4)  # DINOv3 default

    def __post_init__(self):
        """Validate configuration."""
        if self.global_crops_size <= 0:
            raise ValueError(f"global_crops_size must be positive, got {self.global_crops_size}")
        if len(self.global_crops_scale) != 2:
            raise ValueError("global_crops_scale must be a tuple of 2 floats")
        if not (0 < self.global_crops_scale[0] <= self.global_crops_scale[1] <= 1):
            raise ValueError(
                f"global_crops_scale must satisfy 0 < min <= max <= 1, got {self.global_crops_scale}"
            )
        # Validate local crop config
        if self.local_crops_number < 0:
            raise ValueError(f"local_crops_number must be non-negative, got {self.local_crops_number}")
        if self.local_crops_size <= 0:
            raise ValueError(f"local_crops_size must be positive, got {self.local_crops_size}")
        if len(self.local_crops_scale) != 2:
            raise ValueError("local_crops_scale must be a tuple of 2 floats")
        if not (0 < self.local_crops_scale[0] <= self.local_crops_scale[1] <= 1):
            raise ValueError(
                f"local_crops_scale must satisfy 0 < min <= max <= 1, got {self.local_crops_scale}"
            )


@dataclass
class MaskingConfig:
    """
    Masking strategy configuration.

    Attributes:
        mask_ratio_tuple: Range of mask ratios (min, max)
        mask_probability: Probability of masking a sample in the batch
        num_masks: Number of different mask views to generate
    """
    mask_ratio_tuple: Tuple[float, float] = (0.1, 0.5)
    mask_probability: float = 0.5
    num_masks: int = 2

    def __post_init__(self):
        """Validate configuration."""
        if len(self.mask_ratio_tuple) != 2:
            raise ValueError("mask_ratio_tuple must be a tuple of 2 floats")
        if not (0 <= self.mask_ratio_tuple[0] <= self.mask_ratio_tuple[1] <= 1):
            raise ValueError(
                f"mask_ratio_tuple must satisfy 0 <= min <= max <= 1, got {self.mask_ratio_tuple}"
            )
        if not 0 <= self.mask_probability <= 1:
            raise ValueError(f"mask_probability must be in [0, 1], got {self.mask_probability}")
        if self.num_masks <= 0:
            raise ValueError(f"num_masks must be positive, got {self.num_masks}")


@dataclass
class ClassificationConfig:
    """
    Classification-specific configuration.

    Attributes:
        num_classes: Number of output classes
        use_cls_token: Whether to use CLS token for classification
        label_smoothing: Label smoothing factor for cross-entropy loss
    """
    num_classes: int = 10
    use_cls_token: bool = True
    label_smoothing: float = 0.0

    def __post_init__(self):
        """Validate configuration."""
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")


# Helper functions for converting configs
def config_to_dict(config) -> dict:
    """Convert dataclass config to dictionary."""
    return asdict(config)


def merge_configs(*configs) -> dict:
    """Merge multiple config dataclasses into a single dictionary."""
    result = {}
    for config in configs:
        result.update(asdict(config))
    return result
