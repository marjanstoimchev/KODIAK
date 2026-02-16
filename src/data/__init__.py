# =============================================================================
# Data Package
# =============================================================================
# Modular data loading for KODIAK pretraining and classification.
#
# Structure:
#   data/
#   ├── pretraining/       # Self-supervised pretraining
#   │   ├── datamodule.py
#   │   ├── transforms.py
#   │   └── collate.py
#   ├── classification/    # Downstream classification
#   │   ├── datamodule.py
#   │   ├── transforms.py
#   │   └── collate.py
#   ├── utils/             # Shared utilities
#   │   ├── samplers.py
#   │   ├── masking.py
#   │   └── factory.py
#   └── datasets/          # Dataset implementations
#       ├── huggingface/
#       └── custom/
# =============================================================================

# Pretraining
from .pretraining import (
    PretrainingDataModule,
    DINOv3PretrainingTransform,
    MultiCropPretrainingTransform,
    DataAugmentationDINO,
    KodiakCollate,
    MultiCropKodiakCollate,
)

# Classification
from .classification import (
    ClassificationDataModule,
    ClassificationTrainTransform,
    ClassificationValTransform,
    classification_collate,
)

# Utils
from .utils import (
    SamplerType,
    EpochSampler,
    InfiniteSampler,
    ShardedInfiniteSampler,
    make_sampler,
    get_rank,
    get_world_size,
    MaskingGenerator,
    DatasetRegistry,
)

# Datasets
from .datasets import (
    HuggingFaceDataset,
    get_available_splits,
    CustomPatchDataset,
)

# Legacy aliases for backward compatibility
FlexibleDataModule = PretrainingDataModule

__all__ = [
    # Pretraining
    "PretrainingDataModule",
    "DINOv3PretrainingTransform",
    "MultiCropPretrainingTransform",
    "DataAugmentationDINO",
    "KodiakCollate",
    "MultiCropKodiakCollate",
    # Classification
    "ClassificationDataModule",
    "ClassificationTrainTransform",
    "ClassificationValTransform",
    "classification_collate",
    # Utils
    "SamplerType",
    "EpochSampler",
    "InfiniteSampler",
    "ShardedInfiniteSampler",
    "make_sampler",
    "get_rank",
    "get_world_size",
    "MaskingGenerator",
    "DatasetRegistry",
    # Datasets
    "HuggingFaceDataset",
    "get_available_splits",
    "CustomPatchDataset",
    # Legacy
    "FlexibleDataModule",
]
