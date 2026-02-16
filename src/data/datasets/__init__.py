# =============================================================================
# Dataset Implementations
# =============================================================================
# Specific dataset implementations: HuggingFace, custom folder-based, etc.
# =============================================================================

from .huggingface.dataset import HuggingFaceDataset, get_available_splits
from .custom.dataset import CustomPatchDataset

__all__ = [
    "HuggingFaceDataset",
    "get_available_splits",
    "CustomPatchDataset",
]
