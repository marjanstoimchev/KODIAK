# =============================================================================
# Dataset Implementations
# =============================================================================
# Specific dataset implementations: HuggingFace, tissue histology, etc.
# =============================================================================

from .huggingface.dataset import HuggingFaceDataset, get_available_splits
from .tissue.dataset import TissuePatchDataset

# Alias for consistency
TissueDataset = TissuePatchDataset

__all__ = [
    "HuggingFaceDataset",
    "get_available_splits",
    "TissuePatchDataset",
    "TissueDataset",
]
