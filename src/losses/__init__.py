"""
Loss Functions

- MaskLoss: Masked prototype prediction loss with optimal transport
- KoLeoLoss: Kozachenko-Leonenko entropic loss for collapse prevention
- MultiCropPrototypeCLSLoss: Multi-crop CLS loss in prototype space
"""

from .losses import (
    MaskLoss,
    MPPLoss,  # backward compatibility alias
    KoLeoLoss,
    KoLeoLossDistributed,
)

from .multicrop_prototype_cls_loss import (
    MultiCropPrototypeCLSLoss,
)

__all__ = [
    "MaskLoss",
    "MPPLoss",
    "KoLeoLoss",
    "KoLeoLossDistributed",
    "MultiCropPrototypeCLSLoss",
]
