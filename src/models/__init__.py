# =============================================================================
# KODIAK Models Package
# =============================================================================
# Model architectures for KODIAK:
# - components: Encoders, decoders, projectors, prototype layer
#               (encoder/decoder components borrowed from DINOv3)
# - kodiak: Main KODIAK model for self-supervised pretraining
# - classifier: Linear classifier for downstream tasks
# =============================================================================

from .components import (
    MLP,
    TransformerDecoderBlock,
    PrototypeLayer,
    MAEStyleDinoV3Encoder,
    MAEStyleDecoder,
    DINOProjector,
)

from .kodiak import (
    Kodiak,
    MaskedPrototypePredictor,  # backward compatibility alias
    create_kodiak_small,
)

from .classifier import (
    LinearClassifier,
    create_linear_classifier,
)

__all__ = [
    # Components
    "MLP",
    "TransformerDecoderBlock",
    "PrototypeLayer",
    "MAEStyleDinoV3Encoder",
    "MAEStyleDecoder",
    "DINOProjector",
    # KODIAK
    "Kodiak",
    "MaskedPrototypePredictor",
    "create_kodiak_small",
    # Classifier
    "LinearClassifier",
    "create_linear_classifier",
]
