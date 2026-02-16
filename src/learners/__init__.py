# learners/__init__.py

"""
PyTorch Lightning Learners Package

This package contains PyTorch Lightning modules for training various models:
- pretraining: Self-supervised pretraining with KODIAK + Multi-Crop Prototype CLS Loss
- classification: Supervised classification fine-tuning
"""

from .pretraining import (
    MotifLearner,
    create_motif_learner_dinov3,
)

from .classification import (
    ClassificationLearner,
    create_classification_learner,
)

__all__ = [
    "MotifLearner",
    "create_motif_learner_dinov3",
    "ClassificationLearner",
    "create_classification_learner",
]
