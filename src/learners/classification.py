# learners/classification.py

"""
PyTorch Lightning module for fine-tuning the linear classifier.
Similar structure to pretraining.py but for supervised classification.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import Accuracy, AUROC, F1Score, Precision, Recall

from src.models import LinearClassifier


class ClassificationLearner(L.LightningModule):
    """
    PyTorch Lightning module for classification with dependency injection.

    This class follows the Dependency Inversion Principle - it depends on
    abstractions (nn.Module interfaces) rather than concrete implementations.

    Args:
        model: Classification model instance (dependency injection)
        criterion: Loss function (dependency injection)
        num_classes: Number of output classes
        lr: Learning rate
        weight_decay: Weight decay for AdamW optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Maximum training epochs
        freeze_backbone: Whether to freeze backbone parameters
        backbone_lr_scale: Learning rate multiplier for backbone
    """

    def __init__(
        self,
        # Required
        num_classes: int,
        # Dependency injection - SOLID principle
        model: Optional[torch.nn.Module] = None,
        criterion: Optional[torch.nn.Module] = None,
        # Training config
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        freeze_backbone: bool = False,
        backbone_lr_scale: float = 0.1,
        # Legacy support - will be deprecated
        img_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
        vit_depth: Optional[int] = None,
        vit_heads: Optional[int] = None,
        mlp_ratio: Optional[float] = None,
        num_storage_tokens: Optional[int] = None,
        pretrained_path: Optional[str] = None,
        use_cls_token: Optional[bool] = None,
        concat_cls_patch: bool = False,
        label_smoothing: Optional[float] = None,
        encoder_type: str = "teacher",  # "teacher" (default, recommended) or "student"
        mask_k_bias: Optional[bool] = None,  # Auto-detect from checkpoint if None
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion'])
        self.num_classes = num_classes

        # Dependency injection: Use provided instances or create defaults
        if model is None:
            # Legacy mode: Create model from individual parameters
            if img_size is None:
                raise ValueError(
                    "ERROR: Either provide 'model' instance or individual model parameters\n"
                    "\n"
                    "Option 1 (Recommended - Dependency Injection):\n"
                    "  model = LinearClassifier(num_classes=10, ...)\n"
                    "  learner = ClassificationLearner(num_classes=10, model=model)\n"
                    "\n"
                    "Option 2 (Legacy):\n"
                    "  learner = ClassificationLearner(num_classes=10, img_size=256, embed_dim=384, ...)\n"
                )

            model = LinearClassifier(
                num_classes=num_classes,
                img_size=img_size,
                patch_size=patch_size or 16,
                embed_dim=embed_dim,
                depth=vit_depth or 12,
                num_heads=vit_heads or 6,
                mlp_ratio=mlp_ratio or 4.0,
                num_storage_tokens=num_storage_tokens,  # None = auto-detect from checkpoint
                pretrained_path=pretrained_path,
                use_cls_token=use_cls_token if use_cls_token is not None else True,
                encoder_type=encoder_type,
                mask_k_bias=mask_k_bias,  # None = auto-detect from checkpoint
                concat_cls_patch=concat_cls_patch,
            )

        self.model = model

        # Optionally freeze backbone
        if freeze_backbone and hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone()

        # Loss function (dependency injection)
        if criterion is None:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing or 0.0)

        self.criterion = criterion

        # Metrics. The classifier always outputs `num_classes` logits (also for
        # 2 classes), so the multiclass metric family is used throughout.
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        task = "multiclass"
        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_acc = Accuracy(task=task, num_classes=num_classes)

        # Top-5 accuracy (only for multiclass with >5 classes)
        if num_classes > 5:
            self.train_acc_top5 = Accuracy(task=task, num_classes=num_classes, top_k=5)
            self.val_acc_top5 = Accuracy(task=task, num_classes=num_classes, top_k=5)
            self.test_acc_top5 = Accuracy(task=task, num_classes=num_classes, top_k=5)
        else:
            self.train_acc_top5 = None
            self.val_acc_top5 = None
            self.test_acc_top5 = None

        # Test metrics
        self.test_acc = Accuracy(task=task, num_classes=num_classes)

        # Macro averaged
        self.test_f1_macro = F1Score(task=task, num_classes=num_classes, average="macro")
        self.test_precision_macro = Precision(task=task, num_classes=num_classes, average="macro")
        self.test_recall_macro = Recall(task=task, num_classes=num_classes, average="macro")

        # Weighted averaged
        self.test_f1_weighted = F1Score(task=task, num_classes=num_classes, average="weighted")
        self.test_precision_weighted = Precision(task=task, num_classes=num_classes, average="weighted")
        self.test_recall_weighted = Recall(task=task, num_classes=num_classes, average="weighted")

        # AUROC (multiclass)
        self.test_auroc = AUROC(task=task, num_classes=num_classes, average="macro")

        self.backbone_lr_scale = backbone_lr_scale

    @classmethod
    def _detect_architecture_from_state_dict(cls, state_dict):
        """Auto-detect architecture settings from checkpoint state_dict."""
        detected_storage_tokens = None
        detected_mask_k_bias = None

        # Detect num_storage_tokens from state_dict
        for key in state_dict.keys():
            if 'storage_tokens' in key and key.endswith('storage_tokens'):
                detected_storage_tokens = state_dict[key].shape[1]
                print(f"  Auto-detected num_storage_tokens={detected_storage_tokens} from checkpoint state_dict")
                break

        if detected_storage_tokens is None:
            detected_storage_tokens = 4  # DINOv3 official default
            print("  No storage_tokens found in checkpoint, using num_storage_tokens=4")

        # Detect mask_k_bias from state_dict
        for key in state_dict.keys():
            if 'qkv.bias_mask' in key:
                detected_mask_k_bias = True
                print("  Auto-detected mask_k_bias=True from checkpoint state_dict")
                break

        if detected_mask_k_bias is None:
            detected_mask_k_bias = True  # DINOv3 official default

        return detected_storage_tokens, detected_mask_k_bias

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_overrides=None,
        strict=True,
        load_best_only=True,
        **kwargs,
    ):
        """
        Load ClassificationLearner from checkpoint.

        Overrides Lightning's load_from_checkpoint to handle classification-specific logic:
        - For classification checkpoints, skip pretrained_path loading since encoder weights
          are already baked in the checkpoint
        - Auto-detect architecture (num_storage_tokens, mask_k_bias) from checkpoint state_dict

        Args:
            checkpoint_path: Path to Lightning checkpoint
            map_location: Device mapping
            hparams_overrides: Dict of hparams to override
            strict: Strict checkpoint loading
            load_best_only: Load best checkpoint (Lightning default)
            **kwargs: Additional args for load_from_checkpoint (e.g., encoder_type)
        """
        import torch

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Get hparams from checkpoint
        hparams = checkpoint.get('hyper_parameters', {}).copy()

        # For classification checkpoints: don't try to load pretrained path
        # The encoder is already trained and in the checkpoint
        if 'pretrained_path' in hparams:
            pretrained_path = hparams['pretrained_path']
            # Check if this looks like a pretraining path
            if pretrained_path and 'pretraining' in str(pretrained_path):
                print("Classification checkpoint detected (skipping pretrained_path)")
                # Remove pretrained_path from hparams so model doesn't try to load it
                hparams['pretrained_path'] = None

        # Apply any overrides
        if hparams_overrides:
            hparams.update(hparams_overrides)

        # Apply kwargs overrides (e.g., encoder_type from caller)
        hparams.update(kwargs)

        # Auto-detect architecture from checkpoint state_dict
        state_dict = checkpoint.get('state_dict', {})
        detected_storage_tokens, detected_mask_k_bias = cls._detect_architecture_from_state_dict(state_dict)

        # Check if we have the required parameters for legacy mode
        # If the checkpoint doesn't have the model instance and lacks required legacy params,
        # we need to provide defaults to avoid the ValueError
        if 'model' not in hparams or hparams['model'] is None:
            # Check if we have the required legacy parameters
            required_legacy_params = ['img_size', 'embed_dim']
            missing_legacy_params = [param for param in required_legacy_params if param not in hparams or hparams[param] is None]

            if missing_legacy_params:
                # Provide default values for missing legacy parameters
                defaults = {
                    'img_size': 256,  # DINOv3 official default
                    'patch_size': 16,
                    'embed_dim': 384,
                    'vit_depth': 12,
                    'vit_heads': 6,
                    'mlp_ratio': 4.0,
                    'use_cls_token': True,
                    'concat_cls_patch': False,
                    'encoder_type': 'teacher',
                }

                # Apply defaults only if not already present in hparams
                for param, default_value in defaults.items():
                    if param not in hparams or hparams[param] is None:
                        hparams[param] = default_value

                print(f"Applied default values for missing legacy parameters: {missing_legacy_params}")

            # Apply detected architecture params (override any defaults)
            hparams['num_storage_tokens'] = detected_storage_tokens
            hparams['mask_k_bias'] = detected_mask_k_bias

        # Create instance with modified hparams (no pretrained loading will happen)
        model = cls(**hparams)

        # Load state dict directly
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=strict)

        return model

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, is_train: bool):
        # Handle both dictionary format (with string keys) and tuple format (image, label)
        # EXACT match to DinoV3LightningTraining
        if isinstance(batch, dict):
            # Dictionary format: {'images': ..., 'labels': ...}
            images = batch["images"]
            labels = batch["labels"]
        else:
            # Tuple format: (images, labels) - common PyTorch dataset format
            images, labels = batch

        # Optimization: Channels Last (if on CUDA)
        if images.device.type == 'cuda':
            images = images.contiguous(memory_format=torch.channels_last)

        # Forward pass
        logits = self.model(images)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)

        return loss, logits, preds, labels

    def training_step(self, batch, batch_idx):
        loss, logits, preds, labels = self.shared_step(batch, is_train=True)

        # Update metrics
        self.train_acc(preds, labels)

        # Log
        log_dict = {
            "train_loss": loss,
            "train_acc": self.train_acc,
        }

        if self.train_acc_top5 is not None and logits.dim() >= 2 and logits.size(0) > 1:
            self.train_acc_top5(logits, labels)
            log_dict["train_acc_top5"] = self.train_acc_top5

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, preds, labels = self.shared_step(batch, is_train=False)

        # Update metrics
        self.val_acc(preds, labels)

        # Log
        log_dict = {
            "val_loss": loss,
            "val_acc": self.val_acc,
        }

        if self.val_acc_top5 is not None and logits.dim() >= 2 and logits.size(0) > 1:
            self.val_acc_top5(logits, labels)
            log_dict["val_acc_top5"] = self.val_acc_top5

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        loss, logits, preds, labels = self.shared_step(batch, is_train=False)
        probs = torch.softmax(logits, dim=1)

        # Update all test metrics
        self.test_acc(preds, labels)

        # Macro metrics
        self.test_f1_macro(preds, labels)
        self.test_precision_macro(preds, labels)
        self.test_recall_macro(preds, labels)

        # Weighted metrics
        self.test_f1_weighted(preds, labels)
        self.test_precision_weighted(preds, labels)
        self.test_recall_weighted(preds, labels)

        # AUROC (needs probabilities)
        self.test_auroc(probs, labels)

        # Test metrics (epoch-level with sync)
        log_dict = {
            "test_loss": loss,
            "test_acc": self.test_acc,
            "test_f1_macro": self.test_f1_macro,
            "test_precision_macro": self.test_precision_macro,
            "test_recall_macro": self.test_recall_macro,
            "test_f1_weighted": self.test_f1_weighted,
            "test_precision_weighted": self.test_precision_weighted,
            "test_recall_weighted": self.test_recall_weighted,
            "test_auroc": self.test_auroc,
        }

        # Top-5 accuracy if available
        if self.test_acc_top5 is not None and logits.dim() >= 2 and logits.size(0) > 1:
            self.test_acc_top5(logits, labels)
            log_dict["test_acc_top5"] = self.test_acc_top5

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

    def configure_optimizers(self):
        # Separate backbone and head parameters for different learning rates
        backbone_params = []
        head_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder" in name or "backbone" in name or "student.backbone" in name or "teacher.backbone" in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

        # Use fused optimizer on GPU (faster)
        use_fused = self.device.type == 'cuda'

        lr = self.hparams.get('lr', 1e-3)
        weight_decay = self.hparams.get('weight_decay', 0.05)
        warmup_epochs = self.hparams.get('warmup_epochs', 5)
        max_epochs = self.hparams.get('max_epochs', 100)

        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": lr * self.backbone_lr_scale
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": lr
            })

        optimizer = AdamW(
            param_groups,
            weight_decay=weight_decay,
            fused=use_fused if use_fused else False  # Explicitly set fused to False if not supported
        )

        # Cosine annealing with warmup
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(warmup_epochs * (total_steps / max(max_epochs, 1)))

        def lr_lambda(step):
            if total_steps <= 0 or total_steps <= warmup_steps:
                return 1.0
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def create_classification_learner(
    num_classes: int,
    pretrained_path: Optional[str] = None,
    freeze_backbone: bool = False,
    encoder_type: str = "teacher",
    **kwargs
) -> ClassificationLearner:
    """Factory function to create a classification learner.

    Args:
        num_classes: Number of output classes
        pretrained_path: Path to pretrained SSL checkpoint
        freeze_backbone: Whether to freeze backbone (linear eval mode)
        encoder_type: Which encoder to load - "teacher" (default, recommended) or "student"
    """
    defaults = dict(
        num_classes=num_classes,
        img_size=256,  # DINOv3 official default
        patch_size=16,
        embed_dim=384,
        vit_depth=12,
        vit_heads=6,
        mlp_ratio=4.0,
        num_storage_tokens=None,  # Auto-detect from checkpoint
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone,
        backbone_lr_scale=0.1,
        encoder_type=encoder_type,
    )
    defaults.update(kwargs)
    return ClassificationLearner(**defaults)
