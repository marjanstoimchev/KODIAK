# models/classifier.py

"""
Linear Classifier for downstream classification tasks.
Loads pretrained encoder from SSL checkpoint and adds classification head.

Supports:
- KODIAK checkpoints (teacher_encoder, student_encoder prefixes)
- DINOv3 SSL checkpoints (ssl_model.teacher.backbone prefixes)
- Official DINOv3 checkpoints (direct keys)
- Auto-detection of num_storage_tokens and mask_k_bias from checkpoint
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import MAEStyleDinoV3Encoder


class LinearClassifier(nn.Module):
    """
    Linear classifier on top of pretrained encoder.
    Can load pretrained weights from KODIAK/DINOv3 model and optionally freeze the backbone.

    Args:
        num_classes: Number of output classes
        num_storage_tokens: Number of storage/register tokens. If None, auto-detect from checkpoint.
        mask_k_bias: Whether to use mask_k_bias in attention. If None, auto-detect from checkpoint.
        encoder_type: Which encoder to load from SSL checkpoint.
                     "teacher" (default, recommended) - EMA encoder, more stable
                     "student" - trained encoder
    """
    def __init__(self, num_classes, img_size=256, patch_size=16, in_chans=3, embed_dim=384,
                 depth=12, num_heads=6, mlp_ratio=4.0, num_storage_tokens=None,
                 rope_theta=100.0, drop_path_rate=0.0, init_values=1e-5,
                 pretrained_path=None, use_cls_token=True, encoder_type="teacher",
                 mask_k_bias=None, concat_cls_patch=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.concat_cls_patch = concat_cls_patch
        self.embed_dim = embed_dim
        self.encoder_type = encoder_type

        # Auto-detect architecture from checkpoint if not specified
        detected_storage_tokens = num_storage_tokens
        detected_mask_k_bias = mask_k_bias

        if pretrained_path and (num_storage_tokens is None or mask_k_bias is None):
            detected_storage_tokens, detected_mask_k_bias = self._detect_architecture(
                pretrained_path, encoder_type, num_storage_tokens, mask_k_bias
            )

        # Use defaults if still None (DINOv3 official defaults)
        if detected_storage_tokens is None:
            detected_storage_tokens = 4
        if detected_mask_k_bias is None:
            detected_mask_k_bias = True

        self._num_storage_tokens = detected_storage_tokens
        self._mask_k_bias = detected_mask_k_bias

        # Encoder (same architecture as KODIAK)
        self.encoder = MAEStyleDinoV3Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_storage_tokens=detected_storage_tokens,
            rope_theta=rope_theta,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            mask_k_bias=detected_mask_k_bias,
        )

        # Classification head
        head_dim = embed_dim * 2 if concat_cls_patch else embed_dim
        self.head = nn.Linear(head_dim, num_classes)

        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        self._init_head()

    def _detect_architecture(self, path, encoder_type, num_storage_tokens, mask_k_bias):
        """Auto-detect architecture settings from checkpoint."""
        print(f"Auto-detecting architecture from checkpoint: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        # Determine prefix based on checkpoint format
        # Supports: KODIAK, DINOv3 SSL, MAE, I-JEPA, MoCa, official DINOv3
        if encoder_type == "teacher":
            prefixes = [
                # KODIAK format
                "model.teacher_encoder._orig_mod.", "model.teacher_encoder.",
                "teacher_encoder._orig_mod.", "teacher_encoder.",
                # DINOv3 SSL format
                "ssl_model.teacher.backbone._orig_mod.", "ssl_model.teacher.backbone.",
                # I-JEPA format (target_encoder = teacher/EMA)
                "model.target_encoder._orig_mod.", "model.target_encoder.",
                # MoCa format (encoder_teacher.dinov3 = teacher backbone)
                "model.encoder_teacher.dinov3._orig_mod.", "model.encoder_teacher.dinov3.",
                # MAE format (single encoder, no teacher/student split)
                "model.encoder._orig_mod.", "model.encoder.",
                # Direct keys (official checkpoint)
                ""
            ]
        else:
            prefixes = [
                # KODIAK format
                "model.student_encoder._orig_mod.", "model.student_encoder.",
                "student_encoder._orig_mod.", "student_encoder.",
                # DINOv3 SSL format
                "ssl_model.student.backbone._orig_mod.", "ssl_model.student.backbone.",
                # I-JEPA format (context_encoder = student)
                "model.context_encoder._orig_mod.", "model.context_encoder.",
                # MoCa format (encoder.dinov3 = student backbone)
                "model.encoder.dinov3._orig_mod.", "model.encoder.dinov3.",
                # MAE format (single encoder)
                "model.encoder._orig_mod.", "model.encoder.",
                # Direct keys
                ""
            ]

        detected_storage = num_storage_tokens
        detected_mask_bias = mask_k_bias

        # Try to find storage_tokens
        if detected_storage is None:
            for prefix in prefixes:
                key = f"{prefix}storage_tokens"
                if key in state:
                    detected_storage = state[key].shape[1]  # Shape is [1, n_tokens, embed_dim]
                    print(f"  Detected num_storage_tokens={detected_storage} from checkpoint")
                    break
            if detected_storage is None:
                # No storage_tokens found, assume 0
                detected_storage = 0
                print(f"  No storage_tokens found, using num_storage_tokens=0")

        # Try to find qkv.bias_mask to detect mask_k_bias
        if detected_mask_bias is None:
            for prefix in prefixes:
                key = f"{prefix}blocks.0.attn.qkv.bias_mask"
                if key in state:
                    detected_mask_bias = True
                    print(f"  Detected mask_k_bias=True from checkpoint")
                    break
            if detected_mask_bias is None:
                # No bias_mask found, assume False
                detected_mask_bias = False
                print(f"  No qkv.bias_mask found, using mask_k_bias=False")

        return detected_storage, detected_mask_bias

    def _init_head(self):
        """Initialize classification head"""
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _load_pretrained(self, path):
        """Load pretrained weights from SSL model checkpoint.

        Supports multiple checkpoint formats:
        - KODIAK checkpoints: teacher_encoder, student_encoder prefixes
        - DINOv3 SSL checkpoints: ssl_model.teacher.backbone, ssl_model.student.backbone prefixes
        - I-JEPA checkpoints: target_encoder (teacher), context_encoder (student)
        - MoCa checkpoints: encoder_teacher.dinov3 (teacher), encoder.dinov3 (student)
        - MAE checkpoints: model.encoder (single encoder, no teacher/student)
        - Official DINOv3 checkpoints: direct keys (cls_token, patch_embed.proj.weight, etc.)

        Uses self.encoder_type to determine which encoder to load:
        - "teacher" (default): Load teacher/EMA encoder (recommended for evaluation)
        - "student": Load student encoder

        Handles torch.compile wrapped models (with ._orig_mod. prefix).
        """
        print(f"Loading pretrained weights from {path}")
        print(f"Encoder type: {self.encoder_type}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        # Keys to exclude from backbone (heads only)
        exclude_prefixes = ['dino_head.', 'ibot_head.', 'gram_head.', 'prototype_predictor.', 'decoder.']

        def is_backbone_key(key):
            """Check if key is a backbone weight (not head)"""
            for prefix in exclude_prefixes:
                if key.startswith(prefix):
                    return False
            return True

        # Define prefix search order: (prefix_variants, label)
        # Each entry is tried in order; first match wins.
        # More specific prefixes must come before less specific ones
        # (e.g. "model.encoder_teacher.dinov3." before "model.encoder.").
        if self.encoder_type == "teacher":
            prefix_groups = [
                # KODIAK: teacher_encoder
                (["model.teacher_encoder._orig_mod.", "model.teacher_encoder.",
                  "teacher_encoder._orig_mod.", "teacher_encoder."],
                 "KODIAK teacher_encoder"),
                # DINOv3 SSL: ssl_model.teacher.backbone
                (["ssl_model.teacher.backbone._orig_mod.", "ssl_model.teacher.backbone."],
                 "DINOv3 ssl_model.teacher.backbone"),
                # I-JEPA: target_encoder (EMA / teacher)
                (["model.target_encoder._orig_mod.", "model.target_encoder."],
                 "I-JEPA target_encoder"),
                # MoCa: encoder_teacher.dinov3 (must be before model.encoder)
                (["model.encoder_teacher.dinov3._orig_mod.", "model.encoder_teacher.dinov3."],
                 "MoCa encoder_teacher.dinov3"),
                # MAE: model.encoder (single encoder, no teacher/student)
                (["model.encoder._orig_mod.", "model.encoder."],
                 "MAE model.encoder"),
                # KODIAK fallback: student_encoder
                (["model.student_encoder._orig_mod.", "model.student_encoder.",
                  "student_encoder._orig_mod.", "student_encoder."],
                 "KODIAK student_encoder (fallback)"),
                # DINOv3 SSL fallback: student backbone
                (["ssl_model.student.backbone._orig_mod.", "ssl_model.student.backbone."],
                 "DINOv3 ssl_model.student.backbone (fallback)"),
            ]
        else:  # student
            prefix_groups = [
                # KODIAK: student_encoder
                (["model.student_encoder._orig_mod.", "model.student_encoder.",
                  "student_encoder._orig_mod.", "student_encoder."],
                 "KODIAK student_encoder"),
                # DINOv3 SSL: ssl_model.student.backbone
                (["ssl_model.student.backbone._orig_mod.", "ssl_model.student.backbone."],
                 "DINOv3 ssl_model.student.backbone"),
                # I-JEPA: context_encoder (student)
                (["model.context_encoder._orig_mod.", "model.context_encoder."],
                 "I-JEPA context_encoder"),
                # MoCa: encoder.dinov3 (must be before model.encoder)
                (["model.encoder.dinov3._orig_mod.", "model.encoder.dinov3."],
                 "MoCa encoder.dinov3"),
                # MAE: model.encoder (single encoder)
                (["model.encoder._orig_mod.", "model.encoder."],
                 "MAE model.encoder"),
                # KODIAK fallback: teacher_encoder
                (["model.teacher_encoder._orig_mod.", "model.teacher_encoder.",
                  "teacher_encoder._orig_mod.", "teacher_encoder."],
                 "KODIAK teacher_encoder (fallback)"),
                # DINOv3 SSL fallback: teacher backbone
                (["ssl_model.teacher.backbone._orig_mod.", "ssl_model.teacher.backbone."],
                 "DINOv3 ssl_model.teacher.backbone (fallback)"),
            ]

        # Try each prefix group in order
        encoder_state = {}
        matched_label = None

        for prefixes, label in prefix_groups:
            for key, value in state.items():
                for prefix in prefixes:
                    if key.startswith(prefix):
                        new_key = key[len(prefix):]
                        if is_backbone_key(new_key):
                            encoder_state[new_key] = value
                        break
            if encoder_state:
                matched_label = label
                break

        # Try official DINOv3 checkpoint format (direct keys) as last resort
        if not encoder_state:
            if 'cls_token' in state or 'patch_embed.proj.weight' in state:
                print("Detected official DINOv3 checkpoint format (direct keys)")
                for key, value in state.items():
                    if is_backbone_key(key):
                        encoder_state[key] = value
                matched_label = "official DINOv3 (direct keys)"

        # Load encoder weights
        if encoder_state:
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=True)
            print(f"Loaded {matched_label} weights")
            print(f"Loaded {len(encoder_state)} encoder weights")
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if missing:
                print(f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        else:
            print("WARNING: No encoder weights found in checkpoint!")
            print(f"  Checkpoint keys sample: {list(state.keys())[:5]}")
            print("  The encoder will be RANDOMLY INITIALIZED.")

    def freeze_backbone(self):
        """Freeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, H, W)
        Returns:
            logits: Classification logits (B, num_classes)
        """
        # Get features from encoder
        cls_token, patch_tokens, _ = self.encoder.forward_full(x, return_attention=False)

        # Feature extraction mode
        if self.concat_cls_patch:
            features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)  # (B, embed_dim * 2)
        elif self.use_cls_token:
            features = cls_token  # (B, embed_dim)
        else:
            features = patch_tokens.mean(dim=1)  # (B, embed_dim)

        # Classification head
        logits = self.head(features)
        return logits

    def get_features(self, x: torch.Tensor, return_attention: bool = False):
        """Extract features without classification head"""
        if return_attention:
            cls_token, patch_tokens, attn_maps = self.encoder.forward_full(x, return_attention=True)
            return cls_token, patch_tokens, attn_maps
        else:
            cls_token, patch_tokens, _ = self.encoder.forward_full(x, return_attention=False)
            return cls_token, patch_tokens


def create_linear_classifier(num_classes, img_size=256, pretrained_path=None, use_cls_token=True,
                             encoder_type="teacher", concat_cls_patch=False):
    """Create a linear classifier with ViT-S/16 backbone.

    Args:
        num_classes: Number of output classes
        img_size: Input image size
        pretrained_path: Path to pretrained SSL checkpoint
        use_cls_token: Whether to use CLS token for classification (vs mean pooling)
        encoder_type: Which encoder to load - "teacher" (default, recommended) or "student"
        concat_cls_patch: Whether to concatenate CLS token with mean patch tokens (2x embed_dim)
    """
    return LinearClassifier(
        num_classes=num_classes,
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_storage_tokens=None,  # Auto-detect from checkpoint
        pretrained_path=pretrained_path,
        use_cls_token=use_cls_token,
        encoder_type=encoder_type,
        concat_cls_patch=concat_cls_patch,
    )
