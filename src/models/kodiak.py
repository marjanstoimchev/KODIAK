# models/kodiak.py

"""
KODIAK - Main Model

Self-supervised learning with masked prototype prediction.
Encoder and decoder components are borrowed from the DINOv3 architecture.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    MAEStyleDinoV3Encoder,
    MAEStyleDecoder,
    DINOProjector,
    PrototypeLayer,
)


class Kodiak(nn.Module):
    """
    KODIAK model with dynamic RoPE.

    Supports arbitrary input sizes (not just the training resolution) thanks to:
    - Dynamic RoPE position encoding in encoders
    - Interpolated position embeddings in decoder

    This enables:
    - Cross-scale training with local (112x112) + global (256x256) crops
    - Inference at arbitrary resolutions
    - Multi-resolution training
    """
    def __init__(self, num_prototypes=4096, projector_dim=256, img_size=256, patch_size=16,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, num_storage_tokens=4,
                 decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=6,
                 drop_path_rate=0.0, pretrained_path=None):
        super().__init__()
        self.patch_size = patch_size
        # Default grid size (actual computed dynamically from input)
        self.default_grid_size = img_size // patch_size
        self.default_num_patches = self.default_grid_size ** 2
        self.num_prototypes = num_prototypes

        self.student_encoder = MAEStyleDinoV3Encoder(
            img_size, patch_size, 3, embed_dim, depth, num_heads, mlp_ratio, num_storage_tokens,
            drop_path_rate=drop_path_rate
        )
        self.teacher_encoder = MAEStyleDinoV3Encoder(
            img_size, patch_size, 3, embed_dim, depth, num_heads, mlp_ratio, num_storage_tokens,
            drop_path_rate=0.0  # Teacher doesn't use drop path
        )

        self.decoder = MAEStyleDecoder(
            embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio, self.default_num_patches
        )

        self.student_dec_adapter = nn.Sequential(
            nn.LayerNorm(decoder_embed_dim),
            nn.Linear(decoder_embed_dim, embed_dim)
        )

        self.student_patch_head = DINOProjector(
            in_dim=embed_dim,
            out_dim=projector_dim,
            hidden_dim=2048,
            n_layers=3
        )

        self.teacher_patch_head = DINOProjector(
            in_dim=embed_dim,
            out_dim=projector_dim,
            hidden_dim=2048,
            n_layers=3
        )

        self.prototype_layer = PrototypeLayer(projector_dim, num_prototypes)

        if pretrained_path: self._load_pretrained(pretrained_path)
        self._init_teacher()

    @torch.no_grad()
    def _init_teacher(self):
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        self.teacher_patch_head.load_state_dict(self.student_patch_head.state_dict())
        for p in self.teacher_encoder.parameters(): p.requires_grad = False
        for p in self.teacher_patch_head.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum=0.996):
        for ps, pt in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)
        for ps, pt in zip(self.student_patch_head.parameters(), self.teacher_patch_head.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)

    def _load_pretrained(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        self.student_encoder.load_state_dict(state, strict=False)

    # ------------------------------------------------------------------
    # FIX: Robust Ragged Mask Handling
    # ------------------------------------------------------------------
    def _mask_to_indices(self, mask: torch.Tensor):
        """
        Converts boolean mask (B, N) to indices (B, N_masked) and (B, N_visible).
        Handles variable masking ratios by PADDING to the maximum length in the batch.
        Padding value is 0 (re-using the first patch) which is safe for gather/scatter
        as long as we don't calculate loss on it.
        """
        B, N = mask.shape
        device = mask.device

        # 1. Get counts for this batch
        num_masked = mask.sum(dim=1) # (B,)
        num_visible = N - num_masked

        max_masked = num_masked.max().item()
        max_visible = num_visible.max().item()

        # 2. Prepare containers (initialized with 0 for padding)
        visible_indices = torch.zeros(B, max_visible, dtype=torch.long, device=device)
        masked_indices = torch.zeros(B, max_masked, dtype=torch.long, device=device)

        # 3. Fill row by row (Only efficient way for truly ragged tensors in PyTorch)
        # Note: This loop is fast enough on GPU for batch size ~256

        sorted_indices = torch.argsort(mask.int(), dim=1) # (B, N)

        for b in range(B):
            n_vis = num_visible[b].item()
            n_msk = num_masked[b].item()

            visible_indices[b, :n_vis] = sorted_indices[b, :n_vis]
            if n_vis < max_visible:
                visible_indices[b, n_vis:] = visible_indices[b, n_vis-1]

            masked_indices[b, :n_msk] = sorted_indices[b, N-n_msk:]
            if n_msk < max_masked:
                masked_indices[b, n_msk:] = masked_indices[b, n_msk-1]

        return visible_indices, masked_indices

    def forward(
        self,
        teacher_images: torch.Tensor,
        student_images: torch.Tensor,
        masks: List[torch.Tensor]
    ):
        """
        Forward pass for training with cross-view masked prediction.

        Supports arbitrary input sizes thanks to dynamic RoPE.

        Args:
            teacher_images: (B, 3, H, W) images for teacher (crop1, full view)
            student_images: (B, 3, H, W) images for student (crop2, will be masked)
            masks: List of Tensors, each (B, N) boolean masks (True = masked)

        Returns:
            student_outputs: List of (B, N, K) prototype logits per view
            teacher_logits: (B, N, K) teacher prototype logits
            student_cls_tokens: (B * num_views, D) student CLS tokens for KoLeo
            teacher_cls_token: (B, D) teacher CLS token for PCP loss
            teacher_patch_tokens: (B, N, D) teacher patch tokens for CLS Cross-Attention

        Cross-View Learning:
            Teacher encodes crop1 (strong blur) → target prototypes
            Student encodes crop2 (weak blur + solarize) with masks → predicts targets
            This forces the model to learn view-invariant representations.
        """
        # Compute grid dimensions from input (supports arbitrary sizes)
        _, _, H, W = student_images.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        num_patches = grid_h * grid_w

        # 1. Teacher Forward (Once per image, using teacher_images = crop1)
        with torch.no_grad():
            t_cls_tok, t_patch_toks, _ = self.teacher_encoder.forward_full(teacher_images)
            t_proj = F.normalize(self.teacher_patch_head(t_patch_toks), p=2, dim=-1)
            teacher_logits = self.prototype_layer(t_proj)

        # 2. Student Forward (Fused, using student_images = crop2)
        num_views = len(masks)
        combined_images = torch.cat([student_images] * num_views, dim=0)
        combined_masks = torch.cat(masks, dim=0)

        # Helper indices (Now returns PADDED indices)
        vis_idx, masked_idx = self._mask_to_indices(combined_masks)

        # Run Heavy Encoders
        s_cls_toks, s_vis_toks = self.student_encoder(combined_images, vis_idx, combined_masks)

        # Decoder (pass grid dimensions for variable size support)
        s_decoded = self.decoder(s_vis_toks, vis_idx, masked_idx,
                                 num_patches=num_patches, grid_h=grid_h, grid_w=grid_w)

        # Heads
        s_adapt = self.student_dec_adapter(s_decoded)
        s_proj = F.normalize(self.student_patch_head(s_adapt), p=2, dim=-1)
        s_logits = self.prototype_layer(s_proj)

        # 3. Unstack results
        student_outputs = torch.chunk(s_logits, num_views, dim=0)

        # Return CLS tokens for KoLeo and PCP loss
        # Also return teacher patch tokens for optional CLS Cross-Attention
        return list(student_outputs), teacher_logits, s_cls_toks, t_cls_tok, t_patch_toks

    # ------------------------------------------------------------------
    # Multi-Crop Forward (for Multi-Crop Prototype CLS Loss)
    # ------------------------------------------------------------------
    def forward_multicrop(
        self,
        global_crops: List[torch.Tensor],  # List of 2 tensors, each (B, 3, 224, 224)
        local_crops: List[torch.Tensor],   # List of N tensors, each (B, 3, 96, 96)
        global_masks: List[torch.Tensor],  # List of 2 tensors, each (B, N_global) masks
    ):
        """
        Multi-Crop Forward Pass for combined Mask + Prototype CLS Loss.

        This enables multi-crop training with prototype-based losses:
        - Mask Loss: Patch-level masked prototype prediction (on global crops)
        - Prototype CLS Loss: CLS-level prototype matching across all crops

        Architecture:
            Teacher:
                - Encodes 2 global crops (unmasked) → 2 CLS tokens (for CLS loss targets)
                - Produces patch prototype logits (for mask loss targets)

            Student:
                - Encodes 2 global crops (masked) → mask predictions + 2 CLS tokens
                - Encodes N local crops (unmasked) → N CLS tokens (for CLS loss only)

        Args:
            global_crops: List of 2 global crop tensors, each (B, 3, H_g, W_g)
            local_crops: List of N local crop tensors, each (B, 3, H_l, W_l)
            global_masks: List of 2 mask tensors for global crops, each (B, N_patches)

        Returns:
            dict with:
                - student_patch_outputs: List of 2 (B, N, K) student patch prototype logits
                - teacher_patch_logits: (2, B, N, K) teacher patch prototype logits
                - teacher_global_cls: (2, B, D) teacher CLS tokens from global crops
                - student_global_cls: (2, B, D) student CLS tokens from global crops
                - student_local_cls: (N, B, D) student CLS tokens from local crops
                - global_masks: List of masks (for mask loss)
        """
        B = global_crops[0].shape[0]
        device = global_crops[0].device

        # ============================================
        # 1. Teacher Forward: Encode global crops (unmasked)
        # ============================================
        teacher_global_cls = []
        teacher_patch_logits = []

        with torch.no_grad():
            for global_crop in global_crops:
                # Full forward (no masking)
                t_cls, t_patches, _ = self.teacher_encoder.forward_full(global_crop)
                t_proj = F.normalize(self.teacher_patch_head(t_patches), p=2, dim=-1)
                t_logits = self.prototype_layer(t_proj)

                teacher_global_cls.append(t_cls)
                teacher_patch_logits.append(t_logits)

        # Stack: (2, B, D) and (2, B, N, K)
        teacher_global_cls = torch.stack(teacher_global_cls, dim=0)
        teacher_patch_logits = torch.stack(teacher_patch_logits, dim=0)

        # ============================================
        # 2. Student Forward: Encode global crops (masked)
        # ============================================
        student_global_cls = []
        student_patch_outputs = []

        # Get grid dimensions from global crops
        _, _, H_g, W_g = global_crops[0].shape
        grid_h = H_g // self.patch_size
        grid_w = W_g // self.patch_size
        num_patches_global = grid_h * grid_w

        for i, (global_crop, mask) in enumerate(zip(global_crops, global_masks)):
            # Get indices for masked encoding
            vis_idx, masked_idx = self._mask_to_indices(mask)

            # Student encoder (masked)
            s_cls, s_vis_toks = self.student_encoder(global_crop, vis_idx, mask)

            # Decoder
            s_decoded = self.decoder(
                s_vis_toks, vis_idx, masked_idx,
                num_patches=num_patches_global, grid_h=grid_h, grid_w=grid_w
            )

            # Project to prototype space
            s_adapt = self.student_dec_adapter(s_decoded)
            s_proj = F.normalize(self.student_patch_head(s_adapt), p=2, dim=-1)
            s_logits = self.prototype_layer(s_proj)

            student_global_cls.append(s_cls)
            student_patch_outputs.append(s_logits)

        # Stack: (2, B, D)
        student_global_cls = torch.stack(student_global_cls, dim=0)

        # ============================================
        # 3. Student Forward: Encode local crops (unmasked, CLS only)
        # ============================================
        student_local_cls = []

        if local_crops is not None and len(local_crops) > 0:
            # Batch all local crops together for efficiency
            all_local = torch.cat(local_crops, dim=0)  # (N*B, 3, H_l, W_l)

            # Full forward (no masking) - we only need CLS tokens
            local_cls, _, _ = self.student_encoder.forward_full(all_local)

            # Reshape: (N*B, D) → (N, B, D)
            n_local = len(local_crops)
            student_local_cls = local_cls.reshape(n_local, B, -1)
        else:
            student_local_cls = None

        return {
            "student_patch_outputs": student_patch_outputs,  # List of (B, N, K)
            "teacher_patch_logits": teacher_patch_logits,    # (2, B, N, K)
            "teacher_global_cls": teacher_global_cls,        # (2, B, D)
            "student_global_cls": student_global_cls,        # (2, B, D)
            "student_local_cls": student_local_cls,          # (N, B, D) or None
            "global_masks": global_masks,                    # List of (B, N)
        }

    # ------------------------------------------------------------------
    # Downstream / Analysis Utilities
    # ------------------------------------------------------------------
    def forward_features(self, images, return_attention=False, use_teacher=True):
        encoder = self.teacher_encoder if use_teacher else self.student_encoder
        head = self.teacher_patch_head if use_teacher else self.student_patch_head

        res = encoder.forward_full(images, return_attention=return_attention)
        if return_attention:
            _, patch_tokens, attn_maps = res
        else:
            _, patch_tokens, _ = res
            attn_maps = None

        proj = head(patch_tokens)
        proj = F.normalize(proj, p=2, dim=-1)
        patch_logits = self.prototype_layer(proj)

        if return_attention:
            return patch_tokens, patch_logits, attn_maps
        return patch_tokens, patch_logits

    def get_attention_maps(self, images, use_teacher=True):
        encoder = self.teacher_encoder if use_teacher else self.student_encoder
        _, _, attn_maps = encoder.forward_full(images, return_attention=True)
        return attn_maps


# Backward compatibility alias
MaskedPrototypePredictor = Kodiak


def create_kodiak_small(num_prototypes=128, projector_dim=256, decoder_depth=4, img_size=256, pretrained_path=None):
    return Kodiak(
        num_prototypes=num_prototypes, projector_dim=projector_dim, img_size=img_size, patch_size=16,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, num_storage_tokens=4,
        decoder_embed_dim=192, decoder_depth=decoder_depth, decoder_num_heads=6, pretrained_path=pretrained_path,
    )
