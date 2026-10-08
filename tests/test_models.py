"""Tests for src.models: Kodiak (teacher/student), encoder masking, decoder, classifier."""

from pathlib import Path

import pytest
import torch

from src.models import LinearClassifier, MAEStyleDecoder, create_kodiak_small
from src.models.components import DinoV3Encoder

ROOT = Path(__file__).resolve().parent.parent
DINOV3_WEIGHTS = ROOT / "dinov3_weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def _random_masks(B, N, ratios, generator=None):
    """Boolean masks (True = masked) with a given number of masked patches per row."""
    masks = torch.zeros(B, N, dtype=torch.bool)
    for b, r in enumerate(ratios):
        n = int(round(r * N))
        idx = torch.randperm(N, generator=generator)[:n]
        masks[b, idx] = True
    return masks


class TestMaskToIndices:
    def test_indices_partition_mask(self, tiny_kodiak):
        B, N = 4, 16
        g = torch.Generator().manual_seed(0)
        mask = _random_masks(B, N, [0.0, 0.25, 0.5, 0.75], g)
        vis, msk = tiny_kodiak._mask_to_indices(mask)

        assert vis.shape[1] == int((~mask).sum(1).max())
        assert msk.shape[1] == int(mask.sum(1).max())
        for b in range(B):
            n_vis = int((~mask[b]).sum())
            n_msk = int(mask[b].sum())
            # the first n_vis / n_msk entries are exactly the visible / masked positions
            assert set(vis[b, :n_vis].tolist()) == set(torch.where(~mask[b])[0].tolist())
            assert set(msk[b, :n_msk].tolist()) == set(torch.where(mask[b])[0].tolist())
            # padding entries never point to a masked patch for the visible side
            assert not mask[b, vis[b]].any()
            # padding on the masked side only repeats already-masked indices (or a visible one when empty)
            if n_msk > 0:
                assert mask[b, msk[b]].all()

    def test_visible_indices_keep_spatial_order(self, tiny_kodiak):
        mask = torch.zeros(1, 16, dtype=torch.bool)
        mask[0, [1, 5, 9]] = True
        vis, msk = tiny_kodiak._mask_to_indices(mask)
        assert vis[0].tolist() == [i for i in range(16) if i not in (1, 5, 9)]
        assert msk[0].tolist() == [1, 5, 9]

    def test_fully_masked_row_rejected(self, tiny_kodiak):
        mask = torch.ones(1, 16, dtype=torch.bool)
        with pytest.raises(ValueError):
            tiny_kodiak._mask_to_indices(mask)


class TestKodiakForward:
    def test_legacy_forward_shapes(self, tiny_kodiak, tiny_kwargs):
        B, N, K, D = 3, 16, tiny_kwargs["num_prototypes"], tiny_kwargs["embed_dim"]
        s = tiny_kwargs["img_size"]
        teacher = torch.randn(B, 3, s, s)
        student = torch.randn(B, 3, s, s)
        masks = [_random_masks(B, N, [0.5, 0.25, 0.1]), _random_masks(B, N, [0.3, 0.3, 0.3])]

        outs, t_logits, s_cls, t_cls, t_patches = tiny_kodiak(teacher, student, masks)

        assert len(outs) == 2
        assert all(o.shape == (B, N, K) for o in outs)
        assert t_logits.shape == (B, N, K)
        assert s_cls.shape == (B * 2, D)
        assert t_cls.shape == (B, D)
        assert t_patches.shape == (B, N, D)
        assert not t_logits.requires_grad  # teacher is detached
        assert outs[0].requires_grad

    def test_multicrop_forward_shapes(self, tiny_kodiak, tiny_kwargs):
        B, N, K, D = 2, 16, tiny_kwargs["num_prototypes"], tiny_kwargs["embed_dim"]
        s, ls, n_local = tiny_kwargs["img_size"], tiny_kwargs["local_crops_size"], 3
        globals_ = [torch.randn(B, 3, s, s), torch.randn(B, 3, s, s)]
        locals_ = [torch.randn(B, 3, ls, ls) for _ in range(n_local)]
        masks = [_random_masks(B, N, [0.5, 0.25]), _random_masks(B, N, [0.1, 0.5])]

        out = tiny_kodiak.forward_multicrop(globals_, locals_, masks)

        assert len(out["student_patch_outputs"]) == 2
        assert out["student_patch_outputs"][0].shape == (B, N, K)
        assert out["teacher_patch_logits"].shape == (2, B, N, K)
        assert out["teacher_global_cls"].shape == (2, B, D)
        assert out["student_global_cls"].shape == (2, B, D)
        assert out["student_local_cls"].shape == (n_local, B, D)

    def test_multicrop_without_local_crops(self, tiny_kodiak, tiny_kwargs):
        B, s = 2, tiny_kwargs["img_size"]
        globals_ = [torch.randn(B, 3, s, s), torch.randn(B, 3, s, s)]
        masks = [_random_masks(B, 16, [0.5, 0.25])] * 2
        out = tiny_kodiak.forward_multicrop(globals_, [], masks)
        assert out["student_local_cls"] is None

    def test_forward_features(self, tiny_kodiak, tiny_kwargs):
        s = tiny_kwargs["img_size"]
        patches, logits = tiny_kodiak.forward_features(torch.randn(2, 3, s, s))
        assert patches.shape == (2, 16, tiny_kwargs["embed_dim"])
        assert logits.shape == (2, 16, tiny_kwargs["num_prototypes"])

    def test_variable_resolution(self, tiny_kodiak, tiny_kwargs):
        """Dynamic RoPE + interpolated decoder pos-embed support non-default sizes."""
        B, s2 = 2, tiny_kwargs["img_size"] * 2  # 128px -> 8x8 = 64 patches
        masks = [_random_masks(B, 64, [0.5, 0.5])]
        outs, t_logits, *_ = tiny_kodiak(torch.randn(B, 3, s2, s2), torch.randn(B, 3, s2, s2), masks)
        assert outs[0].shape == (B, 64, tiny_kwargs["num_prototypes"])
        assert t_logits.shape == (B, 64, tiny_kwargs["num_prototypes"])


class TestTeacherStudent:
    def test_teacher_initialised_from_student_and_frozen(self, tiny_kodiak):
        for ps, pt in zip(tiny_kodiak.student_encoder.parameters(), tiny_kodiak.teacher_encoder.parameters()):
            assert torch.equal(ps, pt)
            assert not pt.requires_grad
        for p in tiny_kodiak.teacher_patch_head.parameters():
            assert not p.requires_grad

    def test_update_teacher_is_ema(self, tiny_kodiak):
        with torch.no_grad():
            for p in tiny_kodiak.student_encoder.parameters():
                p.add_(1.0)
        before = [p.clone() for p in tiny_kodiak.teacher_encoder.parameters()]
        tiny_kodiak.update_teacher(momentum=0.9)
        for b, ps, pt in zip(before, tiny_kodiak.student_encoder.parameters(), tiny_kodiak.teacher_encoder.parameters()):
            torch.testing.assert_close(pt, 0.9 * b + 0.1 * ps)

    def test_only_student_side_has_gradients(self, tiny_kodiak, tiny_kwargs):
        B, N, s = 2, 16, tiny_kwargs["img_size"]
        masks = [_random_masks(B, N, [0.5, 0.5])]
        outs, *_ = tiny_kodiak(torch.randn(B, 3, s, s), torch.randn(B, 3, s, s), masks)
        outs[0].sum().backward()
        assert all(p.grad is None for p in tiny_kodiak.teacher_encoder.parameters())
        assert any(p.grad is not None for p in tiny_kodiak.student_encoder.parameters())
        assert any(p.grad is not None for p in tiny_kodiak.decoder.parameters())


class TestDecoder:
    def test_pos_embed_interpolation(self):
        dec = MAEStyleDecoder(embed_dim=8, decoder_embed_dim=8, decoder_depth=1, decoder_num_heads=2,
                              mlp_ratio=2.0, num_patches=16)
        assert dec._get_pos_embed(16).shape == (1, 16, 8)
        assert dec._get_pos_embed(64, 8, 8).shape == (1, 64, 8)
        assert dec._get_pos_embed(12, 3, 4).shape == (1, 12, 8)

    def test_masked_positions_get_mask_token(self):
        torch.manual_seed(0)
        dec = MAEStyleDecoder(embed_dim=8, decoder_embed_dim=8, decoder_depth=0, decoder_num_heads=2,
                              mlp_ratio=2.0, num_patches=4)
        vis = torch.tensor([[0, 2]])
        msk = torch.tensor([[1, 3]])
        feats = torch.randn(1, 2, 8)
        out = dec(feats, vis, msk, num_patches=4, grid_h=2, grid_w=2)
        expected_masked = dec.decoder_norm(dec.mask_token[0, 0] + dec.decoder_pos_embed[0, 1])
        torch.testing.assert_close(out[0, 1], expected_masked)


class TestEncoder:
    def test_masked_forward_matches_full_forward_on_visible_tokens_without_masking(self):
        torch.manual_seed(0)
        enc = DinoV3Encoder(img_size=32, patch_size=16, embed_dim=16, depth=1, num_heads=2, num_storage_tokens=2)
        enc.eval()
        x = torch.randn(2, 3, 32, 32)
        cls_full, patches_full, _ = enc.forward_full(x)
        vis = torch.arange(4).unsqueeze(0).expand(2, -1)
        cls_m, patches_m = enc(x, vis, torch.zeros(2, 4, dtype=torch.bool))
        torch.testing.assert_close(cls_full, cls_m, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(patches_full, patches_m, atol=1e-5, rtol=1e-5)

    def test_no_storage_tokens(self):
        enc = DinoV3Encoder(img_size=32, patch_size=16, embed_dim=16, depth=1, num_heads=2, num_storage_tokens=0)
        cls, patches, _ = enc.forward_full(torch.randn(1, 3, 32, 32))
        assert cls.shape == (1, 16) and patches.shape == (1, 4, 16)


class TestLinearClassifier:
    @pytest.mark.parametrize("use_cls_token,concat", [(True, False), (False, False), (True, True)])
    def test_forward(self, tiny_kwargs, use_cls_token, concat):
        k = tiny_kwargs
        clf = LinearClassifier(num_classes=5, img_size=k["img_size"], embed_dim=k["embed_dim"], depth=k["depth"],
                               num_heads=k["num_heads"], num_storage_tokens=k["num_storage_tokens"],
                               use_cls_token=use_cls_token, concat_cls_patch=concat)
        logits = clf(torch.randn(2, 3, k["img_size"], k["img_size"]))
        assert logits.shape == (2, 5)
        assert clf.head.in_features == (2 * k["embed_dim"] if concat else k["embed_dim"])

    def test_freeze_backbone(self, tiny_kwargs):
        k = tiny_kwargs
        clf = LinearClassifier(num_classes=3, img_size=k["img_size"], embed_dim=k["embed_dim"], depth=k["depth"],
                               num_heads=k["num_heads"], num_storage_tokens=k["num_storage_tokens"])
        clf.freeze_backbone()
        assert all(not p.requires_grad for p in clf.encoder.parameters())
        assert all(p.requires_grad for p in clf.head.parameters())
        clf.unfreeze_backbone()
        assert all(p.requires_grad for p in clf.encoder.parameters())

    @pytest.mark.parametrize("encoder_type", ["teacher", "student"])
    def test_load_from_kodiak_lightning_checkpoint(self, tmp_path, tiny_kodiak, tiny_kwargs, encoder_type):
        """A Lightning checkpoint (keys prefixed with `model.`) loads into the classifier encoder."""
        k = tiny_kwargs
        # make teacher and student differ so we can tell which one was loaded
        with torch.no_grad():
            for p in tiny_kodiak.student_encoder.parameters():
                p.add_(0.5)
        ckpt = {"state_dict": {f"model.{n}": v for n, v in tiny_kodiak.state_dict().items()},
                "hyper_parameters": {"num_prototypes": k["num_prototypes"]}}
        path = tmp_path / "last.ckpt"
        torch.save(ckpt, path)

        clf = LinearClassifier(num_classes=3, img_size=k["img_size"], embed_dim=k["embed_dim"], depth=k["depth"],
                               num_heads=k["num_heads"], num_storage_tokens=None, mask_k_bias=None,
                               pretrained_path=str(path), encoder_type=encoder_type)
        src = tiny_kodiak.teacher_encoder if encoder_type == "teacher" else tiny_kodiak.student_encoder
        for (n, p_loaded), (_, p_src) in zip(clf.encoder.state_dict().items(), src.state_dict().items()):
            torch.testing.assert_close(p_loaded, p_src, msg=n)
        assert clf._num_storage_tokens == k["num_storage_tokens"]
        assert clf._mask_k_bias is True

    def test_missing_weights_warns_but_runs(self, tmp_path, tiny_kwargs, capsys):
        k = tiny_kwargs
        path = tmp_path / "junk.ckpt"
        torch.save({"state_dict": {"foo.bar": torch.zeros(1)}}, path)
        LinearClassifier(num_classes=3, img_size=k["img_size"], embed_dim=k["embed_dim"], depth=k["depth"],
                         num_heads=k["num_heads"], num_storage_tokens=4, pretrained_path=str(path))
        assert "RANDOMLY INITIALIZED" in capsys.readouterr().out

    @pytest.mark.skipif(not DINOV3_WEIGHTS.exists(), reason="official DINOv3 weights not present")
    def test_load_official_dinov3_weights(self):
        clf = LinearClassifier(num_classes=10, pretrained_path=str(DINOV3_WEIGHTS))
        logits = clf(torch.randn(1, 3, 64, 64))
        assert logits.shape == (1, 10)
        assert clf._num_storage_tokens == 4

    @pytest.mark.skipif(not DINOV3_WEIGHTS.exists(), reason="official DINOv3 weights not present")
    def test_kodiak_continued_init_from_dinov3(self):
        """Continued pretraining initialises the student (and therefore teacher) from DINOv3 weights."""
        model = create_kodiak_small(num_prototypes=8, pretrained_path=str(DINOV3_WEIGHTS))
        official = torch.load(DINOV3_WEIGHTS, map_location="cpu", weights_only=False)
        torch.testing.assert_close(model.student_encoder.cls_token, official["cls_token"])
        torch.testing.assert_close(model.teacher_encoder.cls_token, official["cls_token"])
