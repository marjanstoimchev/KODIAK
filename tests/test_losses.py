"""Tests for src.losses: Sinkhorn-Knopp, MaskLoss, KoLeo, MultiCropPrototypeCLSLoss."""

import pytest
import torch

from src.losses import KoLeoLoss, KoLeoLossDistributed, MaskLoss, MultiCropPrototypeCLSLoss
from src.losses.losses import sinkhorn_knopp


class TestSinkhorn:
    @staticmethod
    def _column_spread(q):
        col = q.sum(0)
        return ((col.max() - col.min()) / col.mean()).item()

    def test_rows_are_distributions_and_columns_balanced(self):
        torch.manual_seed(0)
        logits = torch.randn(256, 8) * 0.3  # cosine-similarity-like range, as produced by the model
        q = sinkhorn_knopp(logits, epsilon=0.05, niters=3)
        assert q.shape == (256, 8)
        torch.testing.assert_close(q.sum(1), torch.ones(256), atol=1e-5, rtol=0)
        # balanced assignment: prototypes receive far more even mass than a plain softmax ...
        softmax_spread = self._column_spread(torch.softmax(logits / 0.05, dim=-1))
        assert self._column_spread(q) < 0.5 * softmax_spread
        # ... and the balance tightens with more iterations
        assert self._column_spread(sinkhorn_knopp(logits, epsilon=0.05, niters=50)) < 0.05

    def test_flattens_3d_input(self):
        q = sinkhorn_knopp(torch.randn(2, 3, 4))
        assert q.shape == (6, 4)


class TestMaskLoss:
    def test_zero_when_nothing_masked(self):
        loss_fn = MaskLoss(num_prototypes=8)
        s = torch.randn(2, 4, 8, requires_grad=True)
        t = torch.randn(2, 4, 8)
        out = loss_fn(s, t, torch.zeros(2, 4, dtype=torch.bool))
        assert out.item() == 0.0

    def test_only_masked_positions_contribute(self):
        torch.manual_seed(0)
        loss_fn = MaskLoss(num_prototypes=8)
        t = torch.randn(2, 4, 8)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        mask[0, 1] = True
        mask[1, 3] = True
        s1 = torch.randn(2, 4, 8)
        s2 = s1.clone()
        s2[~mask] += torch.randn_like(s2[~mask])  # perturb unmasked positions only
        torch.testing.assert_close(loss_fn(s1, t, mask), loss_fn(s2, t, mask))

    def test_loss_is_cross_entropy_against_balanced_targets(self):
        torch.manual_seed(0)
        loss_fn = MaskLoss(num_prototypes=8, use_sinkhorn=False, use_centering=False, teacher_temp=1.0)
        s = torch.randn(1, 4, 8)
        t = torch.randn(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.bool)
        expected = -(torch.softmax(t[0], -1) * torch.log_softmax(s[0], -1)).sum(-1).mean()
        torch.testing.assert_close(loss_fn(s, t, mask), expected)

    def test_center_update_is_ema(self):
        loss_fn = MaskLoss(num_prototypes=4, center_momentum=0.5)
        t = torch.ones(2, 3, 4)
        loss_fn.update_center(t)
        torch.testing.assert_close(loss_fn.patch_center, torch.full((1, 1, 4), 0.5))
        loss_fn.update_center(t)
        torch.testing.assert_close(loss_fn.patch_center, torch.full((1, 1, 4), 0.75))

    def test_gradient_flows_to_student_only(self):
        loss_fn = MaskLoss(num_prototypes=8)
        s = torch.randn(2, 4, 8, requires_grad=True)
        t = torch.randn(2, 4, 8, requires_grad=True)
        mask = torch.ones(2, 4, dtype=torch.bool)
        loss_fn(s, t, mask).backward()
        assert s.grad is not None and s.grad.abs().sum() > 0
        assert t.grad is None


class TestKoLeo:
    def test_spread_features_have_lower_loss_than_clustered(self):
        torch.manual_seed(0)
        koleo = KoLeoLoss()
        clustered = torch.randn(1, 16).repeat(32, 1) + 1e-3 * torch.randn(32, 16)
        spread = torch.randn(32, 16)
        assert koleo(clustered) > koleo(spread)

    def test_distributed_variant_matches_single_process(self):
        torch.manual_seed(0)
        x = torch.randn(16, 8)
        torch.testing.assert_close(KoLeoLoss()(x), KoLeoLossDistributed(topk=1)(x))

    def test_gradient(self):
        x = torch.randn(8, 4, requires_grad=True)
        KoLeoLoss()(x).backward()
        assert x.grad is not None


class TestMultiCropPrototypeCLSLoss:
    @pytest.fixture
    def loss_fn(self):
        torch.manual_seed(0)
        return MultiCropPrototypeCLSLoss(embed_dim=8, num_prototypes=16)

    def test_terms_and_metrics(self, loss_fn):
        B, D, n_local = 3, 8, 4
        teacher = torch.randn(2, B, D)
        student = torch.randn(2 + n_local, B, D)
        loss, m = loss_fn(teacher, student)
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert m["n_global_terms"] == 2  # cross-view only: (s0,t1), (s1,t0)
        assert m["n_local_terms"] == n_local * 2
        assert m["n_local_crops"] == n_local
        assert 1 <= m["cls_proto_util"] <= 16

    def test_global_only(self, loss_fn):
        loss, m = loss_fn(torch.randn(2, 3, 8), torch.randn(2, 3, 8))
        assert m["n_local_terms"] == 0 and m["cls_local_loss"] == 0.0
        assert torch.isfinite(loss)

    def test_2d_teacher_is_promoted(self, loss_fn):
        # one teacher crop (B, D) + two student global crops -> one cross-view term (s1 -> t0)
        loss, m = loss_fn(torch.randn(3, 8), torch.randn(2, 3, 8))
        assert m["n_global_terms"] == 1 and m["n_local_terms"] == 0
        assert torch.isfinite(loss) and loss.item() > 0

    def test_2d_student_is_rejected(self, loss_fn):
        with pytest.raises(ValueError):
            loss_fn(torch.randn(2, 3, 8), torch.randn(3, 8))  # a single student crop

    def test_too_few_student_crops_rejected(self, loss_fn):
        with pytest.raises(ValueError):
            loss_fn(torch.randn(2, 3, 8), torch.randn(1, 3, 8))

    def test_update_center_flag(self, loss_fn):
        teacher, student = torch.randn(2, 3, 8), torch.randn(4, 3, 8)
        loss_fn(teacher, student, update_center=False)
        assert torch.equal(loss_fn.center, torch.zeros(1, 16))
        loss_fn(teacher, student, update_center=True)
        assert not torch.equal(loss_fn.center, torch.zeros(1, 16))

    def test_head_is_trainable_and_receives_gradients(self, loss_fn):
        loss, _ = loss_fn(torch.randn(2, 3, 8), torch.randn(4, 3, 8, requires_grad=True))
        loss.backward()
        grads = [p.grad for p in loss_fn.cls_prototype_head.parameters()]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)

    def test_top_prototypes(self, loss_fn):
        idx, probs = loss_fn.get_top_prototypes(torch.randn(5, 8), top_k=3)
        assert idx.shape == (5, 3) and probs.shape == (5, 3)
        assert (probs[:, 0] >= probs[:, 1]).all()
