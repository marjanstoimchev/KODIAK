"""Tests for the Lightning learners: schedules, parameter groups, train/val/test steps."""

import math

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.learners import ClassificationLearner, MotifLearner
from src.learners.pretraining import (
    CosineScheduler,
    cosine_schedule,
    get_vit_lr_decay_rate,
    scale_lr,
    should_skip_weight_decay,
)
from src.models import LinearClassifier


def _random_masks(B, N, ratio=0.5):
    masks = torch.zeros(B, N, dtype=torch.bool)
    n = int(ratio * N)
    for b in range(B):
        masks[b, torch.randperm(N)[:n]] = True
    return masks


def _multicrop_batch(tiny_kwargs, B=2):
    k = tiny_kwargs
    s, ls, n_local = k["img_size"], k["local_crops_size"], k["local_crops_number"]
    N = (s // k["patch_size"]) ** 2
    return {
        "global_crops": [torch.randn(B, 3, s, s), torch.randn(B, 3, s, s)],
        "local_crops": [torch.randn(B, 3, ls, ls) for _ in range(n_local)],
        "masks": [_random_masks(B, N), _random_masks(B, N)],
    }


def _legacy_batch(tiny_kwargs, B=2):
    k = tiny_kwargs
    s = k["img_size"]
    N = (s // k["patch_size"]) ** 2
    return {
        "teacher_images": torch.randn(B, 3, s, s),
        "student_images": torch.randn(B, 3, s, s),
        "masks": [_random_masks(B, N), _random_masks(B, N)],
    }


class _DictDataset(Dataset):
    def __init__(self, make_batch, n=6):
        self.items = [make_batch() for _ in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class TestSchedules:
    def test_cosine_scheduler_phases(self):
        sched = CosineScheduler(base_value=1.0, final_value=0.1, total_iters=20, warmup_iters=5, freeze_iters=2)
        assert len(sched) == 20
        assert sched[0] == 0.0 and sched[1] == 0.0  # freeze
        assert sched[2] == 0.0  # warmup starts at start_warmup_value
        assert math.isclose(sched[6], 1.0)  # end of warmup
        assert sched[7] <= 1.0 and sched[19] >= 0.1
        assert sched[100] == 0.1  # beyond schedule -> final value
        assert all(sched[i] >= sched[i + 1] for i in range(7, 19))  # monotone decay

    def test_cosine_scheduler_degenerate(self):
        sched = CosineScheduler(base_value=1.0, final_value=0.0, total_iters=0, warmup_iters=0)
        assert sched[0] == 0.0

    def test_epoch_cosine_schedule_endpoints(self):
        f = cosine_schedule(0.9, 1.0, epochs=10)
        assert f(0) == 0.9
        assert math.isclose(f(10), 1.0)
        assert 0.9 < f(5) < 1.0

    def test_layerwise_decay(self):
        assert get_vit_lr_decay_rate("student_encoder.patch_embed.proj.weight", 0.9, 12) == pytest.approx(0.9 ** 13)
        assert get_vit_lr_decay_rate("student_encoder.cls_token", 0.9, 12) == pytest.approx(0.9 ** 13)
        assert get_vit_lr_decay_rate("student_encoder.blocks.0.attn.qkv.weight", 0.9, 12) == pytest.approx(0.9 ** 12)
        assert get_vit_lr_decay_rate("student_encoder.blocks.11.mlp.fc1.weight", 0.9, 12) == pytest.approx(0.9)
        assert get_vit_lr_decay_rate("student_encoder.norm.weight", 0.9, 12) == pytest.approx(1.0)
        assert get_vit_lr_decay_rate("decoder.blocks.0.weight", 0.9, 12) == pytest.approx(1.0)  # not backbone

    def test_weight_decay_skip(self):
        assert should_skip_weight_decay("blocks.0.attn.qkv.bias")
        assert should_skip_weight_decay("blocks.0.norm1.weight")
        assert should_skip_weight_decay("blocks.0.ls1.gamma")
        assert not should_skip_weight_decay("blocks.0.attn.qkv.weight")
        assert not should_skip_weight_decay("cls_token")

    def test_lr_scaling_uses_per_device_batch_times_world_size(self, tiny_kodiak):
        """Total batch 128 on 2 devices must scale like 128, not 256."""
        learner = MotifLearner(model=tiny_kodiak, batch_size=64, lr=1e-4, max_epochs=1)
        assert scale_lr(1e-4, learner.batch_size, 2) == pytest.approx(scale_lr(1e-4, 128, 1))

    def test_scale_lr(self):
        assert scale_lr(1e-3, 1024, 1) == pytest.approx(4e-3)
        assert scale_lr(1e-3, 256, 1) == pytest.approx(2e-3)
        assert scale_lr(1e-3, 256, 1, "none") == 1e-3
        with pytest.raises(ValueError):
            scale_lr(1e-3, 256, 1, "linear")


class TestMotifLearner:
    @pytest.fixture
    def learner(self, tiny_kodiak):
        return MotifLearner(model=tiny_kodiak, batch_size=4, max_epochs=2, warmup_epochs=1,
                            teacher_temp_warmup_epochs=1, lr=1e-3)

    def test_all_trainable_params_are_optimised(self, learner):
        groups = learner._build_param_groups(scaled_lr=1e-3, weight_decay=0.04, num_layers=1)
        in_opt = {id(p) for g in groups for p in g["params"]}
        trainable = {id(p) for p in learner.parameters() if p.requires_grad}
        assert in_opt == trainable
        # the CLS prototype head is part of the learner, not the model, and must be trained
        head_ids = {id(p) for p in learner.prototype_cls_loss.cls_prototype_head.parameters()}
        assert head_ids <= in_opt
        # frozen teacher is excluded
        assert not any(id(p) in in_opt for p in learner.model.teacher_encoder.parameters())

    def test_last_layer_groups(self, learner):
        groups = learner._build_param_groups(scaled_lr=1e-3, weight_decay=0.04, num_layers=1)
        last = {id(p) for g in groups if g["is_last_layer"] for p in g["params"]}
        assert id(learner.model.prototype_layer.layer.weight) in last
        assert id(learner.prototype_cls_loss.cls_prototype_head[-1].weight) in last
        assert id(learner.prototype_cls_loss.cls_prototype_head[0].weight) not in last
        # bias / norm params get no weight decay
        for g in groups:
            if g["wd_multiplier"] == 0.0:
                assert g["weight_decay"] == 0.0

    def test_freeze_cls_head_reproduces_fixed_random_projection(self, tiny_kodiak):
        learner = MotifLearner(model=tiny_kodiak, freeze_cls_prototype_head=True, batch_size=4)
        head_ids = {id(p) for p in learner.prototype_cls_loss.cls_prototype_head.parameters()}
        groups = learner._build_param_groups(scaled_lr=1e-3, weight_decay=0.04, num_layers=1)
        in_opt = {id(p) for g in groups for p in g["params"]}
        assert not (head_ids & in_opt)
        assert id(tiny_kodiak.prototype_layer.layer.weight) in in_opt
        # the encoder still receives gradients through the frozen head
        total, *_ = learner.shared_step(_multicrop_batch(dict(
            img_size=64, patch_size=16, local_crops_size=32, local_crops_number=2)), is_train=True)
        total.backward()
        assert any(p.grad is not None for p in tiny_kodiak.student_encoder.parameters())

    def test_cls_loss_disabled(self, tiny_kodiak):
        learner = MotifLearner(model=tiny_kodiak, prototype_cls_loss_weight=0.0, batch_size=4)
        assert learner.prototype_cls_loss is None
        total, mask, koleo, proto, *_ = learner.shared_step(_multicrop_batch(dict(
            img_size=64, patch_size=16, local_crops_size=32, local_crops_number=2)), is_train=True)
        assert proto.item() == 0.0 and torch.isfinite(total)

    @pytest.mark.parametrize("make_batch", [_multicrop_batch, _legacy_batch])
    def test_shared_step_modes(self, learner, tiny_kwargs, make_batch):
        total, mask, koleo, proto, t_logits, metrics = learner.shared_step(make_batch(tiny_kwargs), is_train=True)
        assert torch.isfinite(total) and total.requires_grad
        assert t_logits.shape[-1] == tiny_kwargs["num_prototypes"]
        torch.testing.assert_close(
            total, mask + learner.prototype_cls_loss_weight * proto + learner.koleo_loss_weight * koleo)

    def test_validation_does_not_update_centers(self, learner, tiny_kwargs):
        batch = _multicrop_batch(tiny_kwargs)
        learner.shared_step(batch, is_train=False)
        assert torch.equal(learner.loss_fn.patch_center, torch.zeros_like(learner.loss_fn.patch_center))
        assert torch.equal(learner.prototype_cls_loss.center, torch.zeros_like(learner.prototype_cls_loss.center))
        learner.shared_step(batch, is_train=True)
        assert not torch.equal(learner.loss_fn.patch_center, torch.zeros_like(learner.loss_fn.patch_center))
        assert not torch.equal(learner.prototype_cls_loss.center, torch.zeros_like(learner.prototype_cls_loss.center))

    def test_fit_two_epochs_on_cpu(self, tmp_path, learner, tiny_kwargs):
        """Full Lightning loop: schedules applied, teacher EMA updated, checkpoint saved."""
        dataset = _DictDataset(lambda: {k: (v[0] if isinstance(v, torch.Tensor) else [t[0] for t in v])
                                        for k, v in _multicrop_batch(tiny_kwargs, B=1).items()}, n=6)

        def collate(items):
            out = {}
            for key in items[0]:
                if isinstance(items[0][key], list):
                    out[key] = [torch.stack([it[key][i] for it in items]) for i in range(len(items[0][key]))]
                else:
                    out[key] = torch.stack([it[key] for it in items])
            return out

        loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
        teacher_before = learner.model.teacher_encoder.cls_token.detach().clone()
        head_before = learner.prototype_cls_loss.cls_prototype_head[0].weight.detach().clone()

        trainer = pl.Trainer(max_epochs=2, accelerator="cpu", devices=1, logger=False,
                             enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0,
                             num_sanity_val_steps=0, gradient_clip_val=3.0, default_root_dir=str(tmp_path))
        trainer.fit(learner, train_dataloaders=loader)

        assert trainer.global_step == 6
        assert learner.loss_fn.teacher_temp == learner.teacher_temp_final  # warmup finished
        assert not torch.equal(teacher_before, learner.model.teacher_encoder.cls_token)  # EMA moved
        assert not torch.equal(head_before, learner.prototype_cls_loss.cls_prototype_head[0].weight)  # head trained
        lrs = [g["lr"] for g in trainer.optimizers[0].param_groups]
        assert all(math.isfinite(lr) and lr >= 0 for lr in lrs)

        ckpt = tmp_path / "last.ckpt"
        trainer.save_checkpoint(str(ckpt))
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert "model.student_encoder.cls_token" in state["state_dict"]
        assert "prototype_cls_loss.cls_prototype_head.0.weight" in state["state_dict"]
        assert state["hyper_parameters"]["n_local_crops"] == 8  # default stored for folder naming


class TestClassificationLearner:
    @pytest.fixture
    def clf(self, tiny_kwargs):
        k = tiny_kwargs
        return LinearClassifier(num_classes=3, img_size=k["img_size"], embed_dim=k["embed_dim"], depth=k["depth"],
                                num_heads=k["num_heads"], num_storage_tokens=k["num_storage_tokens"])

    def _batch(self, tiny_kwargs, n=4, num_classes=3):
        s = tiny_kwargs["img_size"]
        return {"images": torch.randn(n, 3, s, s), "labels": torch.arange(n) % num_classes}

    def test_steps(self, clf, tiny_kwargs):
        learner = ClassificationLearner(num_classes=3, model=clf, max_epochs=1)
        batch = self._batch(tiny_kwargs)
        loss = learner.training_step(batch, 0)
        assert torch.isfinite(loss) and loss.requires_grad
        learner.validation_step(batch, 0)
        learner.test_step(batch, 0)
        learner.test_step((batch["images"], batch["labels"]), 1)  # tuple batches are accepted
        assert 0.0 <= learner.test_acc.compute() <= 1.0
        assert 0.0 <= learner.test_auroc.compute() <= 1.0

    def test_binary_classification(self, tiny_kwargs):
        k = tiny_kwargs
        clf = LinearClassifier(num_classes=2, img_size=k["img_size"], embed_dim=k["embed_dim"], depth=k["depth"],
                               num_heads=k["num_heads"], num_storage_tokens=k["num_storage_tokens"])
        learner = ClassificationLearner(num_classes=2, model=clf)
        learner.test_step(self._batch(tiny_kwargs, n=6, num_classes=2), 0)
        assert learner.test_acc_top5 is None
        assert 0.0 <= learner.test_f1_macro.compute() <= 1.0

    def test_invalid_num_classes(self, clf):
        with pytest.raises(ValueError):
            ClassificationLearner(num_classes=1, model=clf)

    def test_optimizer_groups_and_freeze(self, clf):
        learner = ClassificationLearner(num_classes=3, model=clf, lr=1e-3, backbone_lr_scale=0.1, max_epochs=1)
        trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                             enable_progress_bar=False)
        learner.trainer = trainer
        trainer.strategy.connect(learner)
        # estimated_stepping_batches needs a datamodule; emulate via a tiny loader
        trainer.fit_loop.setup_data = lambda: None
        trainer.fit_loop._combined_loader = None
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(type(trainer), "estimated_stepping_batches", property(lambda self: 10))
            optimizers, schedulers = learner.configure_optimizers()
        groups = optimizers[0].param_groups
        assert len(groups) == 2
        assert groups[0]["lr"] == pytest.approx(1e-4)  # backbone scaled
        assert groups[1]["lr"] == pytest.approx(1e-3)  # head
        assert schedulers[0]["interval"] == "step"

        frozen = ClassificationLearner(num_classes=3, model=clf, freeze_backbone=True)
        with pytest.MonkeyPatch.context() as mp:
            frozen.trainer = trainer
            mp.setattr(type(trainer), "estimated_stepping_batches", property(lambda self: 10))
            optimizers, _ = frozen.configure_optimizers()
        assert len(optimizers[0].param_groups) == 1  # only the head

    def test_checkpoint_roundtrip(self, tmp_path, clf, tiny_kwargs):
        k = tiny_kwargs
        learner = ClassificationLearner(num_classes=3, img_size=k["img_size"], embed_dim=k["embed_dim"],
                                        vit_depth=k["depth"], vit_heads=k["num_heads"],
                                        num_storage_tokens=k["num_storage_tokens"], max_epochs=1)
        trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                             enable_progress_bar=False)
        trainer.strategy.connect(learner)
        path = tmp_path / "clf.ckpt"
        trainer.save_checkpoint(str(path))
        loaded = ClassificationLearner.load_from_checkpoint(str(path), map_location="cpu")
        for (n, a), (_, b) in zip(learner.state_dict().items(), loaded.state_dict().items()):
            torch.testing.assert_close(a, b, msg=n)
