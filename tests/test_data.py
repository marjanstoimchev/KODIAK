"""Tests for src.data: masking, collates, transforms, datasets, datamodules, samplers."""

import pandas as pd
import pytest
import torch

from src.data import (
    ClassificationDataModule,
    ClassificationTrainTransform,
    ClassificationValTransform,
    CustomPatchDataset,
    DatasetRegistry,
    EpochSampler,
    KodiakCollate,
    MaskingGenerator,
    MultiCropKodiakCollate,
    PretrainingDataModule,
    SamplerType,
    ShardedInfiniteSampler,
    classification_collate,
    make_sampler,
)
from src.data.datasets.huggingface import dataset as hf_dataset
from src.data.pretraining.transforms import (
    DataAugmentationDINO,
    DINOv3PretrainingTransform,
    MultiCropPretrainingTransform,
)


class TestMaskingGenerator:
    @pytest.mark.parametrize("n", [0, 1, 5, 8, 15])
    def test_exact_number_of_masked_patches(self, n):
        gen = MaskingGenerator((4, 4))
        m = gen(n)
        assert m.shape == (4, 4) and m.dtype == bool
        assert int(m.sum()) == n

    def test_int_input_size(self):
        assert MaskingGenerator(4).get_shape() == (4, 4)


class TestCollates:
    def test_multicrop_collate(self):
        gen = MaskingGenerator((4, 4))
        collate = MultiCropKodiakCollate(gen, mask_ratio_tuple=(0.1, 0.5), mask_probability=0.5, num_patches_total=16)
        items = [{"global_crops": [torch.randn(3, 64, 64), torch.randn(3, 64, 64)],
                  "local_crops": [torch.randn(3, 32, 32)] * 3, "label": i % 2} for i in range(8)]
        batch = collate(items)
        assert len(batch["global_crops"]) == 2 and batch["global_crops"][0].shape == (8, 3, 64, 64)
        assert len(batch["local_crops"]) == 3 and batch["local_crops"][0].shape == (8, 3, 32, 32)
        assert len(batch["masks"]) == 2 and batch["masks"][0].shape == (8, 16)
        assert batch["masks"][0].dtype == torch.bool
        # half the samples are masked, with ratios linearly spaced up to the max
        n_masked = (batch["masks"][0].sum(1) > 0).sum().item()
        assert n_masked == 4
        assert batch["masks"][0].sum(1).max() <= 8
        assert batch["labels"].tolist() == [i % 2 for i in range(8)]

    def test_multicrop_collate_without_locals(self):
        collate = MultiCropKodiakCollate(MaskingGenerator((4, 4)), (0.1, 0.5), 0.5, 16)
        batch = collate([{"global_crops": [torch.randn(3, 64, 64)] * 2, "local_crops": []}] * 2)
        assert batch["local_crops"] is None

    def test_cross_view_collate(self):
        collate = KodiakCollate(MaskingGenerator((4, 4)), (0.1, 0.5), 0.5, num_masks=2, num_patches_total=16)
        batch = collate([{"teacher_image": torch.randn(3, 64, 64), "student_image": torch.randn(3, 64, 64)}] * 4)
        assert batch["teacher_images"].shape == (4, 3, 64, 64)
        assert len(batch["masks"]) == 2 and batch["masks"][0].shape == (4, 16)

    def test_single_view_collate(self):
        collate = KodiakCollate(MaskingGenerator((4, 4)), (0.1, 0.5), 1.0, num_masks=1, num_patches_total=16)
        batch = collate([{"images": torch.randn(3, 64, 64), "label": 1}] * 2)
        assert batch["images"].shape == (2, 3, 64, 64)
        assert (batch["masks"][0].sum(1) > 0).all()

    def test_classification_collate(self):
        batch = classification_collate([{"images": torch.zeros(3, 8, 8), "label": 2}, {"images": torch.zeros(3, 8, 8)}])
        assert batch["images"].shape == (2, 3, 8, 8)
        assert batch["labels"].tolist() == [2, -1]


class TestTransforms:
    @pytest.mark.parametrize("shape", [(3, 80, 70), (1, 80, 70), (80, 70), (4, 80, 70)])
    def test_classification_transforms_handle_channel_layouts(self, shape):
        x = torch.rand(*shape)
        assert ClassificationTrainTransform(32)(x).shape == (3, 32, 32)
        assert ClassificationValTransform(32)(x).shape == (3, 32, 32)

    def test_val_transform_is_deterministic_and_normalised(self):
        x = torch.rand(3, 40, 40)
        t = ClassificationValTransform(32)
        torch.testing.assert_close(t(x), t(x))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        torch.testing.assert_close(t(torch.zeros(3, 32, 32)), (0 - mean) / std + torch.zeros(3, 32, 32))

    def test_multicrop_transform(self):
        t = MultiCropPretrainingTransform({"global_crops_size": 32, "local_crops_size": 16, "local_crops_number": 3})
        out = t(torch.rand(3, 50, 50))
        assert len(out["global_crops"]) == 2 and out["global_crops"][0].shape == (3, 32, 32)
        assert len(out["local_crops"]) == 3 and out["local_crops"][0].shape == (3, 16, 16)

    def test_two_view_transform(self):
        out = DINOv3PretrainingTransform({"global_crops_size": 32})(torch.rand(1, 50, 50))
        assert out["teacher_crop"].shape == (3, 32, 32) and out["student_crop"].shape == (3, 32, 32)

    def test_shared_color_jitter_path(self):
        aug = DataAugmentationDINO(global_crops_size=32, local_crops_size=16, local_crops_number=1,
                                   share_color_jitter=True)
        out = aug(torch.rand(3, 40, 40))
        assert out["global_crops"][1].shape == (3, 32, 32) and len(out["local_crops"]) == 1


class TestCustomPatchDataset:
    def test_flat_folder(self, image_folder):
        ds = CustomPatchDataset(root_dir=str(image_folder))
        assert len(ds) == 24 and ds.get_num_classes() == 3
        item = ds[0]
        assert item["image"].mode == "RGB" and item["label"] == 0
        assert ds.get_label_name(2) == "class_2"
        assert ds.get_label_name(99).startswith("unknown")
        assert sorted(set(ds.labels)) == [0, 1, 2]

    def test_nested_folder(self, nested_image_folder):
        ds = CustomPatchDataset(root_dir=str(nested_image_folder))
        assert len(ds) == 10 and ds.get_num_classes() == 2
        assert ds.label_to_idx == {"A": 0, "B": 1}

    def test_csv_mode(self, tmp_path, image_folder):
        paths = sorted(image_folder.rglob("*.png"))
        df = pd.DataFrame({"image_path": [str(p) for p in paths]})
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        ds = CustomPatchDataset(csv_file=str(csv))  # labels inferred from parent folder
        assert len(ds) == 24 and ds.get_num_classes() == 3

        df2 = pd.DataFrame({"image_path": [p.relative_to(image_folder).as_posix() for p in paths],
                            "label": ["x" if i < 12 else "y" for i in range(len(paths))],
                            "magnification": ["10X"] * 12 + ["40X"] * 12})
        csv2 = tmp_path / "data2.csv"
        df2.to_csv(csv2, index=False)
        ds2 = CustomPatchDataset(csv_file=str(csv2), root_path=str(image_folder), magnification_filter="10X")
        assert len(ds2) == 12 and ds2.get_num_classes() == 1
        assert ds2[0]["image"].size[0] > 0

    def test_errors(self, tmp_path):
        with pytest.raises(ValueError):
            CustomPatchDataset()
        with pytest.raises(FileNotFoundError):
            CustomPatchDataset(root_dir=str(tmp_path / "missing"))
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError):
            CustomPatchDataset(root_dir=str(empty))


class TestRegistry:
    def test_builtin_registered(self):
        assert DatasetRegistry.is_registered("pancreatic") and DatasetRegistry.is_registered("huggingface")

    def test_unknown_type_message(self):
        with pytest.raises(ValueError, match="Unknown dataset type"):
            DatasetRegistry.create("nope")

    def test_bad_arguments(self):
        with pytest.raises(TypeError, match="Failed to create dataset"):
            DatasetRegistry.create("pancreatic", bogus=1)


class TestDataModules:
    def test_classification_datamodule_custom_folder(self, image_folder):
        dm = ClassificationDataModule(dataset_type="pancreatic", root_dir=str(image_folder), batch_size=4,
                                      num_workers=0, image_size=32, train_split=0.5, val_split=0.25, test_split=0.25)
        dm.setup()
        assert (len(dm.train_dataset), len(dm.val_dataset), len(dm.test_dataset)) == (12, 6, 6)
        batch = next(iter(dm.train_dataloader()))
        assert batch["images"].shape == (4, 3, 32, 32) and batch["labels"].shape == (4,)
        assert len(dm.val_dataloader()) == 2 and len(dm.test_dataloader()) == 2

    def test_classification_split_is_seeded(self, image_folder):
        def idx(seed):
            dm = ClassificationDataModule(dataset_type="pancreatic", root_dir=str(image_folder), batch_size=4,
                                          num_workers=0, image_size=32, seed=seed)
            dm.setup()
            return list(dm.test_dataset.base.indices)
        assert idx(1) == idx(1) and idx(1) != idx(2)

    @pytest.mark.parametrize("multi_crop", [True, False])
    def test_pretraining_datamodule(self, image_folder, multi_crop):
        dm = PretrainingDataModule(dataset_type="pancreatic", root_dir=str(image_folder), batch_size=4,
                                   num_workers=0, augmentation={"global_crops_size": 64}, multi_crop=multi_crop,
                                   local_crops_number=2, local_crops_size=32, sampler_type=SamplerType.EPOCH)
        dm.setup()
        assert len(dm.train_dataset) == 24
        batch = next(iter(dm.train_dataloader()))
        if multi_crop:
            assert batch["global_crops"][0].shape == (4, 3, 64, 64)
            assert len(batch["local_crops"]) == 2 and batch["local_crops"][0].shape == (4, 3, 32, 32)
        else:
            assert batch["teacher_images"].shape == (4, 3, 64, 64)
        assert batch["masks"][0].shape == (4, 16)


class TestCheckpointSafety:
    """Checkpoints must load with torch.load(weights_only=True), the default in recent torch/Lightning."""

    def test_datamodule_hparams_contain_no_custom_objects(self, image_folder, tmp_path):
        import inspect
        if "weights_only" not in inspect.signature(torch.load).parameters:
            pytest.skip("torch too old for weights_only")
        for dm in [PretrainingDataModule(dataset_type="pancreatic", root_dir=str(image_folder), num_workers=0),
                   ClassificationDataModule(dataset_type="pancreatic", root_dir=str(image_folder), num_workers=0)]:
            assert "sampler_type" not in dm.hparams
            path = tmp_path / f"{type(dm).__name__}.pt"
            torch.save({"datamodule_hyper_parameters": dict(dm.hparams)}, path)
            torch.load(path, weights_only=True)  # must not raise

    def test_legacy_enum_in_checkpoint_is_allowlisted(self, tmp_path):
        import inspect
        if "weights_only" not in inspect.signature(torch.load).parameters or not hasattr(
                torch.serialization, "add_safe_globals"):
            pytest.skip("torch too old for safe globals")
        path = tmp_path / "legacy.pt"
        torch.save({"sampler_type": SamplerType.DISTRIBUTED}, path)
        assert torch.load(path, weights_only=True)["sampler_type"] is SamplerType.DISTRIBUTED


class TestSamplers:
    def test_epoch_sampler_covers_dataset(self):
        s = EpochSampler(size=10, sample_count=10, shuffle=True, seed=3, start=0, step=1)
        assert len(s) == 10
        assert sorted(s) == list(range(10))
        s.set_epoch(1)
        second = list(s)
        assert sorted(second) == list(range(10))

    def test_epoch_sampler_sharding(self):
        a = list(EpochSampler(size=10, sample_count=10, shuffle=False, start=0, step=2))
        b = list(EpochSampler(size=10, sample_count=10, shuffle=False, start=1, step=2))
        assert a == [0, 2, 4, 6, 8] and b == [1, 3, 5, 7, 9]

    def test_sharded_infinite_sampler_is_infinite_and_disjoint(self):
        import itertools
        a = ShardedInfiniteSampler(sample_count=8, shuffle=True, seed=0, start=0, step=2)
        b = ShardedInfiniteSampler(sample_count=8, shuffle=True, seed=0, start=1, step=2)
        ia = list(itertools.islice(iter(a), 4))
        ib = list(itertools.islice(iter(b), 4))
        assert sorted(ia + ib) == list(range(8))
        assert len(list(itertools.islice(iter(a), 20))) == 20

    def test_make_sampler_fallbacks(self):
        ds = list(range(5))
        assert isinstance(make_sampler(dataset=ds, sampler_type=SamplerType.DISTRIBUTED), EpochSampler)
        assert isinstance(make_sampler(dataset=ds, sampler_type=SamplerType.EPOCH, size=3), EpochSampler)
        assert make_sampler(dataset=ds, sampler_type=None) is None
        with pytest.raises(ValueError):
            make_sampler(dataset=ds, sampler_type=SamplerType.INFINITE, size=3)


class TestHuggingFaceCompat:
    def test_hf_call_falls_back_when_trust_remote_code_unsupported(self):
        calls = []

        def new_api(name, split=None):
            calls.append(("new", name, split))
            return "ok"

        def old_api(name, split=None, trust_remote_code=False):
            calls.append(("old", name, split, trust_remote_code))
            return "ok"

        assert hf_dataset._hf_call(new_api, "ds", split="train") == "ok"
        assert hf_dataset._hf_call(old_api, "ds", split="train") == "ok"
        assert calls == [("new", "ds", "train"), ("old", "ds", "train", True)]

    def test_hf_call_propagates_other_type_errors(self):
        def broken(name, trust_remote_code=False):
            raise TypeError("something else")

        with pytest.raises(TypeError, match="something else"):
            hf_dataset._hf_call(broken, "ds")

    def test_split_detection(self, monkeypatch):
        monkeypatch.setattr(hf_dataset, "HF_AVAILABLE", True)
        monkeypatch.setattr(hf_dataset, "get_dataset_split_names", lambda name, **kw: ["train", "validation"])
        info = hf_dataset.get_available_splits("dummy")
        assert info == {"train": True, "val": True, "test": False, "available": ["train", "validation"],
                        "val_key": "validation"}
