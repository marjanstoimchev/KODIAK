"""Tests for src.utils (config, validation, dataclasses, loggers, runtime) and callbacks."""

import os

import pytest
import torch

from src.callbacks import TrainingTimer
from src.utils import (
    AugmentationConfig,
    Config,
    DataConfig,
    LossConfig,
    MaskingConfig,
    ModelConfig,
    TrainingConfig,
    check_config,
    load_config,
    override_config,
    save_config,
    setup_logger,
    validate_config,
)
from src.utils.config import expand_env_vars
from src.utils.runtime import count_devices, resolve_accelerator_and_devices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestConfig:
    def test_dot_access_and_roundtrip(self, tmp_path):
        cfg = Config({"a": {"b": 1, "c": [1, 2]}, "d": "x"})
        assert cfg.a.b == 1 and cfg["d"] == "x" and cfg.a.get("missing", 5) == 5
        cfg.a.b = 2
        assert cfg.to_dict() == {"a": {"b": 2, "c": [1, 2]}, "d": "x"}
        path = tmp_path / "sub" / "cfg.yaml"
        save_config(cfg, path)
        assert load_config(path).to_dict() == cfg.to_dict()

    def test_override_creates_nested_keys(self):
        cfg = Config({"model": {"num_prototypes": 4096}, "flat": 1})
        new = override_config(cfg, {"model.num_prototypes": 128, "new.nested.key": "v", "flat.now_nested": 2})
        assert new.model.num_prototypes == 128 and new.new.nested.key == "v" and new.flat.now_nested == 2
        assert cfg.model.num_prototypes == 4096  # original untouched

    def test_env_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KODIAK_TEST_DIR", "/data/root")
        assert expand_env_vars({"a": "${KODIAK_TEST_DIR}/x", "b": ["$KODIAK_TEST_DIR", 3], "c": 1}) == {
            "a": "/data/root/x", "b": ["/data/root", 3], "c": 1}
        p = tmp_path / "c.yaml"
        p.write_text('data:\n  root_dir: "${KODIAK_TEST_DIR}/pancreatic"\n  home: "~/x"\n')
        cfg = load_config(p)
        assert cfg.data.root_dir == "/data/root/pancreatic"
        assert cfg.data.home == os.path.expanduser("~/x")

    def test_load_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "missing.yaml")
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ValueError):
            load_config(empty)

    @pytest.mark.parametrize("path", [
        "configs/cifar100/pretrain.yaml", "configs/cifar100/pretrain_continued.yaml", "configs/cifar100/classify.yaml",
        "configs/DTD/pretrain.yaml", "configs/eurosat/classify.yaml", "configs/oxford_pets/pretrain_continued.yaml",
        "configs/NCTCRCHE100K/classify.yaml", "configs/imagenet1k/pretrain.yaml", "configs/pancreatic/pretrain.yaml",
    ])
    def test_shipped_configs_parse(self, path):
        cfg = load_config(os.path.join(ROOT, path))
        assert cfg.experiment.name and cfg.data.dataset_type in ("huggingface", "pancreatic")
        if cfg.data.dataset_type == "huggingface":
            assert validate_config(cfg.to_dict()) == []


class TestValidation:
    def test_detects_problems(self, tmp_path):
        errors = validate_config({
            "experiment": {"name": "bad/name"},
            "data": {"dataset_type": "pancreatic", "root_dir": str(tmp_path / "nope"), "train_split": 0.9,
                     "val_split": 0.2, "batch_size": 0},
            "model": {"image_size": 250, "vit_patch_size": 16, "vit_embed_dim": 384, "vit_heads": 7,
                      "mask_strategy": "random"},
        })
        joined = "\n".join(errors)
        for needle in ["invalid characters", "not found", "<= 1.0", "batch_size", "divisible by vit_patch_size",
                       "divisible by vit_heads", "mask_strategy"]:
            assert needle in joined, needle
        assert check_config({"experiment": {"name": "ok"}}) is True
        assert check_config({"experiment": {"name": ""}}) is False

    def test_huggingface_requires_name(self):
        assert any("hf_dataset_name" in e for e in validate_config({"data": {"dataset_type": "huggingface"}}))


class TestDataclasses:
    def test_model_config(self):
        assert ModelConfig().num_patches == 256
        with pytest.raises(ValueError):
            ModelConfig(img_size=250)

    def test_training_config(self):
        TrainingConfig()
        with pytest.raises(ValueError):
            TrainingConfig(min_lr=1.0)
        with pytest.raises(ValueError):
            TrainingConfig(lr_scaling="linear")

    def test_other_configs(self, tmp_path):
        with pytest.raises(ValueError):
            LossConfig(teacher_temp=0)
        with pytest.raises(ValueError):
            AugmentationConfig(global_crops_scale=(0.5, 0.2))
        with pytest.raises(ValueError):
            MaskingConfig(mask_ratio_tuple=(0.6, 0.5))
        with pytest.raises(ValueError):
            DataConfig(dataset_type="huggingface")
        with pytest.raises(FileNotFoundError):
            DataConfig(dataset_type="pancreatic", root_dir=str(tmp_path / "nope"))
        DataConfig(dataset_type="pancreatic", root_dir=str(tmp_path))


class TestRuntime:
    @pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only expectations")
    def test_cpu_fallback(self):
        assert resolve_accelerator_and_devices([0, 1]) == ("cpu", 1)
        assert resolve_accelerator_and_devices("auto") == ("cpu", 1)
        assert resolve_accelerator_and_devices(None) == ("cpu", 1)
        assert resolve_accelerator_and_devices(2) == ("cpu", 2)

    def test_count_devices(self):
        assert count_devices([0, 1, 2]) == 3
        assert count_devices(2) == 2
        assert count_devices("auto") >= 1


class TestLoggers:
    def test_csv_logger(self, tmp_path):
        logger = setup_logger("csv", "exp", save_dir=str(tmp_path / "logs"))
        assert logger.name == "exp"

    def test_unknown_logger(self, tmp_path):
        with pytest.raises(ValueError):
            setup_logger("mlflow", "exp", save_dir=str(tmp_path))
        with pytest.raises(ValueError):
            setup_logger("wandb", "exp", save_dir=str(tmp_path))  # needs a project


class TestCallbacks:
    def test_format_time(self):
        assert TrainingTimer._format_time(5) == "5s"
        assert TrainingTimer._format_time(125) == "2m 5s"
        assert TrainingTimer._format_time(3725) == "1h 2m 5s"
