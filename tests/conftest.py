"""Shared fixtures for the KODIAK test suite.

All tests run on CPU with tiny models so the whole suite finishes in a few
minutes without a GPU or any external dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Tiny ViT used across tests (must stay consistent between model and configs)
TINY = dict(
    img_size=64,
    patch_size=16,
    embed_dim=32,
    depth=1,
    num_heads=2,
    num_storage_tokens=4,
    num_prototypes=16,
    projector_dim=16,
    decoder_embed_dim=16,
    decoder_depth=1,
    decoder_num_heads=2,
    local_crops_size=32,
    local_crops_number=2,
)
NUM_CLASSES = 3
IMAGES_PER_CLASS = 8


@pytest.fixture(scope="session")
def tiny_kwargs():
    return dict(TINY)


def _write_images(root: Path, classes, per_class: int, size: int = 72, seed: int = 0):
    rng = np.random.default_rng(seed)
    for c in classes:
        d = root / c
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"img_{i}.png")


@pytest.fixture(scope="session")
def image_folder(tmp_path_factory) -> Path:
    """Flat folder dataset: root/<class>/<image>.png with 3 classes x 8 images."""
    root = tmp_path_factory.mktemp("flat_dataset")
    _write_images(root, [f"class_{i}" for i in range(NUM_CLASSES)], IMAGES_PER_CLASS)
    return root


@pytest.fixture(scope="session")
def nested_image_folder(tmp_path_factory) -> Path:
    """Nested folder dataset: root/<slide>/<class>/<image>.png (two slides)."""
    root = tmp_path_factory.mktemp("nested_dataset")
    _write_images(root / "SLIDE-1", ["A", "B"], 3, seed=1)
    _write_images(root / "SLIDE-2", ["A", "B"], 2, seed=2)
    return root


@pytest.fixture
def tiny_kodiak(tiny_kwargs):
    from src.models import Kodiak

    torch.manual_seed(0)
    k = tiny_kwargs
    return Kodiak(
        num_prototypes=k["num_prototypes"],
        projector_dim=k["projector_dim"],
        img_size=k["img_size"],
        patch_size=k["patch_size"],
        embed_dim=k["embed_dim"],
        depth=k["depth"],
        num_heads=k["num_heads"],
        num_storage_tokens=k["num_storage_tokens"],
        decoder_embed_dim=k["decoder_embed_dim"],
        decoder_depth=k["decoder_depth"],
        decoder_num_heads=k["decoder_num_heads"],
    )


def _tiny_pretrain_overrides(root_dir: Path, out_dir: Path, max_epochs: int = 2):
    k = TINY
    return {
        "experiment": {"name": "tiny", "seed": 0},
        "data": {
            "root_dir": str(root_dir),
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "multi_crop": True,
            "local_crops_number": k["local_crops_number"],
            "local_crops_size": k["local_crops_size"],
            "augmentation": {
                "global_crops_size": k["img_size"],
                "local_crops_number": k["local_crops_number"],
                "local_crops_size": k["local_crops_size"],
            },
        },
        "model": {
            "image_size": k["img_size"],
            "vit_embed_dim": k["embed_dim"],
            "vit_depth": k["depth"],
            "vit_heads": k["num_heads"],
            "num_storage_tokens": k["num_storage_tokens"],
            "num_prototypes": k["num_prototypes"],
            "projector_dim": k["projector_dim"],
            "decoder_dim": k["decoder_embed_dim"],
            "decoder_depth": k["decoder_depth"],
            "decoder_heads": k["decoder_num_heads"],
            "local_crop_size": k["local_crops_size"],
            "num_local_crops": k["local_crops_number"],
        },
        "loss": {"n_local_crops": k["local_crops_number"]},
        "optimizer": {"warmup_epochs": 1, "teacher_temp_warmup_epochs": 1, "freeze_last_layer_epochs": 1},
        "training": {
            "max_epochs": max_epochs,
            "precision": "32",
            "devices": [0],
            "checkpoint": {"base_dir": str(out_dir / "checkpoints")},
        },
        "logging": {"base_dir": str(out_dir / "logs"), "log_every_n_steps": 1},
    }


def _deep_update(base: dict, upd: dict) -> dict:
    for key, value in upd.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@pytest.fixture(scope="session")
def tiny_pretrain_config(tmp_path_factory, image_folder) -> Path:
    """A tiny pretraining config derived from the shipped pancreatic config."""
    out_dir = tmp_path_factory.mktemp("pretrain_out")
    with open(ROOT / "configs/pancreatic/pretrain.yaml") as f:
        cfg = yaml.safe_load(f)
    _deep_update(cfg, _tiny_pretrain_overrides(image_folder, out_dir))
    path = out_dir / "pretrain_tiny.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


@pytest.fixture(scope="session")
def tiny_classify_config(tmp_path_factory, image_folder) -> Path:
    """A tiny classification config derived from the shipped pancreatic config."""
    out_dir = tmp_path_factory.mktemp("classify_out")
    with open(ROOT / "configs/pancreatic/classify.yaml") as f:
        cfg = yaml.safe_load(f)
    k = TINY
    _deep_update(cfg, {
        "experiment": {"name": "tiny_classify", "seed": 0},
        "data": {
            "root_dir": str(image_folder),
            "batch_size": 4,
            "num_workers": 0,
            "image_size": k["img_size"],
            "num_classes": NUM_CLASSES,
            "train_split": 0.7,
            "val_split": 0.1,
            "test_split": 0.2,
        },
        "model": {
            "image_size": k["img_size"],
            "embed_dim": k["embed_dim"],
            "depth": k["depth"],
            "num_heads": k["num_heads"],
            "num_storage_tokens": k["num_storage_tokens"],
            "num_prototypes": k["num_prototypes"],
            "projector_dim": k["projector_dim"],
            "decoder_dim": k["decoder_embed_dim"],
            "decoder_depth": k["decoder_depth"],
            "decoder_heads": k["decoder_num_heads"],
        },
        "optimizer": {"warmup_epochs": 1},
        "training": {"max_epochs": 2, "precision": "32", "devices": [0], "early_stopping_patience": 5},
        "logging": {"base_dir": str(out_dir / "logs"), "checkpoint_dir": str(out_dir / "checkpoints")},
    })
    path = out_dir / "classify_tiny.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path
