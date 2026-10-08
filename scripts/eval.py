"""
eval.py

Standalone evaluation script for classification models.
Runs test evaluation on a trained checkpoint and saves results to JSON.

Usage:
    python scripts/eval.py \
        --config configs/DTD/classify.yaml \
        --checkpoint output/classification/dtd/finetune_seed_0/checkpoints \
        --checkpoint_type best \
        --output_dir output/classification/dtd/finetune_seed_0/results
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch

# Add repo root to import path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.learners import ClassificationLearner
from src.data import ClassificationDataModule
from src.data.utils import SamplerType
from src.utils.config import load_config, override_config
from src.utils.runtime import resolve_accelerator_and_devices, count_devices


def find_checkpoint(path: str, checkpoint_type: str = "best") -> Optional[str]:
    """Find checkpoint file from a path (file or directory)."""
    path = Path(path)

    if path.is_file():
        return str(path)

    if path.is_dir():
        if checkpoint_type == "best":
            # Look for best.ckpt first
            best_ckpt = path / "best.ckpt"
            if best_ckpt.exists():
                return str(best_ckpt)
            # Try to find checkpoint with best val_loss in filename
            ckpts = list(path.glob("classifier-*.ckpt"))
            if ckpts:
                best_ckpt = None
                best_loss = float('inf')
                for ckpt in ckpts:
                    match = re.search(r'val_loss[=_](\d+\.?\d*)', ckpt.name)
                    if match:
                        loss = float(match.group(1))
                        if loss < best_loss:
                            best_loss = loss
                            best_ckpt = ckpt
                if best_ckpt:
                    return str(best_ckpt)

        # Fallback to last.ckpt
        last_ckpt = path / "last.ckpt"
        if last_ckpt.exists():
            return str(last_ckpt)

        # Fallback to most recent checkpoint
        ckpts = list(path.glob("*.ckpt"))
        if ckpts:
            return str(sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1])

    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="Run test evaluation on a trained classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint file or directory")
    p.add_argument("--checkpoint_type", type=str, choices=["best", "last"], default="best",
                   help="Which checkpoint to load")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save results JSON")
    p.add_argument("--devices", type=str, default="0",
                   help="GPU devices (e.g., '0' or '0,1')")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size")
    p.add_argument("--root_dir", type=str, default=None,
                   help="Root directory of a folder-based (custom) dataset (overrides data.root_dir)")
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader workers (overrides data.num_workers)")

    return p.parse_args()


def cfg_to_classification_kwargs(cfg):
    """Map YAML config structure to ClassificationLearner kwargs."""
    m = cfg.model
    opt = cfg.optimizer
    tr = cfg.training
    d = cfg.data

    num_classes = d.get("num_classes", None) or m.get("num_classes", None)
    if num_classes is None:
        raise ValueError("num_classes must be specified")

    try:
        loss_cfg = cfg.loss
        label_smoothing = loss_cfg.get("label_smoothing", 0.0) if loss_cfg else 0.0
    except AttributeError:
        label_smoothing = 0.0

    return dict(
        num_classes=num_classes,
        img_size=m.get("image_size", 256),
        patch_size=m.get("patch_size", m.get("vit_patch_size", 16)),
        embed_dim=m.get("embed_dim", m.get("vit_embed_dim", 384)),
        vit_depth=m.get("depth", m.get("vit_depth", 12)),
        vit_heads=m.get("num_heads", m.get("vit_heads", 6)),
        mlp_ratio=m.get("mlp_ratio", m.get("vit_mlp_ratio", 4.0)),
        num_storage_tokens=m.get("num_storage_tokens", 4),
        pretrained_path=None,  # We'll load from checkpoint
        lr=opt.get("learning_rate", opt.get("lr", 1e-4)),
        weight_decay=opt.get("weight_decay", 0.04),
        warmup_epochs=opt.get("warmup_epochs", 5),
        max_epochs=tr.max_epochs,
        freeze_backbone=m.get("freeze_backbone", False),
        backbone_lr_scale=m.get("backbone_lr_scale", opt.get("backbone_lr_scale", 0.1)),
        use_cls_token=m.get("use_cls_token", True),
        concat_cls_patch=m.get("concat_cls_patch", False),
        label_smoothing=label_smoothing,
        encoder_type=m.get("encoder_type", "teacher"),
    )


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # CLI overrides
    overrides = {}
    if args.batch_size is not None:
        overrides["data.batch_size"] = args.batch_size
    if args.root_dir is not None:
        overrides["data.root_dir"] = args.root_dir
    if args.num_workers is not None:
        overrides["data.num_workers"] = args.num_workers
    if overrides:
        cfg = override_config(cfg, overrides)

    # Parse devices
    device_list = [int(x.strip()) for x in args.devices.split(",")]
    accelerator, devices = resolve_accelerator_and_devices(device_list)

    # Find checkpoint
    ckpt_path = find_checkpoint(args.checkpoint, args.checkpoint_type)
    if ckpt_path is None:
        print(f"ERROR: No checkpoint found at {args.checkpoint}")
        sys.exit(1)

    print(f"Using checkpoint: {ckpt_path}")

    # Set seed
    pl.seed_everything(cfg.experiment.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-GPU batch size
    num_gpus = count_devices(devices)
    per_gpu_batch_size = max(1, cfg.data.batch_size // num_gpus)

    # Get augmentation config
    aug_config = cfg.data.get("augmentation", None)
    if aug_config is not None and hasattr(aug_config, 'to_dict'):
        aug = aug_config.to_dict()
    else:
        aug = aug_config if isinstance(aug_config, dict) else {}

    image_size = cfg.data.get("image_size", aug.get("global_crops_size", 256))

    # Parse sampler type
    sampler_type_str = str(cfg.data.get("sampler_type", "distributed")).upper()
    sampler_type = SamplerType[sampler_type_str]

    # Create datamodule
    datamodule = ClassificationDataModule(
        dataset_type=cfg.data.get("dataset_type", "huggingface"),
        batch_size=per_gpu_batch_size,
        num_workers=cfg.data.get("num_workers", 8),
        pin_memory=cfg.data.get("pin_memory", True),
        persistent_workers=cfg.data.get("persistent_workers", True),
        train_split=cfg.data.get("train_split", 0.7),
        val_split=cfg.data.get("val_split", 0.1),
        test_split=cfg.data.get("test_split", 0.2),
        image_size=image_size,
        normalize_mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
        normalize_std=aug.get("normalize_std", [0.229, 0.224, 0.225]),
        sampler_type=sampler_type,
        seed=cfg.experiment.seed,
        csv_path=cfg.data.get("csv_path", None),
        root_dir=cfg.data.get("root_dir", None),
        magnification=cfg.data.get("magnification", None),
        root_path=cfg.data.get("root_path", None),
        hf_dataset_name=cfg.data.get("hf_dataset_name", None),
        hf_split=cfg.data.get("hf_split", None),
        hf_cache_dir=cfg.data.get("hf_cache_dir", None),
    )

    # Create model
    classification_kwargs = cfg_to_classification_kwargs(cfg)
    model = ClassificationLearner(**classification_kwargs)

    # Create trainer for evaluation
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=cfg.training.get("precision", "bf16-mixed"),
        logger=False,
        enable_progress_bar=True,
    )

    # Run test evaluation
    print("\n" + "=" * 60)
    print("Running Test Evaluation")
    print("=" * 60)
    print(f"Dataset: {cfg.data.get('name', 'unknown')}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Devices: {devices}")
    print("=" * 60 + "\n")

    test_results = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Save results
    if test_results:
        dataset_name = cfg.data.get("name", "unknown")
        mode = "lineareval" if cfg.model.get("freeze_backbone", False) else "finetune"

        results = {
            "dataset": dataset_name,
            "mode": mode,
            "seed": cfg.experiment.seed,
            "checkpoint": ckpt_path,
            "test_acc": float(test_results[0].get("test_acc", 0)),
            "test_loss": float(test_results[0].get("test_loss", 0)),
        }

        # Add additional metrics if available
        for key, value in test_results[0].items():
            if key not in results and key.startswith("test_"):
                results[key] = float(value)

        results_file = output_dir / "single_eval.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print(f"Test acc:   {results['test_acc']:.4f}")
        print(f"Test F1:    {results.get('test_f1_macro', 'N/A'):.4f}" if isinstance(results.get('test_f1_macro'), float) else "Test F1:    N/A")
        print(f"Test AUROC: {results.get('test_auroc', 'N/A'):.4f}" if isinstance(results.get('test_auroc'), float) else "Test AUROC: N/A")
        print(f"Test loss:  {results['test_loss']:.4f}")
        print(f"Results saved to: {results_file}")
        print("=" * 60)
    else:
        print("WARNING: No test results returned")
        sys.exit(1)


if __name__ == "__main__":
    main()
