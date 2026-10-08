"""
train_classifier.py

Train a classifier using a pretrained SSL encoder.
Supports both fine-tuning (trainable backbone) and linear evaluation (frozen backbone).

Output directory structure (automatically includes pretraining config):
  logs/classification/{dataset}/{pretrain_info}/{mode}_seed_{seed}/
  checkpoints/classification/{dataset}/{pretrain_info}/{mode}_seed_{seed}/

Example:
  logs/classification/dtd/proto2048_koleo0.1_cls1_mc/finetune_seed_0/
  logs/classification/dtd/proto2048_koleo0.1_cls1_mc/finetune_seed_1/
  logs/classification/dtd/proto128_koleo0.1_cls0.5_mc/finetune_seed_0/

The pretrain_info folder name is auto-extracted from checkpoint hyperparameters.

Usage examples:

  # Fine-tuning (trainable backbone) - auto-find last checkpoint
  python scripts/train_classifier.py \
      --config configs/eurosat/classify.yaml \
      --pretrained_path output/checkpoints/eurosat/pretraining

  # Use best checkpoint instead of last
  python scripts/train_classifier.py \
      --config configs/eurosat/classify.yaml \
      --pretrained_path output/checkpoints/eurosat/pretraining \
      --checkpoint_type best

  # Linear evaluation (frozen backbone)
  python scripts/train_classifier.py \
      --config configs/eurosat/classify.yaml \
      --pretrained_path path/to/checkpoint.ckpt \
      --freeze_backbone

  # Override directories
  python scripts/train_classifier.py \
      --config configs/eurosat/classify.yaml \
      --log_base_dir logs/my_experiment \
      --checkpoint_base_dir checkpoints/my_experiment
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.strategies import DDPStrategy
import torch

# Add repo root to import path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.learners import ClassificationLearner
from src.data import ClassificationDataModule
from src.data.utils import SamplerType
from src.utils.config import load_config, override_config, save_config
from src.utils.config_validation import check_config
from src.utils.loggers import get_logger_from_config
from src.utils.runtime import resolve_accelerator_and_devices, count_devices
from src.callbacks import TrainingTimer


# ---------------------------------------------------------
# Checkpoint Resolution
# ---------------------------------------------------------
def extract_pretraining_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Extract pretraining hyperparameters from a checkpoint file.

    Returns dict with keys like:
        - num_prototypes
        - koleo_loss_weight
        - multi_crop (bool)
        - projector_dim
        - etc.
    """
    info = {}

    try:
        # Load checkpoint metadata only (not full model weights)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract hyperparameters
        hparams = ckpt.get("hyper_parameters", {})

        # Key pretraining params
        info["num_prototypes"] = hparams.get("num_prototypes", None)
        info["koleo_loss_weight"] = hparams.get("koleo_loss_weight", None)
        info["cls_loss_weight"] = hparams.get("prototype_cls_loss_weight", hparams.get("cls_loss_weight", None))
        info["projector_dim"] = hparams.get("projector_dim", None)
        info["multi_crop"] = hparams.get("multi_crop", False)
        info["n_local_crops"] = hparams.get("n_local_crops", 0)
        # The pretraining learner does not store `multi_crop`; infer it from n_local_crops
        if not info["multi_crop"] and info["n_local_crops"] and info["n_local_crops"] > 0:
            info["multi_crop"] = True
        info["mask_ratio"] = hparams.get("mask_ratio", None)
        info["patch_loss_weight"] = hparams.get("patch_loss_weight", None)
        info["prototype_loss_weight"] = hparams.get("prototype_loss_weight", None)

        # Also try to get from nested structure if available
        if not info["num_prototypes"]:
            model_cfg = hparams.get("model", {})
            if isinstance(model_cfg, dict):
                info["num_prototypes"] = model_cfg.get("num_prototypes", None)

    except Exception as e:
        print(f"Warning: Could not extract pretraining info from checkpoint: {e}")

    return info


def build_pretraining_folder_name(pretrain_info: Dict[str, Any]) -> str:
    """
    Build a folder name from pretraining info.

    Returns something like: proto2048_koleo0.1_cls1_mc8
    """
    parts = []

    # Prototypes
    num_proto = pretrain_info.get("num_prototypes")
    if num_proto is not None:
        parts.append(f"proto{num_proto}")

    # KoLeo weight
    koleo = pretrain_info.get("koleo_loss_weight")
    if koleo is not None:
        # Format nicely (0.1 not 0.10000000149011612)
        if isinstance(koleo, float):
            koleo_str = f"{koleo:.4g}"  # Use general format, removes trailing zeros
        else:
            koleo_str = str(koleo)
        parts.append(f"koleo{koleo_str}")

    # CLS loss weight
    cls_weight = pretrain_info.get("cls_loss_weight")
    if cls_weight is not None:
        if isinstance(cls_weight, float):
            cls_str = f"{cls_weight:.4g}"
        else:
            cls_str = str(cls_weight)
        parts.append(f"cls{cls_str}")

    # Multi-crop flag (with number of local crops when known)
    if pretrain_info.get("multi_crop", False):
        n_local = pretrain_info.get("n_local_crops", 0)
        parts.append(f"mc{n_local}" if n_local else "mc")

    # If we couldn't extract any info, use "unknown"
    if not parts:
        return "unknown_pretraining"

    return "_".join(parts)


def find_checkpoint(path: str, checkpoint_type: str = "last") -> Optional[str]:
    """
    Find checkpoint file from a path (file or directory).

    Args:
        path: Path to checkpoint file or directory containing checkpoints
        checkpoint_type: "last" for last.ckpt, "best" for lowest train_loss

    Returns:
        Path to checkpoint file, or None if not found
    """
    path = Path(path)

    # If it's already a file, return it
    if path.is_file():
        return str(path)

    # If it's a directory, find the appropriate checkpoint
    if path.is_dir():
        if checkpoint_type == "last":
            # Look for last.ckpt
            last_ckpt = path / "last.ckpt"
            if last_ckpt.exists():
                return str(last_ckpt)
            # Fallback: most recently modified .ckpt
            ckpts = list(path.glob("*.ckpt"))
            if ckpts:
                return str(sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1])

        elif checkpoint_type == "best":
            # Find checkpoint with lowest train_loss in filename
            # Format: kodiak-epoch=XX-train_loss=Y.YYYY.ckpt
            ckpts = list(path.glob("kodiak-*.ckpt"))
            if not ckpts:
                ckpts = list(path.glob("*.ckpt"))

            best_ckpt = None
            best_loss = float('inf')

            for ckpt in ckpts:
                if ckpt.name == "last.ckpt":
                    continue
                # Extract train_loss from filename
                match = re.search(r'train_loss[=_](\d+\.?\d*)', ckpt.name)
                if match:
                    loss = float(match.group(1))
                    if loss < best_loss:
                        best_loss = loss
                        best_ckpt = ckpt

            if best_ckpt:
                return str(best_ckpt)

            # Fallback to last.ckpt if no loss-based checkpoint found
            last_ckpt = path / "last.ckpt"
            if last_ckpt.exists():
                print("Warning: No best checkpoint found, falling back to last.ckpt")
                return str(last_ckpt)

        # Nothing directly inside: look one or more levels down (e.g. the dataset
        # folder `checkpoints/pretraining/eurosat` containing one experiment folder).
        nested = sorted(path.rglob("last.ckpt"))
        if len(nested) == 1:
            print(f"Found checkpoint in sub-folder: {nested[0]}")
            return str(nested[0])
        if len(nested) > 1:
            listing = "\n".join(f"  {c.parent}" for c in nested)
            raise FileNotFoundError(
                f"{path} contains {len(nested)} experiments; pass the experiment folder explicitly:\n{listing}"
            )

    return None


def find_classification_checkpoint(checkpoint_dir: Path) -> Optional[str]:
    """Pick the checkpoint to evaluate from a classification run directory.

    Priority: lowest ``val_loss`` encoded in a ``classifier-*.ckpt`` filename,
    then ``last.ckpt``, then the most recently modified ``*.ckpt``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None

    best_ckpt, best_loss = None, float("inf")
    for ckpt in checkpoint_dir.glob("*.ckpt"):
        if ckpt.name == "last.ckpt":
            continue
        match = re.search(r"val_loss[=_](\d+\.?\d*)", ckpt.name)
        if match and float(match.group(1)) < best_loss:
            best_loss = float(match.group(1))
            best_ckpt = ckpt
    if best_ckpt is not None:
        return str(best_ckpt)

    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)

    ckpts = list(checkpoint_dir.glob("*.ckpt"))
    if ckpts:
        return str(sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1])
    return None


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Linear Classifier with pretrained encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Path to config.yaml")

    # Common overrides
    p.add_argument("--batch_size", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--max_epochs", type=int)
    p.add_argument("--pretrained_path", type=str,
                   help="Path to pretrained checkpoint file or directory containing checkpoints")
    p.add_argument("--checkpoint_type", type=str, choices=["best", "last"], default="last",
                   help="Which checkpoint to load: 'last' (default) or 'best' (lowest train_loss)")
    p.add_argument("--precision", type=str)
    p.add_argument("--devices", type=str)
    p.add_argument("--logger", type=str, choices=["csv", "wandb", "tensorboard"])
    p.add_argument("--name", type=str)
    p.add_argument("--resume", type=str)
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--use_cls_token", type=str, choices=["true", "false"], default=None,
                   help="Use CLS token for classification (true) or mean pooling of patch tokens (false)")
    p.add_argument("--concat_cls_patch", action="store_true",
                   help="Concatenate CLS token + mean patch tokens for classification (2x embed_dim)")
    p.add_argument("--encoder_type", type=str, choices=["teacher", "student"], default=None,
                   help="Which encoder to load from SSL checkpoint: 'teacher' (default, recommended) or 'student'")
    p.add_argument("--exp_base_dir", type=str, help="Base directory for logs and checkpoints (creates logs/ and checkpoints/ subdirs)")
    p.add_argument("--log_dir", type=str, help="Override logging.save_dir (full path)")
    p.add_argument("--checkpoint_dir", type=str, help="Override training.checkpoint.dirpath (full path)")
    p.add_argument("--log_base_dir", type=str, help="Override logging.base_dir")
    p.add_argument("--checkpoint_base_dir", type=str, help="Override training.checkpoint.base_dir")
    p.add_argument("--seed", type=int, help="Random seed (overrides config and PL_GLOBAL_SEED)")
    p.add_argument("--root_dir", type=str, default=None,
                   help="Root directory of a folder-based (custom) dataset (overrides data.root_dir)")
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader workers (overrides data.num_workers)")
    p.add_argument("--eval_only", action="store_true",
                   help="Only run test evaluation on existing checkpoint (no training)")

    return p.parse_args()


def build_overrides(args) -> Dict[str, Any]:
    """Apply CLI overrides to YAML config."""
    o = {}
    if args.batch_size is not None:
        o["data.batch_size"] = args.batch_size
    if args.learning_rate is not None:
        o["optimizer.learning_rate"] = args.learning_rate
    if args.max_epochs is not None:
        o["training.max_epochs"] = args.max_epochs
    if args.pretrained_path is not None:
        # Resolve checkpoint path (handles both files and directories)
        resolved_path = find_checkpoint(args.pretrained_path, args.checkpoint_type)
        if resolved_path is None:
            raise FileNotFoundError(
                f"Could not find checkpoint at: {args.pretrained_path}\n"
                f"Checkpoint type: {args.checkpoint_type}"
            )
        print(f"Resolved checkpoint ({args.checkpoint_type}): {resolved_path}")
        o["model.pretrained_path"] = resolved_path
    if args.precision is not None:
        o["training.precision"] = args.precision
    if args.devices is not None:
        if args.devices.lower() == "auto":
            o["training.devices"] = "auto"
        else:
            o["training.devices"] = [int(x.strip()) for x in args.devices.split(",")]
    if args.logger is not None:
        o["logging.logger"] = args.logger
    if args.name is not None:
        o["experiment.name"] = args.name
    if args.freeze_backbone:
        o["model.freeze_backbone"] = True
    if args.use_cls_token is not None:
        o["model.use_cls_token"] = args.use_cls_token.lower() == "true"
    if args.encoder_type is not None:
        o["model.encoder_type"] = args.encoder_type
    if args.concat_cls_patch:
        o["model.concat_cls_patch"] = True

    # Directory overrides
    if args.exp_base_dir is not None:
        # exp_base_dir sets both log and checkpoint dirs
        o["logging.save_dir"] = str(Path(args.exp_base_dir) / "logs")
        o["training.checkpoint.dirpath"] = str(Path(args.exp_base_dir) / "checkpoints")
    # Base dir overrides (for structured paths: base_dir/dataset/experiment)
    if args.log_base_dir is not None:
        o["logging.base_dir"] = args.log_base_dir
    if args.checkpoint_base_dir is not None:
        o["training.checkpoint.base_dir"] = args.checkpoint_base_dir
    # Individual full-path overrides take highest precedence
    if args.log_dir is not None:
        o["logging.save_dir"] = args.log_dir
    if args.checkpoint_dir is not None:
        o["training.checkpoint.dirpath"] = args.checkpoint_dir
    if args.seed is not None:
        o["experiment.seed"] = args.seed
    if args.root_dir is not None:
        o["data.root_dir"] = args.root_dir
    if args.num_workers is not None:
        o["data.num_workers"] = args.num_workers

    return o


# ---------------------------------------------------------
# YAML → LightningModule kwargs
# ---------------------------------------------------------
def cfg_to_classification_kwargs(cfg) -> Dict[str, Any]:
    """Map YAML config structure to ClassificationLearner(**kwargs)."""

    m = cfg.model
    opt = cfg.optimizer
    tr = cfg.training
    d = cfg.data

    # num_classes can be in data or model section
    num_classes = d.get("num_classes", None) or m.get("num_classes", None)
    if num_classes is None:
        raise ValueError("num_classes must be specified in data or model section")

    # Handle optional loss config section
    try:
        loss_cfg = cfg.loss
        label_smoothing = loss_cfg.get("label_smoothing", 0.0) if loss_cfg else 0.0
    except AttributeError:
        label_smoothing = 0.0

    # Handle patch_size which may be named differently
    patch_size = m.get("patch_size", m.get("vit_patch_size", 16))

    # Handle embed_dim which may be named differently
    embed_dim = m.get("embed_dim", m.get("vit_embed_dim", 384))

    # Handle depth which may be named differently
    vit_depth = m.get("depth", m.get("vit_depth", 12))

    # Handle heads which may be named differently
    vit_heads = m.get("num_heads", m.get("vit_heads", 6))

    return dict(
        # Model
        num_classes=num_classes,
        img_size=m.get("image_size", 256),
        patch_size=patch_size,
        embed_dim=embed_dim,
        vit_depth=vit_depth,
        vit_heads=vit_heads,
        mlp_ratio=m.get("mlp_ratio", m.get("vit_mlp_ratio", 4.0)),
        num_storage_tokens=m.get("num_storage_tokens", 4),  # Must match DINOv3 pretraining

        # Pretrained weights
        pretrained_path=m.get("pretrained_path", None),

        # Training config
        lr=opt.get("learning_rate", opt.get("lr", 1e-4)),
        weight_decay=opt.get("weight_decay", 0.04),
        warmup_epochs=opt.get("warmup_epochs", 5),
        max_epochs=tr.max_epochs,

        # Backbone fine-tuning
        freeze_backbone=m.get("freeze_backbone", False),
        backbone_lr_scale=m.get("backbone_lr_scale", opt.get("backbone_lr_scale", 0.1)),
        use_cls_token=m.get("use_cls_token", True),
        concat_cls_patch=m.get("concat_cls_patch", False),
        label_smoothing=label_smoothing,
        encoder_type=m.get("encoder_type", "teacher"),  # teacher (default) or student
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()

    # 1. Load + overrides
    cfg = load_config(args.config)
    overrides = build_overrides(args)
    if overrides:
        print(f"Applying overrides: {overrides}")
        cfg = override_config(cfg, overrides)

    # Validate configuration
    if not check_config(cfg.to_dict()):
        sys.exit(1)
    print("Configuration validated successfully")

    # 2. Seed & precision
    pl.seed_everything(cfg.experiment.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # 3. Summary
    print("\n" + "=" * 80)
    print(f"Fine-tuning: {cfg.experiment.name}")
    print("=" * 80)
    num_classes = cfg.data.get("num_classes", None) or cfg.model.get("num_classes", None)
    print(f"Task:          Classification ({num_classes} classes)")
    print(f"Backbone:      ViT-S/16 (Depth={cfg.model.get('depth', cfg.model.get('vit_depth', 12))})")
    print(f"Pretrained:    {cfg.model.get('pretrained_path', 'None')}")
    print(f"Encoder Type:  {cfg.model.get('encoder_type', 'teacher')} (from SSL checkpoint)")
    print(f"Freeze:        {cfg.model.get('freeze_backbone', False)}")
    print(f"Batch Size:    {cfg.data.batch_size}")
    print(f"Learning Rate: {cfg.optimizer.get('learning_rate', cfg.optimizer.get('lr', 1e-4))}")
    print(f"Precision:     {cfg.training.get('precision', 'bf16-mixed')}")
    print(f"Devices:       {cfg.training.get('devices', 'auto')}")
    print("=" * 80 + "\n")

    # 4. DataModule (NO MASKING for classification, smart train/val/test splitting)
    # Compute per-GPU batch size (DinoV3 style: batch_size is TOTAL, divide by num GPUs)
    accelerator, devices = resolve_accelerator_and_devices(cfg.training.get("devices", "auto"))
    num_gpus = count_devices(devices)

    per_gpu_batch_size = max(1, cfg.data.batch_size // num_gpus)
    print(f"Batch size: {cfg.data.batch_size} total -> {per_gpu_batch_size} per device ({num_gpus} x {accelerator})")

    # Parse sampler_type from config (default: DISTRIBUTED)
    sampler_type_str = str(cfg.data.get("sampler_type", "distributed")).upper()
    sampler_type = SamplerType[sampler_type_str]

    # Get augmentation config with defaults
    aug_config = cfg.data.get("augmentation", None)
    if aug_config is not None and hasattr(aug_config, 'to_dict'):
        aug = aug_config.to_dict()
    else:
        aug = aug_config if isinstance(aug_config, dict) else {}

    # Get image_size from data section first, then fall back to augmentation
    image_size = cfg.data.get("image_size", aug.get("global_crops_size", 256))

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

        # Custom dataset specific
        csv_path=cfg.data.get("csv_path", None),
        root_dir=cfg.data.get("root_dir", None),
        magnification=cfg.data.get("magnification", None),
        root_path=cfg.data.get("root_path", None),

        # HF specific
        hf_dataset_name=cfg.data.get("hf_dataset_name", None),
        hf_split=cfg.data.get("hf_split", None),
        hf_cache_dir=cfg.data.get("hf_cache_dir", None),
    )

    # 5. Model
    classification_kwargs = cfg_to_classification_kwargs(cfg)
    model = ClassificationLearner(**classification_kwargs)

    # 6. Construct output directories
    # Structure: {base_dir}/{dataset}/{pretrain_info}/{mode}_seed_{seed}/
    # Example: output/classification/dtd/proto2048_koleo0.1_mc/finetune_seed_0/
    dataset_name = cfg.data.get("name", "default")
    seed = cfg.experiment.seed
    mode = "lineareval" if cfg.model.get("freeze_backbone", False) else "finetune"
    pretrained_path = cfg.model.get("pretrained_path", None)

    # Extract pretraining info from checkpoint for folder naming
    pretrain_info = {}
    pretrain_folder = "no_pretrain"
    if pretrained_path and Path(pretrained_path).exists():
        pretrain_info = extract_pretraining_info(pretrained_path)
        pretrain_folder = build_pretraining_folder_name(pretrain_info)
        print(f"Pretraining config: {pretrain_info}")
        print(f"Pretraining folder: {pretrain_folder}")

    # Build run name: finetune_seed_0 or lineareval_seed_42
    run_name = f"{mode}_seed_{seed}"

    # Log directory - CLI --log_dir takes precedence
    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
    else:
        log_base_dir = cfg.logging.get("base_dir", None)
        if log_base_dir:
            # New structure: base_dir/dataset/pretrain_folder/run_name
            log_dir = Path(log_base_dir) / dataset_name / pretrain_folder / run_name
        else:
            save_dir = cfg.logging.get("save_dir", "logs/classification")
            log_dir = Path(save_dir) / dataset_name / pretrain_folder / run_name

    # Checkpoint directory - CLI --checkpoint_dir takes precedence
    if args.checkpoint_dir is not None:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        # Handle both nested checkpoint config and flat config
        ckpt_config = cfg.training.get("checkpoint", {})
        if isinstance(ckpt_config, dict) or (hasattr(ckpt_config, 'get') and not hasattr(ckpt_config, 'to_dict')):
            ckpt_base_dir = ckpt_config.get("base_dir", None) if isinstance(ckpt_config, dict) else None
            ckpt_dirpath = ckpt_config.get("dirpath", None) if isinstance(ckpt_config, dict) else None
        else:
            ckpt_base_dir = ckpt_config.get("base_dir", None) if ckpt_config else None
            ckpt_dirpath = ckpt_config.get("dirpath", None) if ckpt_config else None

        if ckpt_base_dir:
            # New structure: base_dir/dataset/pretrain_folder/run_name
            checkpoint_dir = Path(ckpt_base_dir) / dataset_name / pretrain_folder / run_name
        elif ckpt_dirpath:
            checkpoint_dir = Path(ckpt_dirpath)
        else:
            # Fall back to logging.checkpoint_dir or default
            checkpoint_dir = Path(cfg.logging.get("checkpoint_dir", "checkpoints")) / dataset_name / pretrain_folder / run_name

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Log directory:        {log_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("=" * 80 + "\n")

    # Logger
    logger = get_logger_from_config(cfg)

    # Save config
    save_config(cfg, log_dir / "config.yaml")

    # 7. Callbacks
    # Get checkpoint config with defaults (handle both nested and flat formats)
    ckpt_cfg = cfg.training.get("checkpoint", {})
    if hasattr(ckpt_cfg, 'get'):
        ckpt_monitor = ckpt_cfg.get("monitor", "val_loss")
        ckpt_mode = ckpt_cfg.get("mode", "min")
        ckpt_save_top_k = ckpt_cfg.get("save_top_k", 1)
        ckpt_save_last = ckpt_cfg.get("save_last", True)
        ckpt_every_n_epochs = ckpt_cfg.get("every_n_epochs", None)
        ckpt_filename = ckpt_cfg.get("filename", "classifier-{epoch:02d}-{val_loss:.4f}")
    else:
        ckpt_monitor = "val_loss"
        ckpt_mode = "min"
        ckpt_save_top_k = 1
        ckpt_save_last = True
        ckpt_every_n_epochs = None
        ckpt_filename = "classifier-{epoch:02d}-{val_loss:.4f}"

    checkpoint_callback = ModelCheckpoint(
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        save_top_k=ckpt_save_top_k,
        save_last=ckpt_save_last,
        every_n_epochs=ckpt_every_n_epochs,
        dirpath=str(checkpoint_dir),
        filename=ckpt_filename,
        verbose=True,
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
        TrainingTimer(save_dir=str(log_dir)),
    ]

    # Handle early stopping (both nested object and flat config)
    es_cfg = cfg.training.get("early_stopping", None)
    es_patience = cfg.training.get("early_stopping_patience", None)

    if es_cfg and hasattr(es_cfg, 'get') and es_cfg.get("enabled", False):
        callbacks.append(EarlyStopping(
            monitor=es_cfg.get("monitor", "val_loss"),
            patience=es_cfg.get("patience", 10),
            mode=es_cfg.get("mode", "min"),
            verbose=True,
        ))
    elif es_patience is not None:
        # Flat config with early_stopping_patience
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=es_patience,
            mode="min",
            verbose=True,
        ))

    # 8. Strategy
    strategy = cfg.training.get("strategy", "auto")
    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False)

    # 9. Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=accelerator,
        devices=devices,
        num_nodes=cfg.training.get("num_nodes", 1),
        strategy=strategy,
        precision=cfg.training.get("precision", "bf16-mixed"),
        gradient_clip_val=cfg.training.get("gradient_clip_val", 3.0),
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        check_val_every_n_epoch=cfg.training.get("check_val_every_n_epoch", 1),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.logging.get("log_every_n_steps", 10),
        fast_dev_run=args.fast_dev_run,
        use_distributed_sampler=False,  # KODIAK handles its own distributed sampling
    )

    # 10. Eval-only mode: skip training and just run test evaluation
    if args.eval_only:
        print("\n" + "=" * 60)
        print("EVAL-ONLY MODE: Running Test Evaluation")
        print("=" * 60)

        # Find checkpoint to evaluate: best (lowest val_loss in filename) > last > newest
        eval_ckpt = find_classification_checkpoint(checkpoint_dir)
        if eval_ckpt is None:
            print(f"ERROR: No checkpoint found in {checkpoint_dir}")
            sys.exit(1)

        print(f"Using checkpoint: {eval_ckpt}")

        # Run test
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path=eval_ckpt)

        # Save test results
        if test_results:
            test_metrics = test_results[0]
            results = {
                "dataset": dataset_name,
                "mode": mode,
                "seed": seed,
                "pretrained_path": pretrained_path,
                # Pretraining hyperparameters
                "pretrain_num_prototypes": pretrain_info.get("num_prototypes"),
                "pretrain_koleo_loss_weight": pretrain_info.get("koleo_loss_weight"),
                "pretrain_cls_loss_weight": pretrain_info.get("cls_loss_weight"),
                "pretrain_multi_crop": pretrain_info.get("multi_crop"),
                "pretrain_projector_dim": pretrain_info.get("projector_dim"),
                # Validation results
                "val_acc": None,  # Not available in eval-only mode
                # Test results - all metrics from ClassificationLearner
                "test_loss": float(test_metrics.get("test_loss", 0)),
                "test_acc": float(test_metrics.get("test_acc", 0)),
                "test_acc_top5": float(test_metrics.get("test_acc_top5", 0)) if "test_acc_top5" in test_metrics else None,
                "test_f1_macro": float(test_metrics.get("test_f1_macro", 0)),
                "test_f1_weighted": float(test_metrics.get("test_f1_weighted", 0)),
                "test_precision_macro": float(test_metrics.get("test_precision_macro", 0)),
                "test_precision_weighted": float(test_metrics.get("test_precision_weighted", 0)),
                "test_recall_macro": float(test_metrics.get("test_recall_macro", 0)),
                "test_recall_weighted": float(test_metrics.get("test_recall_weighted", 0)),
                "test_auroc": float(test_metrics.get("test_auroc", 0)),
            }

            results_file = log_dir / "test_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            print("\n" + "=" * 60)
            print("Test Evaluation Complete!")
            print(f"Test acc:   {test_metrics.get('test_acc', 'N/A'):.4f}")
            print(f"Test F1:    {test_metrics.get('test_f1_macro', 'N/A'):.4f} (macro)")
            print(f"Test AUROC: {test_metrics.get('test_auroc', 'N/A'):.4f}")
            print(f"Results saved to: {results_file}")
            print("=" * 60)

        sys.exit(0)

    # 11. Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume or None)

    # 12. Run test evaluation after training
    print("\n" + "=" * 60)
    print("Running Test Evaluation...")
    print("=" * 60)

    # Load the best checkpoint (lowest val_loss) for testing; fall back to last / in-memory weights
    best_path = checkpoint_callback.best_model_path
    if best_path and Path(best_path).exists():
        print(f"Testing best checkpoint: {best_path}")
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path=best_path)
    elif (checkpoint_dir / "last.ckpt").exists():
        print(f"Testing last checkpoint: {checkpoint_dir / 'last.ckpt'}")
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path=str(checkpoint_dir / "last.ckpt"))
    else:
        print("Testing in-memory weights (no checkpoint saved)")
        test_results = trainer.test(model, datamodule=datamodule)

    # 13. Save test results to JSON
    if test_results:
        test_metrics = test_results[0]
        results = {
            "dataset": dataset_name,
            "mode": mode,
            "seed": seed,
            "pretrained_path": pretrained_path,
            # Pretraining hyperparameters
            "pretrain_num_prototypes": pretrain_info.get("num_prototypes"),
            "pretrain_koleo_loss_weight": pretrain_info.get("koleo_loss_weight"),
            "pretrain_multi_crop": pretrain_info.get("multi_crop"),
            "pretrain_projector_dim": pretrain_info.get("projector_dim"),
            # Validation results
            "val_acc": float(trainer.callback_metrics.get("val_acc", 0)),
            # Test results - all metrics from ClassificationLearner
            "test_loss": float(test_metrics.get("test_loss", 0)),
            "test_acc": float(test_metrics.get("test_acc", 0)),
            "test_acc_top5": float(test_metrics.get("test_acc_top5", 0)) if "test_acc_top5" in test_metrics else None,
            "test_f1_macro": float(test_metrics.get("test_f1_macro", 0)),
            "test_f1_weighted": float(test_metrics.get("test_f1_weighted", 0)),
            "test_precision_macro": float(test_metrics.get("test_precision_macro", 0)),
            "test_precision_weighted": float(test_metrics.get("test_precision_weighted", 0)),
            "test_recall_macro": float(test_metrics.get("test_recall_macro", 0)),
            "test_recall_weighted": float(test_metrics.get("test_recall_weighted", 0)),
            "test_auroc": float(test_metrics.get("test_auroc", 0)),
        }

        results_file = log_dir / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print("Training & Evaluation Complete!")
        print(f"Best val_acc: {trainer.callback_metrics.get('val_acc', 'N/A')}")
        print(f"Test acc:     {test_metrics.get('test_acc', 'N/A'):.4f}")
        print(f"Test F1:      {test_metrics.get('test_f1_macro', 'N/A'):.4f} (macro)")
        print(f"Test AUROC:   {test_metrics.get('test_auroc', 'N/A'):.4f}")
        print(f"Results saved to: {results_file}")
        print(f"Checkpoints: {checkpoint_dir}")
        print(f"Logs: {log_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
