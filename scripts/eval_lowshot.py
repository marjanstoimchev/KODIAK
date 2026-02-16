"""
eval_lowshot.py — K-shot classification evaluation.

Evaluates SSL representations under limited labelled data.
Supports both fine-tuning (FT) and linear probing (LP).

Protocol (k-shot):
  - Each class gets exactly k labelled examples
  - label_seed  controls WHICH samples are selected  (3 label subsets)
  - train_seed  controls weight init + augmentation   (3 training seeds)
  - Total: 3 x 3 = 9 runs per (method, dataset, k, mode)
  - Test on the FULL held-out test set (never subsampled)

Output:
  {output_dir}/{dataset}/{method_tag}/{mode}_{k}shot_ls{label_seed}_ts{train_seed}/
    ├── logs/        (CSV logger)
    ├── checkpoints/ (best + last)
    └── test_results.json

Usage:
  # Single run (5-shot fine-tuning)
  python scripts/eval_lowshot.py \\
      --config configs/DTD/classify.yaml \\
      --pretrained_path /path/to/checkpoint.ckpt \\
      --k_shot 5 \\
      --label_seed 0 \\
      --train_seed 0 \\
      --freeze_backbone          # LP mode (omit for FT)

  # The run_lowshot.sh script automates all combinations.

Author: KODIAK Team
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import Subset, DataLoader

# Add repo root and scripts dir to import path
ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.learners import ClassificationLearner
from src.data.classification.datamodule import (
    ClassificationDataModule,
    TransformDataset,
    ClassificationTrainTransform,
    ClassificationValTransform,
)
from src.data.classification.collate import classification_collate
from src.data.utils import SamplerType, make_sampler
from src.callbacks import MPPProgressBar
from src.utils.config import load_config, override_config

# Re-use checkpoint helpers from train_classifier (same scripts/ directory)
from train_classifier import (
    extract_pretraining_info,
    build_pretraining_folder_name,
    find_checkpoint,
    cfg_to_classification_kwargs,
)


# ─────────────────────────────────────────────────────────────────────
# Stratified k-shot sampler
# ─────────────────────────────────────────────────────────────────────
def get_all_labels(dataset) -> np.ndarray:
    """Extract integer labels from every sample in *dataset*.

    Works with:
      - HuggingFaceDataset  (dict with 'label' key)
      - TransformDataset    (dict with 'labels' key after transform)
      - Subset wrappers     (recurse into .dataset)
    """
    labels = []

    # Unwrap Subset / TransformDataset to reach the base dataset
    base = dataset
    indices = None
    while True:
        if isinstance(base, Subset):
            indices = base.indices if indices is None else [base.indices[i] for i in indices]
            base = base.dataset
        elif isinstance(base, TransformDataset):
            base = base.base
        else:
            break

    # Read labels from the base dataset
    n = len(dataset)
    for i in range(n):
        real_idx = indices[i] if indices is not None else i
        item = base[real_idx]
        if isinstance(item, dict):
            lbl = item.get("label", item.get("labels", -1))
        elif isinstance(item, (tuple, list)):
            lbl = item[1]
        else:
            lbl = -1
        labels.append(int(lbl))

    return np.asarray(labels, dtype=np.int64)


def kshot_indices(
    labels: np.ndarray,
    k_shot: int,
    seed: int,
) -> List[int]:
    """Return indices for a k-shot subset of *labels*.

    Each class gets exactly *k_shot* samples (or all available if fewer).
    Sampling is deterministic given *seed*.

    Args:
        labels:  1-D integer array of class labels.
        k_shot:  Number of labelled examples per class.
        seed:    Random seed for reproducible subset selection.

    Returns:
        Sorted list of selected indices.
    """
    rng = np.random.RandomState(seed)
    class_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        class_indices[lbl].append(idx)

    selected: List[int] = []
    for cls in sorted(class_indices.keys()):
        idxs = np.array(class_indices[cls])
        k = min(k_shot, len(idxs))  # clamp to available
        chosen = rng.choice(idxs, size=k, replace=False)
        selected.extend(chosen.tolist())

    selected.sort()
    return selected


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="K-shot evaluation (FT / LP) for SSL encoders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to classify.yaml (same configs as full eval)")
    p.add_argument("--pretrained_path", type=str, required=True,
                   help="Path to pretrained SSL checkpoint (file or directory)")
    p.add_argument("--checkpoint_type", type=str, choices=["best", "last"],
                   default="last",
                   help="Which checkpoint to load from directory")

    # K-shot protocol
    p.add_argument("--k_shot", type=int, required=True,
                   help="Number of labelled examples per class (e.g. 1, 5, 10)")
    p.add_argument("--label_seed", type=int, required=True,
                   help="Seed for label subset selection (controls WHICH samples)")
    p.add_argument("--train_seed", type=int, required=True,
                   help="Seed for training (controls init / augmentation)")

    # Mode
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Linear probing mode (freeze encoder)")

    # Training overrides
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--precision", type=str, default=None)
    p.add_argument("--encoder_type", type=str, choices=["teacher", "student"],
                   default=None)

    # Output
    p.add_argument("--output_dir", type=str, default=None,
                   help="Base output directory (default: output/lowshot)")
    p.add_argument("--method_tag", type=str, default=None,
                   help="Method name for folder structure (e.g. kodiak, dinov3). "
                        "Auto-detected from checkpoint if omitted.")

    p.add_argument("--fast_dev_run", action="store_true")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── 1. Load config ────────────────────────────────────────────────
    cfg = load_config(args.config)

    # Apply overrides
    overrides: Dict[str, Any] = {}
    resolved_path = find_checkpoint(args.pretrained_path, args.checkpoint_type)
    if resolved_path is None:
        raise FileNotFoundError(
            f"Could not find checkpoint at: {args.pretrained_path}\n"
            f"Checkpoint type: {args.checkpoint_type}"
        )
    overrides["model.pretrained_path"] = resolved_path

    # Low-shot training defaults (more epochs + patience for tiny datasets)
    if args.max_epochs is None:
        overrides["training.max_epochs"] = 100
    overrides["training.early_stopping_patience"] = 20

    if args.freeze_backbone:
        overrides["model.freeze_backbone"] = True
    if args.batch_size is not None:
        overrides["data.batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["optimizer.lr"] = args.learning_rate
    if args.max_epochs is not None:
        overrides["training.max_epochs"] = args.max_epochs
    if args.precision is not None:
        overrides["training.precision"] = args.precision
    if args.encoder_type is not None:
        overrides["model.encoder_type"] = args.encoder_type

    # Use train_seed for reproducibility
    overrides["experiment.seed"] = args.train_seed

    if overrides:
        cfg = override_config(cfg, overrides)

    # ── 2. Seed everything ────────────────────────────────────────────
    pl.seed_everything(args.train_seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # ── 3. Build output directory ─────────────────────────────────────
    dataset_name = cfg.data.get("name", "default")
    mode = "lineareval" if args.freeze_backbone else "finetune"

    # Method tag
    method_tag = args.method_tag
    if method_tag is None:
        pretrain_info = extract_pretraining_info(resolved_path)
        method_tag = build_pretraining_folder_name(pretrain_info)

    # Output base
    output_base = args.output_dir
    if output_base is None:
        output_base = "output/lowshot"

    run_name = f"{mode}_{args.k_shot}shot_ls{args.label_seed}_ts{args.train_seed}"
    run_dir = Path(output_base) / dataset_name / method_tag / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. Summary ────────────────────────────────────────────────────
    num_classes = (
        cfg.data.get("num_classes", None) or cfg.model.get("num_classes", None)
    )
    print("\n" + "=" * 72)
    print("K-SHOT EVALUATION")
    print("=" * 72)
    print(f"  Dataset:          {dataset_name} ({num_classes} classes)")
    print(f"  Mode:             {mode}")
    print(f"  K-shot:           {args.k_shot} examples/class")
    print(f"  Label seed:       {args.label_seed}")
    print(f"  Train seed:       {args.train_seed}")
    print(f"  Method tag:       {method_tag}")
    print(f"  Checkpoint:       {resolved_path}")
    print(f"  Max epochs:       {cfg.training.max_epochs}")
    print(f"  Output:           {run_dir}")
    print("=" * 72 + "\n")

    # ── 5. Build DataModule (full) to get splits ──────────────────────
    # Always single GPU for k-shot (CUDA_VISIBLE_DEVICES selects the GPU)
    per_gpu_bs = cfg.data.batch_size

    aug_config = cfg.data.get("augmentation", None)
    if aug_config is not None and hasattr(aug_config, "to_dict"):
        aug = aug_config.to_dict()
    else:
        aug = aug_config if isinstance(aug_config, dict) else {}

    image_size = cfg.data.get("image_size", aug.get("global_crops_size", 256))
    mean = aug.get("normalize_mean", [0.485, 0.456, 0.406])
    std = aug.get("normalize_std", [0.229, 0.224, 0.225])

    # We build the datamodule to get the full train/val/test splits,
    # then replace the train set with the k-shot subset.
    datamodule = ClassificationDataModule(
        dataset_type=cfg.data.get("dataset_type", "tissue"),
        batch_size=per_gpu_bs,
        num_workers=cfg.data.get("num_workers", 8),
        pin_memory=cfg.data.get("pin_memory", True),
        persistent_workers=cfg.data.get("persistent_workers", True),
        train_split=cfg.data.get("train_split", 0.7),
        val_split=cfg.data.get("val_split", 0.1),
        test_split=cfg.data.get("test_split", 0.2),
        image_size=image_size,
        normalize_mean=mean,
        normalize_std=std,
        sampler_type=SamplerType.DISTRIBUTED,
        seed=cfg.experiment.seed,
        csv_path=cfg.data.get("csv_path", None),
        magnification=cfg.data.get("magnification", None),
        root_path=cfg.data.get("root_path", None),
        hf_dataset_name=cfg.data.get("hf_dataset_name", None),
        hf_split=cfg.data.get("hf_split", None),
        hf_cache_dir=cfg.data.get("hf_cache_dir", None),
    )
    datamodule.setup()

    # ── 6. K-shot subsampling ─────────────────────────────────────────
    full_train = datamodule.train_dataset  # TransformDataset wrapping base
    train_labels = get_all_labels(full_train)
    selected_idx = kshot_indices(
        train_labels,
        k_shot=args.k_shot,
        seed=args.label_seed,
    )

    n_full = len(full_train)
    n_sub = len(selected_idx)
    print(f"K-shot subset: {n_sub}/{n_full} samples "
          f"({args.k_shot} per class)")

    # Per-class breakdown
    sub_labels = train_labels[selected_idx]
    unique, counts = np.unique(sub_labels, return_counts=True)
    print(f"  Classes: {len(unique)}, "
          f"min/class: {counts.min()}, "
          f"max/class: {counts.max()}, "
          f"mean/class: {counts.mean():.1f}")

    # Replace training set with the k-shot subset
    lowshot_train = Subset(full_train, selected_idx)
    datamodule.train_dataset = lowshot_train

    # Auto-clamp batch size so we get at least 1 batch
    effective_bs = min(per_gpu_bs, n_sub)
    effective_bs = max(effective_bs, 1)
    if effective_bs != per_gpu_bs:
        print(f"  Clamped per-GPU batch size: {per_gpu_bs} -> {effective_bs} "
              f"(subset too small for original BS)")

    # Patch train_dataloader to use the subset
    _original_train_dl = datamodule.train_dataloader

    def lowshot_train_dataloader():
        sampler = make_sampler(
            dataset=lowshot_train,
            sampler_type=SamplerType.DISTRIBUTED,
            shuffle=True,
            seed=args.train_seed,
        )
        return DataLoader(
            lowshot_train,
            batch_size=effective_bs,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=cfg.data.get("num_workers", 8),
            pin_memory=True,
            persistent_workers=bool(cfg.data.get("num_workers", 8) > 0),
            drop_last=False,  # keep all samples for tiny subsets
            collate_fn=classification_collate,
        )

    datamodule.train_dataloader = lowshot_train_dataloader

    # ── 7. Model ──────────────────────────────────────────────────────
    classification_kwargs = cfg_to_classification_kwargs(cfg)
    model = ClassificationLearner(**classification_kwargs)

    # ── 8. Callbacks (lightweight — no checkpoint saving) ──────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.get("early_stopping_patience", 20),
            mode="min",
            verbose=True,
        ),
        MPPProgressBar(mode="classification", refresh_rate=1, leave=True),
    ]

    # ── 9. Logger ─────────────────────────────────────────────────────
    logger = False  # disable logging to save I/O

    # ── 10. Trainer (single GPU — CUDA_VISIBLE_DEVICES selects GPU) ──
    steps_per_epoch = max(1, n_sub // max(effective_bs, 1))
    val_interval = max(1, 5 // steps_per_epoch) if steps_per_epoch < 5 else 1

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.training.get("precision", "bf16-mixed"),
        gradient_clip_val=cfg.training.get("gradient_clip_val", 3.0),
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        check_val_every_n_epoch=val_interval,
        log_every_n_steps=max(1, steps_per_epoch),
        fast_dev_run=args.fast_dev_run,
        use_distributed_sampler=False,
    )

    # ── 11. Train ─────────────────────────────────────────────────────
    t0 = time.time()
    trainer.fit(model, datamodule=datamodule)
    train_secs = time.time() - t0

    # ── 12. Test (full test set, in-memory model) ─────────────────────
    print("\n" + "=" * 60)
    print("Test Evaluation (full test set)")
    print("=" * 60)
    t1 = time.time()
    test_results = trainer.test(model, datamodule=datamodule)
    test_secs = time.time() - t1
    total_secs = train_secs + test_secs

    # ── 13. Save results JSON ─────────────────────────────────────────
    if test_results:
        tm = test_results[0]
        results = {
            # Identifiers
            "dataset": dataset_name,
            "mode": mode,
            "k_shot": args.k_shot,
            "label_seed": args.label_seed,
            "train_seed": args.train_seed,
            "method_tag": method_tag,
            "pretrained_path": str(resolved_path),
            # Subset stats
            "n_train_full": n_full,
            "n_train_lowshot": n_sub,
            "n_classes": int(len(unique)),
            "samples_per_class": int(args.k_shot),
            # Metrics (top-1 accuracy only)
            "val_acc": float(trainer.callback_metrics.get("val_acc", 0)),
            "test_acc": float(tm.get("test_acc", 0)),
            # Timing
            "train_seconds": round(train_secs, 1),
            "test_seconds": round(test_secs, 1),
            "total_seconds": round(total_secs, 1),
            "gpu_hours": round(total_secs / 3600, 4),
        }

        results_file = run_dir / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print("K-Shot Evaluation Complete!")
        print(f"  Test acc:    {tm.get('test_acc', 0):.4f}")
        print(f"  Time:        {train_secs:.0f}s train + {test_secs:.0f}s test = {total_secs:.0f}s ({total_secs/3600:.3f} GPU-h)")
        print(f"  Results:     {results_file}")
        print("=" * 60)


if __name__ == "__main__":
    main()
