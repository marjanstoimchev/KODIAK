"""
eval_knn.py

k-NN evaluation of pretrained SSL encoder features.
Standard protocol: extract features → L2-normalize → weighted k-NN voting.

Follows the DINOv3/DINO evaluation protocol:
- Features: CLS token from teacher encoder
- Normalization: L2
- Voting: temperature-scaled softmax weighting (T=0.07)
- Default k: 20

Supports:
- Two pretraining modes: scratch and continued
- Multiple seeds: load model once, re-split data per seed, report mean ± std

Usage:
    # Single seed
    python scripts/eval_knn.py \
        --config configs/DTD/classify.yaml \
        --pretrained_path output/checkpoints/pretraining/dtd/last.ckpt

    # Multiple seeds
    python scripts/eval_knn.py \
        --config configs/DTD/classify.yaml \
        --pretrained_path output/checkpoints/pretraining/dtd/last.ckpt \
        --seeds "0 1 42"

    # Continued pretraining mode
    python scripts/eval_knn.py \
        --config configs/DTD/classify.yaml \
        --pretrained_path output/checkpoints/pretraining/dtd/last.ckpt \
        --init_mode continued --seeds "0 1 42"

Output:
    output/knn/{dataset}/{pretrain_folder}/knn_results.json
    output/knn_from_continued/{dataset}/{pretrain_folder}/knn_results.json
"""

import argparse
import json
import sys
import re
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.models import LinearClassifier
from src.data import ClassificationDataModule
from src.data.utils import SamplerType
from src.utils.config import load_config


# ---------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------
def find_checkpoint(path: str, checkpoint_type: str = "last") -> Optional[str]:
    path = Path(path)
    if path.is_file():
        return str(path)
    if path.is_dir():
        if checkpoint_type == "best":
            ckpts = list(path.glob("kodiak-*.ckpt"))
            if not ckpts:
                ckpts = list(path.glob("*.ckpt"))
            best_ckpt, best_loss = None, float("inf")
            for ckpt in ckpts:
                if ckpt.name == "last.ckpt":
                    continue
                match = re.search(r"train_loss[=_](\d+\.?\d*)", ckpt.name)
                if match:
                    loss = float(match.group(1))
                    if loss < best_loss:
                        best_loss = loss
                        best_ckpt = ckpt
            if best_ckpt:
                return str(best_ckpt)
        last = path / "last.ckpt"
        if last.exists():
            return str(last)
        ckpts = list(path.glob("*.ckpt"))
        if ckpts:
            return str(sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1])
    return None


def extract_pretraining_info(checkpoint_path: str) -> Dict[str, Any]:
    """Extract pretraining hyperparameters from checkpoint for folder naming."""
    info = {}
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hparams = ckpt.get("hyper_parameters", {})
        info["num_prototypes"] = hparams.get("num_prototypes", None)
        info["koleo_loss_weight"] = hparams.get("koleo_loss_weight", None)
        info["cls_loss_weight"] = hparams.get(
            "prototype_cls_loss_weight", hparams.get("cls_loss_weight", None)
        )
        info["multi_crop"] = hparams.get("multi_crop", False)
        if not info["num_prototypes"]:
            model_cfg = hparams.get("model", {})
            if isinstance(model_cfg, dict):
                info["num_prototypes"] = model_cfg.get("num_prototypes", None)
    except Exception as e:
        print(f"Warning: Could not extract pretraining info: {e}")
    return info


def build_pretrain_folder_name(info: Dict[str, Any]) -> str:
    """Build folder name from pretraining info: proto4096_koleo0.1_cls1_mc"""
    parts = []
    num_proto = info.get("num_prototypes")
    if num_proto is not None:
        parts.append(f"proto{num_proto}")
    koleo = info.get("koleo_loss_weight")
    if koleo is not None:
        parts.append(f"koleo{koleo:.4g}" if isinstance(koleo, float) else f"koleo{koleo}")
    cls_w = info.get("cls_loss_weight")
    if cls_w is not None:
        parts.append(f"cls{cls_w:.4g}" if isinstance(cls_w, float) else f"cls{cls_w}")
    if info.get("multi_crop", False):
        parts.append("mc")
    return "_".join(parts) if parts else "unknown_pretraining"


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------
@torch.no_grad()
def extract_features(model, dataloader, device, concat_cls_patch=False):
    """Extract L2-normalized features and labels from a dataloader.

    Args:
        model: LinearClassifier model
        dataloader: DataLoader
        device: torch device
        concat_cls_patch: If True, concatenate CLS token with mean patch tokens
    """
    model.eval()
    all_features = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        if isinstance(batch, dict):
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

        if images.device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        cls_token, patch_tokens, _ = model.encoder.forward_full(images, return_attention=False)

        if concat_cls_patch:
            features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
        else:
            features = cls_token

        features = F.normalize(features, dim=-1, p=2)

        all_features.append(features.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------
# k-NN evaluation
# ---------------------------------------------------------
def knn_evaluate(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
    num_classes: int = None,
) -> dict:
    """
    Weighted k-NN classification (DINOv3 protocol).

    1. Cosine similarity (features are already L2-normed)
    2. Top-k neighbors
    3. Temperature-scaled softmax weighting
    4. Weighted vote → predicted class
    """
    if num_classes is None:
        num_classes = int(train_labels.max().item()) + 1

    k = min(k, len(train_features))

    similarity = test_features @ train_features.T
    topk_sims, topk_indices = similarity.topk(k, dim=1, largest=True, sorted=True)
    neighbors_labels = train_labels[topk_indices]

    weights = F.softmax(topk_sims / temperature, dim=1)
    one_hot_labels = F.one_hot(neighbors_labels, num_classes=num_classes).float()
    weighted_votes = (weights.unsqueeze(-1) * one_hot_labels).sum(dim=1)

    preds = weighted_votes.argmax(dim=1)
    top1_acc = (preds == test_labels).sum().item() / len(test_labels) * 100.0

    results = {"k": k, "temperature": temperature, "top1_acc": top1_acc}

    if num_classes > 5:
        top5_preds = weighted_votes.topk(5, dim=1).indices
        top5_acc = (top5_preds == test_labels.unsqueeze(1)).any(dim=1).sum().item()
        results["top5_acc"] = top5_acc / len(test_labels) * 100.0

    return results


# ---------------------------------------------------------
# Data setup helper
# ---------------------------------------------------------
def create_datamodule(cfg, batch_size, seed):
    """Create a ClassificationDataModule with a specific seed for splitting."""
    aug_config = cfg.data.get("augmentation", None)
    if aug_config is not None and hasattr(aug_config, "to_dict"):
        aug = aug_config.to_dict()
    else:
        aug = aug_config if isinstance(aug_config, dict) else {}

    image_size = cfg.data.get("image_size", aug.get("global_crops_size", 256))

    pl.seed_everything(seed, workers=True)

    datamodule = ClassificationDataModule(
        dataset_type=cfg.data.get("dataset_type", "huggingface"),
        batch_size=batch_size,
        num_workers=cfg.data.get("num_workers", 8),
        pin_memory=True,
        persistent_workers=True,
        train_split=cfg.data.get("train_split", 0.7),
        val_split=cfg.data.get("val_split", 0.1),
        test_split=cfg.data.get("test_split", 0.2),
        image_size=image_size,
        normalize_mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
        normalize_std=aug.get("normalize_std", [0.229, 0.224, 0.225]),
        sampler_type=SamplerType.EPOCH,
        seed=seed,
        csv_path=cfg.data.get("csv_path", None),
        magnification=cfg.data.get("magnification", None),
        root_path=cfg.data.get("root_path", None),
        hf_dataset_name=cfg.data.get("hf_dataset_name", None),
        hf_split=cfg.data.get("hf_split", None),
        hf_cache_dir=cfg.data.get("hf_cache_dir", None),
    )
    datamodule.setup()

    # Replace train augmentation with val transform (no augmentation for k-NN)
    if hasattr(datamodule.train_dataset, "transform"):
        datamodule.train_dataset.transform = datamodule.val_transform

    return datamodule


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="k-NN evaluation of pretrained SSL features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Path to classify config YAML")
    p.add_argument("--pretrained_path", type=str, required=True,
                   help="Pretrained SSL checkpoint (file or directory)")
    p.add_argument("--checkpoint_type", type=str, choices=["last", "best"], default="last")
    p.add_argument("--init_mode", type=str, choices=["scratch", "continued"], default="scratch",
                   help="Pretraining mode: scratch or continued (affects output directory)")
    p.add_argument("--encoder_type", type=str, choices=["teacher", "student"], default="teacher")
    p.add_argument("--k", type=int, default=20, help="Number of nearest neighbors")
    p.add_argument("--temperature", type=float, default=0.07, help="Softmax temperature")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size for feature extraction")
    p.add_argument("--devices", type=str, default="0", help="GPU device index (single GPU)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output directory (default: auto-constructed)")
    p.add_argument("--output_base_dir", type=str, default="output",
                   help="Base output directory")
    p.add_argument("--seeds", type=str, default="42",
                   help="Space-separated seeds for evaluation (e.g., '0 1 42')")
    p.add_argument("--concat_cls_patch", action="store_true",
                   help="Concatenate CLS token + mean patch tokens for k-NN features (2x embed_dim)")
    return p.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split()]

    # Load config
    cfg = load_config(args.config)

    # Resolve checkpoint
    ckpt_path = find_checkpoint(args.pretrained_path, args.checkpoint_type)
    if ckpt_path is None:
        print(f"ERROR: No checkpoint found at {args.pretrained_path}")
        sys.exit(1)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Device (single GPU)
    device_idx = int(args.devices.split(",")[0])
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")

    # Extract pretraining info for output directory naming
    pretrain_info = extract_pretraining_info(ckpt_path)
    pretrain_folder = build_pretrain_folder_name(pretrain_info)
    dataset_name = cfg.data.get("name", "unknown")
    m = cfg.model
    num_classes = cfg.data.get("num_classes", m.get("num_classes", None))

    # Construct output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        knn_dir = "knn_from_continued" if args.init_mode == "continued" else "knn"
        output_dir = Path(args.output_base_dir) / knn_dir / dataset_name / pretrain_folder

    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    print("\n" + "=" * 60)
    print("k-NN EVALUATION")
    print("=" * 60)
    print(f"Dataset:       {dataset_name}")
    print(f"Init mode:     {args.init_mode}")
    print(f"Checkpoint:    {ckpt_path}")
    print(f"Encoder:       {args.encoder_type}")
    print(f"k:             {args.k}")
    print(f"Temperature:   {args.temperature}")
    print(f"Seeds:         {seeds}")
    print(f"Concat CLS+Patch: {args.concat_cls_patch}")
    print(f"Pretrain info: {pretrain_folder}")
    print(f"Output:        {output_dir}")
    print("=" * 60)

    # Load model ONCE
    print("\nLoading model...")
    model = LinearClassifier(
        num_classes=cfg.data.get("num_classes", m.get("num_classes", 10)),
        img_size=m.get("image_size", 256),
        patch_size=m.get("patch_size", m.get("vit_patch_size", 16)),
        embed_dim=m.get("embed_dim", m.get("vit_embed_dim", 384)),
        depth=m.get("depth", m.get("vit_depth", 12)),
        num_heads=m.get("num_heads", m.get("vit_heads", 6)),
        mlp_ratio=m.get("mlp_ratio", m.get("vit_mlp_ratio", 4.0)),
        num_storage_tokens=None,
        pretrained_path=ckpt_path,
        use_cls_token=True,
        encoder_type=args.encoder_type,
    )
    model = model.to(device)
    model.eval()

    # Run k-NN for each seed
    all_top1 = []
    all_top5 = []
    per_seed_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Setup data with this seed
        datamodule = create_datamodule(cfg, args.batch_size, seed)
        train_loader = datamodule.train_dataloader()
        test_loader = datamodule.test_dataloader()

        # Extract features
        print(f"Extracting train features ({len(datamodule.train_dataset)} samples)...")
        train_features, train_labels = extract_features(model, train_loader, device, concat_cls_patch=args.concat_cls_patch)

        print(f"Extracting test features ({len(datamodule.test_dataset)} samples)...")
        test_features, test_labels = extract_features(model, test_loader, device, concat_cls_patch=args.concat_cls_patch)

        # k-NN evaluation
        results = knn_evaluate(
            train_features, train_labels,
            test_features, test_labels,
            k=args.k,
            temperature=args.temperature,
            num_classes=num_classes,
        )

        all_top1.append(results["top1_acc"])
        if "top5_acc" in results:
            all_top5.append(results["top5_acc"])

        print(f"  Top-1: {results['top1_acc']:.2f}%", end="")
        if "top5_acc" in results:
            print(f"  Top-5: {results['top5_acc']:.2f}%", end="")
        print()

        per_seed_results.append({"seed": seed, **results})

    # Summary
    top1_mean = np.mean(all_top1)
    top1_std = np.std(all_top1)

    print("\n" + "=" * 60)
    print(f"k-NN Results (k={args.k}, T={args.temperature})")
    print("=" * 60)
    if len(seeds) > 1:
        print(f"  Top-1 Accuracy: {top1_mean:.2f} +/- {top1_std:.2f}%")
        if all_top5:
            top5_mean = np.mean(all_top5)
            top5_std = np.std(all_top5)
            print(f"  Top-5 Accuracy: {top5_mean:.2f} +/- {top5_std:.2f}%")
        print(f"  Seeds: {seeds}")
        print(f"  Per-seed Top-1: {['%.2f' % x for x in all_top1]}")
    else:
        print(f"  Top-1 Accuracy: {all_top1[0]:.2f}%")
        if all_top5:
            print(f"  Top-5 Accuracy: {all_top5[0]:.2f}%")
    print("=" * 60)

    # Save results
    results_full = {
        "dataset": dataset_name,
        "init_mode": args.init_mode,
        "checkpoint": ckpt_path,
        "encoder_type": args.encoder_type,
        "k": args.k,
        "temperature": args.temperature,
        "seeds": seeds,
        "num_classes": num_classes,
        # Aggregated
        "knn_top1_acc_mean": round(top1_mean, 4),
        "knn_top1_acc_std": round(top1_std, 4),
    }
    if all_top5:
        results_full["knn_top5_acc_mean"] = round(float(np.mean(all_top5)), 4)
        results_full["knn_top5_acc_std"] = round(float(np.std(all_top5)), 4)

    # Pretraining config
    results_full["pretrain_num_prototypes"] = pretrain_info.get("num_prototypes")
    results_full["pretrain_koleo_weight"] = pretrain_info.get("koleo_loss_weight")
    results_full["pretrain_cls_weight"] = pretrain_info.get("cls_loss_weight")
    results_full["pretrain_multi_crop"] = pretrain_info.get("multi_crop")

    # Per-seed detail
    results_full["per_seed"] = per_seed_results

    results_file = output_dir / "knn_results.json"
    with open(results_file, "w") as f:
        json.dump(results_full, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
