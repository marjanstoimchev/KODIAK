"""
train.py

Self-supervised pretraining with KODIAK + KoLeo.
Uses a ViT backbone (borrowed from DINOv3).

Output directory structure:
  logs/pretraining/{dataset}/{experiment_name}/
  checkpoints/pretraining/{dataset}/{experiment_name}/

Usage examples:

  # Basic pretraining
  python scripts/train.py --config configs/eurosat/pretrain.yaml

  # Override directories
  python scripts/train.py \
      --config configs/eurosat/pretrain.yaml \
      --log_base_dir logs/my_experiment \
      --checkpoint_base_dir checkpoints/my_experiment

  # Resume training
  python scripts/train.py \
      --config configs/eurosat/pretrain.yaml \
      --resume checkpoints/pretraining/eurosat/my_exp/last.ckpt

  # Experiment with num_prototypes and koleo_weight
  python scripts/train.py \
      --config configs/eurosat/pretrain.yaml \
      --num_prototypes 8192 \
      --koleo_weight 0.2
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.strategies import DDPStrategy
import torch

# Add repo root to import path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.learners import MotifLearner
from src.data import PretrainingDataModule
from src.data.utils import SamplerType
from src.utils.config import load_config, override_config, save_config
from src.utils.loggers import get_logger_from_config
from src.utils.runtime import resolve_accelerator_and_devices, count_devices
from src.callbacks import TrainingTimer


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train KODIAK + KoLeo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Path to config.yaml")

    # Common overrides
    p.add_argument("--batch_size", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--max_epochs", type=int)
    p.add_argument("--pretrained_path", type=str)
    p.add_argument("--precision", type=str)
    p.add_argument("--devices", type=str)
    p.add_argument("--logger", type=str, choices=["csv", "wandb", "tensorboard"])
    p.add_argument("--name", type=str)
    p.add_argument("--resume", type=str)
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--log_base_dir", type=str, help="Override logging.base_dir")
    p.add_argument("--log_dir", type=str, help="Exact log directory (no auto-construction)")
    p.add_argument("--checkpoint_base_dir", type=str, help="Override training.checkpoint.base_dir")
    p.add_argument("--checkpoint_dir", type=str, help="Exact checkpoint directory (no auto-construction)")
    p.add_argument("--save_every_n_epochs", type=int, default=None,
                   help="Save checkpoint every N epochs (default: from config)")
    p.add_argument("--seed", type=int, default=None, help="Random seed (overrides experiment.seed)")
    p.add_argument("--root_dir", type=str, default=None,
                   help="Root directory of a folder-based (custom) dataset (overrides data.root_dir)")
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader workers (overrides data.num_workers)")

    # Model hyperparameters for experimentation (default: from config)
    p.add_argument("--num_prototypes", type=int, default=None,
                   help="Number of prototypes in the prototype layer (default: from config)")
    p.add_argument("--koleo_weight", type=float, default=None,
                   help="Weight for KoLeo loss (collapse prevention, default: from config)")

    # Multi-crop pretraining (for Multi-Crop Prototype CLS Loss)
    p.add_argument("--multi_crop", action="store_true",
                   help="Enable multi-crop training (2 global + N local crops)")
    p.add_argument("--no_multi_crop", action="store_true",
                   help="Disable multi-crop training (only 2 global crops)")
    p.add_argument("--local_crops_number", type=int, default=None,
                   help="Number of local crops (default: from config, typically 8)")

    # Ablation flags for controlled experiments
    p.add_argument("--no_sinkhorn", action="store_true",
                   help="Disable Sinkhorn-Knopp, use softmax instead")
    p.add_argument("--cls_weight", type=float, default=None,
                   help="Override CLS loss weight (set to 0 to disable)")
    p.add_argument("--freeze_cls_head", action="store_true",
                   help="Keep the CLS prototype head at its random init (reproduces pre-fix runs)")

    # Optimization flags
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile() for ~15-30%% speedup (PyTorch 2.0+)")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile() even if set in config")

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
        o["model.pretrained_path"] = args.pretrained_path
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
    if args.log_base_dir is not None:
        o["logging.base_dir"] = args.log_base_dir
    if args.log_dir is not None:
        o["logging.exact_dir"] = args.log_dir  # Exact path, no auto-construction
    if args.checkpoint_base_dir is not None:
        o["training.checkpoint.base_dir"] = args.checkpoint_base_dir
    if args.checkpoint_dir is not None:
        o["training.checkpoint.exact_dir"] = args.checkpoint_dir  # Exact path, no auto-construction
    if args.save_every_n_epochs is not None:
        o["training.checkpoint.every_n_epochs"] = args.save_every_n_epochs
    if args.seed is not None:
        o["experiment.seed"] = args.seed
    if args.root_dir is not None:
        o["data.root_dir"] = args.root_dir
    if args.num_workers is not None:
        o["data.num_workers"] = args.num_workers
    # Model hyperparameters for experimentation
    if args.num_prototypes is not None:
        o["model.num_prototypes"] = args.num_prototypes
    if args.koleo_weight is not None:
        o["loss.koleo_loss_weight"] = args.koleo_weight
    # Multi-crop pretraining
    if args.multi_crop:
        o["data.multi_crop"] = True
    if args.no_multi_crop:
        o["data.multi_crop"] = False
    if args.local_crops_number is not None:
        o["data.local_crops_number"] = args.local_crops_number
        o["data.augmentation.local_crops_number"] = args.local_crops_number
        o["model.num_local_crops"] = args.local_crops_number
        o["loss.n_local_crops"] = args.local_crops_number
    # Ablation flags
    if args.no_sinkhorn:
        o["loss.use_sinkhorn"] = False
    if args.cls_weight is not None:
        o["loss.prototype_cls_loss_weight"] = args.cls_weight
    if args.freeze_cls_head:
        o["loss.freeze_cls_prototype_head"] = True
    # Optimization flags
    if args.compile:
        o["model.compile_model"] = True
    if args.no_compile:
        o["model.compile_model"] = False
    return o


# ---------------------------------------------------------
# YAML → LightningModule kwargs
# ---------------------------------------------------------
def cfg_to_motif_kwargs(cfg) -> Dict[str, Any]:
    """Map YAML config structure to MotifLearner(**kwargs).

    This function maps EXACT DINOv3 parameters from DinoV3LightningTraining.
    """

    m = cfg.model
    opt = cfg.optimizer
    tr = cfg.training
    loss = cfg.loss
    data = cfg.data

    return dict(
        # Backbone
        img_size=m.get("image_size", 256),
        patch_size=m.get("vit_patch_size", 16),
        in_chans=m.get("vit_in_chans", 3),
        embed_dim=m.get("vit_embed_dim", 384),
        vit_depth=m.get("vit_depth", 12),
        vit_heads=m.get("vit_heads", 6),
        mlp_ratio=m.get("vit_mlp_ratio", 4.0),
        num_storage_tokens=m.get("num_storage_tokens", 4),  # DINOv3 register tokens
        drop_path_rate=m.get("drop_path_rate", 0.3),  # DINOv3 default

        # Pretrained weights
        pretrained_path=m.get("pretrained_path", None),

        # Heads
        num_prototypes=m.get("num_prototypes", 4096),
        projector_dim=m.get("projector_dim", 256),
        decoder_embed_dim=m.get("decoder_dim", 192),
        decoder_depth=m.get("decoder_depth", 4),
        decoder_num_heads=m.get("decoder_heads", 6),

        # NOTE: Mask parameters are NOT passed here anymore.
        # They belong to the DataModule.

        # Losses (Mask + KoLeo)
        center_momentum=loss.get("center_momentum", 0.9),
        sinkhorn_iters=loss.get("sinkhorn_iters", 3),
        koleo_loss_weight=loss.get("koleo_loss_weight", 0.1),
        koleo_distributed=loss.get("koleo_distributed", False),

        # Multi-Crop Prototype CLS Loss
        prototype_cls_loss_weight=loss.get("prototype_cls_loss_weight", 1.0),
        prototype_cls_student_temp=loss.get("prototype_cls_student_temp", 0.1),
        prototype_cls_teacher_temp=loss.get("prototype_cls_teacher_temp", 0.04),
        n_local_crops=loss.get("n_local_crops", 8),

        # Teacher temperature schedule (DINOv3 style: linear warmup)
        teacher_temp_warmup=opt.get("teacher_temp_warmup", 0.04),  # DINOv3 default
        teacher_temp=opt.get("teacher_temp", 0.07),  # DINOv3 default
        teacher_temp_warmup_epochs=opt.get("teacher_temp_warmup_epochs", 30),  # DINOv3 default

        # Learning rate (DINOv3 defaults)
        lr=opt.get("learning_rate", 1e-3),  # DINOv3 base LR
        min_lr=opt.get("min_lr", 1e-6),  # DINOv3 default
        lr_scaling=opt.get("lr_scaling", "sqrt_wrt_1024"),  # DINOv3 scaling rule
        batch_size=data.get("batch_size", 64),  # For LR scaling calculation

        # Weight decay schedule (DINOv3: cosine from 0.04 to 0.4)
        weight_decay=opt.get("weight_decay", 0.04),  # DINOv3 default start
        weight_decay_end=opt.get("weight_decay_end", 0.4),  # DINOv3 default end

        # Warmup and freeze (DINOv3 defaults)
        warmup_epochs=opt.get("warmup_epochs", 10),  # DINOv3 default
        freeze_last_layer_epochs=opt.get("freeze_last_layer_epochs", 1),  # DINOv3 default
        max_epochs=tr.max_epochs,

        # EMA momentum schedule (DINOv3: cosine from 0.992 to 1.0)
        ema_momentum=opt.get("ema_momentum", 0.992),  # DINOv3 default start
        ema_momentum_end=opt.get("ema_momentum_end", 1.0),  # DINOv3 default end

        # Backbone settings
        freeze_backbone=m.get("freeze_backbone", False),
        backbone_lr_scale=m.get("backbone_lr_scale", 1.0),  # 1.0 for pretraining

        # Layerwise LR decay (DINOv3 defaults)
        layerwise_decay=opt.get("layerwise_decay", 0.9),  # DINOv3 default
        patch_embed_lr_mult=opt.get("patch_embed_lr_mult", 0.2),  # DINOv3 default

        # AdamW optimizer (EXACT DINOv3 defaults)
        adamw_beta1=opt.get("adamw_beta1", 0.9),  # DINOv3 default
        adamw_beta2=opt.get("adamw_beta2", 0.999),  # DINOv3 default

        # Weight decay multipliers (DINOv3 defaults)
        dino_head_wd_multiplier=opt.get("dino_head_wd_multiplier", 1.0),  # DINOv3 default

        # Ablation flags
        use_sinkhorn=loss.get("use_sinkhorn", True),
        freeze_cls_prototype_head=loss.get("freeze_cls_prototype_head", False),

        # Optimization
        compile_model=m.get("compile_model", False),
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

    # 2. Validate configuration
    from src.utils.config_validation import check_config
    if not check_config(cfg.to_dict()):
        sys.exit(1)
    print("Configuration validated successfully")

    # 3. Seed & precision
    pl.seed_everything(cfg.experiment.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # 4. Training summary
    mask_min_max = cfg.model.get("mask_ratio_min_max", [0.1, 0.5])
    mask_prob = cfg.model.get("mask_sample_probability", 0.5)
    num_prototypes = cfg.model.get("num_prototypes", 4096)
    koleo_weight = cfg.loss.get("koleo_loss_weight", 0.1)
    proto_cls_weight = cfg.loss.get("prototype_cls_loss_weight", 1.0)
    multi_crop = cfg.data.get("multi_crop", False)

    print("\n" + "=" * 80)
    print(f"Training: {cfg.experiment.name}")
    print("=" * 80)
    print(f"Backbone:           ViT-S/16 (Depth={cfg.model.vit_depth})")
    print(f"Pretrained:         {cfg.model.get('pretrained_path', 'None')}")
    print(f"Masking:            DINOv3 variable, Range={mask_min_max}, Probability={mask_prob}")
    print(f"Prototypes:         {num_prototypes}")
    print(f"KoLeo Weight:       {koleo_weight}")
    print(f"Proto CLS Weight:   {proto_cls_weight}  (Multi-crop prototype CLS loss)")
    print(f"Multi-Crop:         {multi_crop}")
    if multi_crop:
        print(f"  Local Crops:      {cfg.data.get('local_crops_number', 8)} x {cfg.data.get('local_crops_size', 96)}px")
    print(f"Batch Size:         {cfg.data.batch_size}")
    print(f"Precision:          {cfg.training.precision}")
    print(f"Devices:            {cfg.training.get('devices', 'auto')}")
    print("-" * 80)
    print("DINOv3 Scheduler Parameters (EXACT MATCH):")
    print(f"  Base LR:          {cfg.optimizer.get('learning_rate', 1e-3)}")
    print(f"  LR Scaling:       {cfg.optimizer.get('lr_scaling', 'sqrt_wrt_1024')}")
    print(f"  Min LR:           {cfg.optimizer.get('min_lr', 1e-6)}")
    print(f"  Warmup Epochs:    {cfg.optimizer.get('warmup_epochs', 10)}")
    print(f"  Freeze Last Layer: {cfg.optimizer.get('freeze_last_layer_epochs', 1)} epochs (ONLY last layer frozen)")
    print(f"  Layerwise Decay:  {cfg.optimizer.get('layerwise_decay', 0.9)}")
    print(f"  Patch Embed Mult: {cfg.optimizer.get('patch_embed_lr_mult', 0.2)}")
    print(f"  Weight Decay:     {cfg.optimizer.get('weight_decay', 0.04)} -> {cfg.optimizer.get('weight_decay_end', 0.4)}")
    print(f"  EMA Momentum:     {cfg.optimizer.get('ema_momentum', 0.992)} -> {cfg.optimizer.get('ema_momentum_end', 1.0)}")
    print(f"  Teacher Temp:     {cfg.optimizer.get('teacher_temp_warmup', 0.04)} -> {cfg.optimizer.get('teacher_temp', 0.07)}")
    print(f"  AdamW Betas:      ({cfg.optimizer.get('adamw_beta1', 0.9)}, {cfg.optimizer.get('adamw_beta2', 0.999)})")
    print(f"  DINO Head WD:     {cfg.optimizer.get('dino_head_wd_multiplier', 1.0)}x")
    print("=" * 80 + "\n")

    # 5. DataModule

    # Ensure min_max is a tuple
    if isinstance(mask_min_max, list):
        mask_min_max = tuple(mask_min_max)

    # Compute per-GPU batch size (DinoV3 style: batch_size is TOTAL, divide by num GPUs)
    accelerator, devices = resolve_accelerator_and_devices(cfg.training.get("devices", "auto"))
    num_gpus = count_devices(devices)

    per_gpu_batch_size = max(1, cfg.data.batch_size // num_gpus)
    print(f"Batch size: {cfg.data.batch_size} total -> {per_gpu_batch_size} per device ({num_gpus} x {accelerator})")

    # Multi-crop config
    local_crops_scale = cfg.data.get("local_crops_scale", [0.05, 0.4])
    if isinstance(local_crops_scale, list):
        local_crops_scale = tuple(local_crops_scale)

    # Parse sampler_type from config (default: DISTRIBUTED)
    sampler_type_str = str(cfg.data.get("sampler_type", "distributed")).upper()
    sampler_type = SamplerType[sampler_type_str]

    datamodule = PretrainingDataModule(
        dataset_type=cfg.data.get("dataset_type", "huggingface"),
        batch_size=per_gpu_batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        augmentation=cfg.data.augmentation.to_dict(),

        # --- DINOv3 Masking Args ---
        mask_ratio_tuple=mask_min_max,
        mask_probability=mask_prob,
        num_masks=2,

        # --- Multi-Crop Args (for Multi-Crop Prototype CLS Loss) ---
        multi_crop=multi_crop,
        local_crops_number=cfg.data.get("local_crops_number", 8),
        local_crops_size=cfg.data.get("local_crops_size", 96),
        local_crops_scale=local_crops_scale,

        # Sampler config
        sampler_type=sampler_type,

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
    motif_kwargs = cfg_to_motif_kwargs(cfg)
    model = MotifLearner(**motif_kwargs)

    # 6. Construct output directories
    # Structure: {base_dir}/{dataset_name}/{experiment_name}/
    # Experiment name includes key hyperparameters for easy identification
    dataset_name = cfg.data.get("name", "default")

    # Build experiment name with key hyperparameters
    base_exp_name = cfg.experiment.name
    exp_name = f"{base_exp_name}_proto{num_prototypes}_koleo{koleo_weight}_cls{proto_cls_weight}"
    if multi_crop:
        local_crops_number = cfg.data.get("local_crops_number", 8)
        exp_name = f"{exp_name}_mc{local_crops_number}"

    # Update config with constructed experiment name
    cfg.experiment.name = exp_name
    print(f"Experiment: {exp_name}")

    # Log directory
    # Priority: exact_dir > base_dir > config default
    log_exact_dir = cfg.logging.get("exact_dir", None)
    log_base_dir = cfg.logging.get("base_dir", None)
    if log_exact_dir:
        # Use exact directory as-is (no auto-construction)
        log_dir = Path(log_exact_dir)
        cfg.logging.save_dir = str(log_dir)
    elif log_base_dir:
        # Auto-construct: base_dir/dataset_name (logger adds exp_name)
        log_dir = Path(log_base_dir) / dataset_name
        cfg.logging.save_dir = str(log_dir)
    else:
        log_dir = Path(cfg.logging.save_dir)

    # Checkpoint directory
    # Priority: exact_dir > base_dir > config default
    ckpt_exact_dir = cfg.training.checkpoint.get("exact_dir", None)
    ckpt_base_dir = cfg.training.checkpoint.get("base_dir", None)
    if ckpt_exact_dir:
        # Use exact directory as-is (no auto-construction)
        checkpoint_dir = Path(ckpt_exact_dir)
        cfg.training.checkpoint.dirpath = str(checkpoint_dir)
    elif ckpt_base_dir:
        # Auto-construct: base_dir/dataset_name/exp_name
        checkpoint_dir = Path(ckpt_base_dir) / dataset_name / exp_name
        cfg.training.checkpoint.dirpath = str(checkpoint_dir)
    else:
        checkpoint_dir = Path(cfg.training.checkpoint.dirpath)

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = get_logger_from_config(cfg)

    # Save config
    save_config(cfg, log_dir / exp_name / "config.yaml")

    # 7. Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        TrainingTimer(save_dir=str(log_dir / exp_name)),
    ]

    # Periodic checkpointing (needed for --resume and for long runs). The final
    # checkpoint is always written to {checkpoint_dir}/last.ckpt after training.
    every_n_epochs = cfg.training.checkpoint.get("every_n_epochs", None)
    enable_checkpointing = every_n_epochs is not None and int(every_n_epochs) > 0
    if enable_checkpointing:
        callbacks.append(ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=cfg.training.checkpoint.get("filename", "kodiak-{epoch:02d}"),
            every_n_epochs=int(every_n_epochs),
            save_top_k=-1,  # keep every periodic checkpoint
            save_last=True,
            save_on_train_epoch_end=True,
            verbose=True,
        ))
        print(f"Periodic checkpointing: every {every_n_epochs} epoch(s) -> {checkpoint_dir}")

    if cfg.training.early_stopping.enabled:
        callbacks.append(EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            verbose=True,
        ))

    # 8. Strategy
    strategy = cfg.training.strategy
    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False)

    # 9. Trainer
    # Note: No validation during pretraining (DINOv3 style)
    # - num_sanity_val_steps=0 disables the sanity check
    # - limit_val_batches=0 skips validation entirely
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=accelerator,
        devices=devices,
        num_nodes=cfg.training.num_nodes,
        strategy=strategy,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        num_sanity_val_steps=0,  # No validation sanity check (no val set)
        limit_val_batches=0,  # Skip validation entirely during pretraining
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        fast_dev_run=args.fast_dev_run,
        enable_checkpointing=enable_checkpointing,
        use_distributed_sampler=False,  # KODIAK handles its own distributed sampling
    )

    # 10. Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume or None)

    # 11. Save final checkpoint (once, after training completes)
    # Note: trainer.save_checkpoint handles rank-0-only saving internally
    ckpt_path = checkpoint_dir / "last.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    if trainer.is_global_zero:
        print(f"Saved final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
