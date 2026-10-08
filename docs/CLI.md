# Command-line reference

All scripts are run from the repository root and take a YAML config plus optional overrides. Every flag below overrides the corresponding config value for that run only.

#### `train.py` — Pretraining

Self-supervised pretraining with KODIAK.

```bash
python scripts/train.py \
    --config configs/pancreatic/pretrain.yaml \
    --devices 0,1 \
    --batch_size 64 \
    --max_epochs 500 \
    --num_prototypes 128 \
    --multi_crop \
    --logger tensorboard
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config **(required)** | — |
| `--devices` | GPU indices, comma-separated | from config |
| `--batch_size` | Total batch size | from config |
| `--learning_rate` | Peak learning rate | from config |
| `--max_epochs` | Training epochs | from config |
| `--num_prototypes` | Number of prototype vectors | from config |
| `--koleo_weight` | KoLeo regularization weight | from config |
| `--cls_weight` | Multi-crop CLS distillation weight | from config |
| `--multi_crop` | Enable multi-crop (2 global + 8 local) | off |
| `--no_sinkhorn` | Use softmax instead of Sinkhorn-Knopp | off |
| `--freeze_cls_head` | Keep the CLS prototype head at its random init (reproduces runs made before the head was trainable) | off |
| `--compile` | Enable `torch.compile()` | off |
| `--precision` | Training precision | bf16-mixed |
| `--logger` | `csv`, `tensorboard`, or `wandb` | csv |
| `--name` | Experiment name | from config |
| `--resume` | Resume from a Lightning checkpoint (`last.ckpt`) | — |
| `--pretrained_path` | Initialize from pretrained weights | — |
| `--save_every_n_epochs` | Save a checkpoint every N epochs (enables `last.ckpt` updates for `--resume`) | from config (off) |
| `--seed` | Random seed | from config |
| `--root_dir` | Folder-based dataset root (overrides `data.root_dir`) | from config |
| `--num_workers` | DataLoader workers (overrides `data.num_workers`) | from config |
| `--fast_dev_run` | Quick sanity check (1 batch) | off |

Without `--save_every_n_epochs` only the final `last.ckpt` is written when training completes.

#### `train_classifier.py` — Classification

Fine-tuning or linear evaluation using a pretrained KODIAK encoder.

```bash
python scripts/train_classifier.py \
    --config configs/pancreatic/classify.yaml \
    --pretrained_path output/checkpoints/pretraining/pancreatic/last.ckpt \
    --devices 0,1 \
    --max_epochs 100 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --seed 42 \
    --freeze_backbone \
    --logger tensorboard
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config **(required)** | — |
| `--pretrained_path` | Pretrained checkpoint file or directory | — |
| `--checkpoint_type` | `last` or `best` (lowest loss) | last |
| `--encoder_type` | `teacher` or `student` | teacher |
| `--freeze_backbone` | Freeze encoder for linear evaluation | off |
| `--use_cls_token` | Use CLS token or mean patch pooling | from config |
| `--concat_cls_patch` | Concatenate CLS + mean patch tokens (2x features) | off |
| `--devices` | GPU indices | from config |
| `--batch_size` | Batch size | from config |
| `--learning_rate` | Learning rate | from config |
| `--max_epochs` | Training epochs | from config |
| `--precision` | Training precision | bf16-mixed |
| `--seed` | Random seed | from config |
| `--logger` | `csv`, `tensorboard`, or `wandb` | csv |
| `--eval_only` | Test evaluation only (no training) | off |
| `--root_dir` | Folder-based dataset root (overrides `data.root_dir`) | from config |
| `--num_workers` | DataLoader workers (overrides `data.num_workers`) | from config |
| `--fast_dev_run` | Quick sanity check | off |

After training, the checkpoint with the lowest `val_loss` is evaluated on the test set and the metrics are written to `test_results.json`.

#### `eval_knn.py` — k-NN Evaluation

Evaluates feature quality using weighted k-NN classification (no training). Follows the DINO/DINOv3 protocol.

```bash
python scripts/eval_knn.py \
    --config configs/pancreatic/classify.yaml \
    --pretrained_path output/checkpoints/pretraining/pancreatic/last.ckpt \
    --k 20 \
    --temperature 0.07 \
    --batch_size 256 \
    --seeds "0 1 42"
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to classify YAML config **(required)** | — |
| `--pretrained_path` | Pretrained checkpoint **(required)** | — |
| `--checkpoint_type` | `last` or `best` | last |
| `--init_mode` | `scratch` or `continued` | scratch |
| `--encoder_type` | `teacher` or `student` | teacher |
| `--k` | Number of nearest neighbors | 20 |
| `--temperature` | Softmax temperature for voting | 0.07 |
| `--batch_size` | Feature extraction batch size | 256 |
| `--devices` | GPU device index (single GPU) | 0 |
| `--concat_cls_patch` | Concatenate CLS + mean patch tokens | off |
| `--output_base_dir` | Base output directory | output |
| `--seeds` | Evaluation seeds, space-separated | "42" |
| `--root_dir` | Folder-based dataset root (overrides `data.root_dir`) | from config |
| `--num_workers` | DataLoader workers (overrides `data.num_workers`) | from config |

#### `eval_lowshot.py` — K-Shot Evaluation

Trains a classifier with limited labeled data (k examples per class).

```bash
python scripts/eval_lowshot.py \
    --config configs/pancreatic/classify.yaml \
    --pretrained_path output/checkpoints/pretraining/pancreatic/last.ckpt \
    --k_shot 4 \
    --label_seed 0 \
    --train_seed 42
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to classify YAML config **(required)** | — |
| `--pretrained_path` | Pretrained checkpoint **(required)** | — |
| `--k_shot` | Labeled examples per class **(required)** | — |
| `--label_seed` | Seed for label subset selection **(required)** | — |
| `--train_seed` | Seed for training **(required)** | — |
| `--freeze_backbone` | Freeze backbone (linear probing) | off |
| `--batch_size` | Batch size | from config |
| `--learning_rate` | Learning rate | from config |
| `--max_epochs` | Training epochs | 100 |
| `--output_dir` | Base output directory | output/lowshot |
| `--method_tag` | Sub-folder name for the method | auto from checkpoint |
| `--root_dir` | Folder-based dataset root (overrides `data.root_dir`) | from config |
| `--num_workers` | DataLoader workers (overrides `data.num_workers`) | from config |

The baseline methods handled by `scripts/run_lowshot.sh` (`dinov3`, `mae`, `ijepa`, `moca`) are not part of this
repository. Point `DINOV3_WEIGHTS`, `DINOV3_CONTINUED_DIR`, `MAE_OUTPUT_DIR`, `IJEPA_OUTPUT_DIR` and `MOCA_OUTPUT_DIR`
at your own checkpoints; methods whose variable is unset are skipped.

---

---

### Configuration

Configs are YAML files with sections that can be overridden via CLI arguments.

#### Key Hyperparameters

```yaml
model:
  num_prototypes: 128               # Number of prototype vectors (codebook size K)
  projector_dim: 256                # Projection head dimension
  mask_ratio_min_max: [0.1, 0.5]   # Masking ratio range

optimizer:
  learning_rate: 1.0e-4             # Peak LR (with sqrt_wrt_1024 scaling)
  min_lr: 1.0e-6                    # Final LR after cosine decay
  warmup_epochs: 10                 # Linear warmup
  weight_decay: 0.04                # Start weight decay
  weight_decay_end: 0.4             # End weight decay (cosine)
  ema_momentum: 0.992               # Teacher EMA start
  ema_momentum_end: 1.0             # Teacher EMA end
  layerwise_decay: 0.9              # Layer-wise LR decay

loss:
  koleo_loss_weight: 0.1            # KoLeo regularization weight
  prototype_cls_loss_weight: 1.0    # Multi-crop CLS distillation weight
  teacher_temp: 0.04                # Teacher softmax temperature
  sinkhorn_iters: 3                 # Sinkhorn-Knopp iterations

training:
  precision: bf16-mixed
  gradient_clip_val: 3.0
  strategy: ddp
```

---

---
