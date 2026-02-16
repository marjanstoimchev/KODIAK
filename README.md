# KODIAK

PyTorch Lightning implementation of **KODIAK** — a self-supervised learning method combining masked image modeling with prototype learning using a teacher-student architecture.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full Pipeline](#full-pipeline)
- [Training Modes](#training-modes)
  - [From Scratch](#from-scratch)
  - [Continued from DINOv3](#continued-from-dinov3)
  - [Classification Only (Skip Pretraining)](#classification-only-skip-pretraining)
- [Scripts Reference](#scripts-reference)
  - [train.py — Pretraining](#trainpy--pretraining)
  - [train_classifier.py — Classification](#train_classifierpy--classification)
  - [eval_knn.py — k-NN Evaluation](#eval_knnpy--k-nn-evaluation)
  - [eval_lowshot.py — K-Shot Evaluation](#eval_lowshotpy--k-shot-evaluation)
  - [run_sweep.sh — Full Pipeline](#run_sweepsh--full-pipeline)
  - [run_knn.sh — k-NN Wrapper](#run_knnsh--k-nn-wrapper)
  - [run_ablation.sh — Ablation Studies](#run_ablationsh--ablation-studies)
  - [run_lowshot.sh — K-Shot Wrapper](#run_lowshotsh--k-shot-wrapper)
- [SLURM Scripts](#slurm-scripts)
  - [run_experiment.sh — Single SLURM Job](#run_experimentsh--single-slurm-job)
  - [run_proto_analysis.sh — Prototype Sweep](#run_proto_analysissh--prototype-sweep)
  - [run_sweep.sh (SLURM)](#run_sweepsh-slurm)
  - [run_ablation.sh (SLURM)](#run_ablationsh-slurm)
- [Supported Datasets](#supported-datasets)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Logging](#logging)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Installation

```bash
git clone https://github.com/marjanstoimchev/KODIAK.git
cd KODIAK
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- CUDA-capable GPU(s)

---

## Quick Start

### Pretraining

```bash
python scripts/train.py \
    --config configs/DTD/pretrain.yaml \
    --devices 0,1 \
    --logger tensorboard
```

### Classification

```bash
python scripts/train_classifier.py \
    --config configs/DTD/classify.yaml \
    --pretrained_path output/checkpoints/pretraining/dtd/last.ckpt \
    --logger tensorboard
```

### k-NN Evaluation

```bash
python scripts/eval_knn.py \
    --config configs/DTD/classify.yaml \
    --pretrained_path output/checkpoints/pretraining/dtd/last.ckpt
```

---

## Full Pipeline

The `run_sweep.sh` script runs the complete pretrain + classify pipeline in one command:

```bash
./scripts/run_sweep.sh \
    --dataset dtd \
    --gpus 0,1 \
    --pretrain-epochs 500 \
    --classify-epochs 100 \
    --batch-size 128 \
    --seeds "0 1 42" \
    --logger tensorboard
```

This works with conda, venv, or any local Python environment — no container needed.

---

## Training Modes

KODIAK supports three training workflows via the `--init-mode` and `--skip-pretrain` flags:

### From Scratch

Train from random initialization (default):

```bash
./scripts/run_sweep.sh \
    --dataset dtd \
    --gpus 0,1 \
    --init-mode scratch \
    --pretrain-epochs 500 \
    --classify-epochs 100 \
    --batch-size 128 \
    --seeds "0 1 42" \
    --logger tensorboard
```

Uses `configs/{dataset}/pretrain.yaml`. The model is initialized randomly, then pretrained with KODIAK's self-supervised objective, followed by classification.

### Continued from DINOv3

Initialize from DINOv3 pretrained weights, then continue pretraining on the target dataset:

```bash
./scripts/run_sweep.sh \
    --dataset dtd \
    --gpus 0,1 \
    --init-mode continued \
    --pretrain-epochs 100 \
    --classify-epochs 100 \
    --batch-size 128 \
    --seeds "0 1 42" \
    --logger tensorboard
```

Uses `configs/{dataset}/pretrain_continued.yaml`, which specifies the DINOv3 weights path:

```yaml
model:
  pretrained_path: "dinov3_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
```

The DINOv3 weights are loaded into the student encoder, the teacher is initialized from the student via EMA, and pretraining continues on the target dataset. Since the model starts from strong features, fewer epochs are needed (default: 100 vs 300 for scratch).

**Requirements:** Download the DINOv3 ViT-S/16 weights and place them at `dinov3_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` (or update the path in the config).

### Classification Only (Skip Pretraining)

Skip pretraining entirely and run classification from an existing checkpoint:

```bash
# From a specific checkpoint file
./scripts/run_sweep.sh \
    --dataset dtd \
    --gpus 0,1 \
    --skip-pretrain \
    --pretrained-path /path/to/checkpoint.ckpt \
    --classify-epochs 100 \
    --classify-lr 1e-4 \
    --batch-size 128 \
    --seeds "0 1 42" \
    --logger tensorboard

# Auto-detect checkpoint from a previous run (looks in the expected output directory)
./scripts/run_sweep.sh \
    --dataset dtd \
    --gpus 0,1 \
    --skip-pretrain \
    --classify-epochs 100 \
    --seeds "0 1 42"
```

| Flag | Description |
|------|-------------|
| `--skip-pretrain` | Skips pretraining entirely, jumps to classification |
| `--pretrained-path` | Explicit path to a `.ckpt` checkpoint file. If omitted, the script looks for a checkpoint in the expected output directory from a prior pretraining run. |

### Mode Comparison

| Mode | Config | Weights Init | Default Epochs | Output Dir |
|------|--------|-------------|----------------|------------|
| `scratch` | `pretrain.yaml` | Random | 300 | `classification/` |
| `continued` | `pretrain_continued.yaml` | DINOv3 ViT-S/16 | 100 | `classification_from_continued/` |
| `--skip-pretrain` | — | From checkpoint | — | Based on init-mode used during pretraining |

---

## Scripts Reference

### `train.py` — Pretraining

Self-supervised pretraining with KODIAK. Uses a ViT backbone with a teacher-student architecture, masked image modeling, prototype learning, and KoLeo regularization.

```bash
python scripts/train.py \
    --config configs/DTD/pretrain.yaml \
    --devices 0,1,2,3 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --max_epochs 500 \
    --num_prototypes 4096 \
    --koleo_weight 0.1 \
    --cls_weight 1.0 \
    --multi_crop \
    --compile \
    --logger tensorboard \
    --name my_experiment
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config **(required)** | — |
| `--devices` | GPU indices, comma-separated (e.g., `0,1,2,3`) or `auto` | from config |
| `--batch_size` | Total training batch size (divided across GPUs) | from config |
| `--learning_rate` | Peak learning rate | from config |
| `--max_epochs` | Number of training epochs | from config |
| `--num_prototypes` | Number of prototype vectors | 4096 |
| `--koleo_weight` | KoLeo regularization loss weight | 0.1 |
| `--cls_weight` | Multi-crop CLS distillation loss weight | from config |
| `--multi_crop` | Enable multi-crop augmentation (2 global + 8 local crops) | off |
| `--no_sinkhorn` | Use softmax instead of Sinkhorn-Knopp normalization | off |
| `--compile` | Enable `torch.compile()` for ~15-30% speedup | off |
| `--no_compile` | Explicitly disable `torch.compile()` | — |
| `--precision` | Training precision (e.g., `bf16-mixed`, `32`) | bf16-mixed |
| `--logger` | Logger type: `csv`, `tensorboard`, or `wandb` | csv |
| `--name` | Experiment name (used in folder paths) | from config |
| `--resume` | Resume training from checkpoint path | — |
| `--pretrained_path` | Initialize from pretrained weights | — |
| `--save_every_n_epochs` | Save intermediate checkpoints every N epochs | — |
| `--log_dir` | Exact log directory (full path) | auto |
| `--log_base_dir` | Base directory for logs | auto |
| `--checkpoint_dir` | Exact checkpoint directory (full path) | auto |
| `--checkpoint_base_dir` | Base directory for checkpoints | auto |
| `--fast_dev_run` | Quick sanity check (1 batch) | off |

### `train_classifier.py` — Classification

Fine-tuning or linear evaluation using a pretrained KODIAK encoder.

```bash
python scripts/train_classifier.py \
    --config configs/DTD/classify.yaml \
    --pretrained_path output/checkpoints/pretraining/dtd/last.ckpt \
    --devices 0,1 \
    --max_epochs 100 \
    --learning_rate 1e-4 \
    --batch_size 128 \
    --seed 42 \
    --encoder_type teacher \
    --logger tensorboard
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config **(required)** | — |
| `--pretrained_path` | Pretrained checkpoint file or directory | — |
| `--checkpoint_type` | Which checkpoint to load: `last` or `best` (lowest loss) | last |
| `--encoder_type` | Encoder to extract from SSL checkpoint: `teacher` or `student` | teacher |
| `--freeze_backbone` | Freeze encoder for linear evaluation | off |
| `--use_cls_token` | Use CLS token (`true`) or mean patch pooling (`false`) | from config |
| `--concat_cls_patch` | Concatenate CLS + mean patch tokens (2x features) | off |
| `--devices` | GPU indices | from config |
| `--batch_size` | Batch size | from config |
| `--learning_rate` | Learning rate | from config |
| `--max_epochs` | Training epochs | from config |
| `--precision` | Training precision | bf16-mixed |
| `--seed` | Random seed | from config |
| `--logger` | Logger type: `csv`, `tensorboard`, or `wandb` | csv |
| `--name` | Experiment name | from config |
| `--resume` | Resume from checkpoint | — |
| `--eval_only` | Run test evaluation only (no training) | off |
| `--exp_base_dir` | Base directory for both logs and checkpoints | — |
| `--log_dir` | Exact log directory (full path) | auto |
| `--log_base_dir` | Base directory for logs | auto |
| `--checkpoint_dir` | Exact checkpoint directory (full path) | auto |
| `--checkpoint_base_dir` | Base directory for checkpoints | auto |
| `--fast_dev_run` | Quick sanity check | off |

### `eval_knn.py` — k-NN Evaluation

Evaluates pretrained feature quality using weighted k-NN classification (no training needed). Follows the standard DINO/DINOv3 protocol.

```bash
python scripts/eval_knn.py \
    --config configs/DTD/classify.yaml \
    --pretrained_path output/checkpoints/pretraining/dtd/last.ckpt \
    --k 20 \
    --temperature 0.07 \
    --batch_size 256 \
    --seeds "0 1 42"
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to classify YAML config **(required)** | — |
| `--pretrained_path` | Pretrained SSL checkpoint **(required)** | — |
| `--checkpoint_type` | `last` or `best` | last |
| `--init_mode` | Pretraining mode: `scratch` or `continued` | scratch |
| `--encoder_type` | `teacher` or `student` | teacher |
| `--k` | Number of nearest neighbors | 20 |
| `--temperature` | Softmax temperature for voting | 0.07 |
| `--batch_size` | Batch size for feature extraction | 256 |
| `--devices` | GPU device index (single GPU) | 0 |
| `--concat_cls_patch` | Concatenate CLS + mean patch tokens | off |
| `--output_dir` | Override output directory | auto |
| `--output_base_dir` | Base output directory | output |
| `--seeds` | Evaluation seeds, space-separated | "42" |

**Output:** `output/knn/{dataset}/{pretrain_folder}/knn_results.json`

### `eval_lowshot.py` — K-Shot Evaluation

Trains a classifier with limited labeled data (k examples per class).

```bash
python scripts/eval_lowshot.py \
    --config configs/DTD/classify.yaml \
    --pretrained_path output/checkpoints/pretraining/dtd/last.ckpt \
    --k_shot 4 \
    --label_seed 0 \
    --train_seed 42
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to classify YAML config **(required)** | — |
| `--pretrained_path` | Pretrained SSL checkpoint **(required)** | — |
| `--k_shot` | Number of labeled examples per class **(required)** | — |
| `--label_seed` | Seed for label subset selection **(required)** | — |
| `--train_seed` | Seed for training **(required)** | — |
| `--checkpoint_type` | `last` or `best` | last |
| `--encoder_type` | `teacher` or `student` | from config |
| `--freeze_backbone` | Freeze backbone (linear probing) | off |
| `--batch_size` | Batch size | from config |
| `--learning_rate` | Learning rate | from config |
| `--max_epochs` | Training epochs | from config |
| `--precision` | Training precision | from config |
| `--output_dir` | Output directory | output/lowshot |
| `--method_tag` | Method name for folder structure | — |
| `--fast_dev_run` | Quick sanity check | off |

---

### `run_sweep.sh` — Full Pipeline

Runs the complete pretrain + classify pipeline. Supports training from scratch and continued pretraining from DINOv3.

```bash
./scripts/run_sweep.sh \
    --dataset dtd \
    --gpus 0,1,2,3 \
    --init-mode scratch \
    --pretrain-epochs 500 \
    --pretrain-lr 1e-4 \
    --classify-epochs 100 \
    --classify-lr 1e-4 \
    --classify-mode finetune \
    --batch-size 128 \
    --num-prototypes 4096 \
    --koleo-weight 0.1 \
    --cls-weight 1.0 \
    --seeds "0 1 42" \
    --multi-crop \
    --compile \
    --logger tensorboard \
    --output-dir output
```

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Dataset name (see [Supported Datasets](#supported-datasets)) **(required)** | — |
| `--gpus` | GPU indices, comma-separated **(required)** | — |
| `--init-mode` | `scratch` or `continued` (from DINOv3) | scratch |
| `--pretrain-epochs` | Pretraining epochs | 300 (scratch) / 100 (continued) |
| `--pretrain-lr` | Pretraining learning rate | 1e-4 |
| `--pretrain-seed` | Pretraining seed | 42 |
| `--classify-epochs` | Classification epochs | 100 |
| `--classify-lr` | Classification learning rate | 1e-4 |
| `--classify-mode` | `finetune` or `lineareval` | finetune |
| `--batch-size` | Batch size | 128 |
| `--num-prototypes` | Number of prototypes | 4096 |
| `--koleo-weight` | KoLeo loss weight | 0.1 |
| `--cls-weight` | CLS loss weight | 1.0 |
| `--seeds` | Classification seeds (space-separated in quotes) | "0 1 42" |
| `--multi-crop` | Enable multi-crop augmentation | off |
| `--compile` | Enable `torch.compile()` for ~15-30% speedup | off |
| `--no-compile` | Disable `torch.compile()` | — |
| `--concat-cls-patch` | Concatenate CLS + mean patch tokens | off |
| `--logger` | Logger type: `csv`, `tensorboard`, `wandb` | csv |
| `--output-dir` | Output directory | output |
| `--save-every` | Save checkpoint every N epochs | from config |
| `--skip-pretrain` | Skip pretraining (use existing checkpoint) | off |
| `--skip-classify` | Skip classification | off |

### `run_knn.sh` — k-NN Wrapper

Convenience wrapper that auto-finds checkpoints by experiment name. Loops over multiple prototype counts.

```bash
./scripts/run_knn.sh \
    --dataset dtd \
    --gpus 0 \
    --init-mode scratch \
    --num-prototypes "256 1024 4096" \
    --koleo-weight 0.1 \
    --cls-weight 1.0 \
    --multi-crop \
    --k 20 \
    --temperature 0.07 \
    --batch-size 256 \
    --checkpoint-type last \
    --output-base-dir output \
    --seeds "0 1 42"
```

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Dataset name **(required)** | — |
| `--gpus` | GPU index (single GPU) **(required)** | — |
| `--init-mode` | `scratch` or `continued` | scratch |
| `--num-prototypes` | Prototype counts (space-separated in quotes) | "4096" |
| `--koleo-weight` | KoLeo weight | 0.1 |
| `--cls-weight` | CLS weight | 1.0 |
| `--multi-crop` / `--no-multi-crop` | Multi-crop flag | on |
| `--concat-cls-patch` | Concatenate CLS + patch tokens | off |
| `--k` | Number of nearest neighbors | 20 |
| `--temperature` | Softmax temperature | 0.07 |
| `--batch-size` | Batch size for feature extraction | 256 |
| `--checkpoint-type` | `last` or `best` | last |
| `--pretrained-path` | Override checkpoint path (single run) | auto |
| `--output-base-dir` | Base output directory | output |
| `--seeds` | Evaluation seeds (space-separated) | "0 1 42" |

**Output:** `output/knn/{dataset}/{pretrain_folder}/knn_results.json`

### `run_ablation.sh` — Ablation Studies

Runs ablation experiments by selectively disabling components.

```bash
./scripts/run_ablation.sh \
    --dataset dtd \
    --gpus 0,1 \
    --ablations "full no_sinkhorn no_cls_loss no_koleo" \
    --pretrain-epochs 500 \
    --classify-epochs 100 \
    --batch-size 128 \
    --num-prototypes 1024 \
    --output-dir ablations
```

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Dataset name **(required)** | — |
| `--gpus` | GPU indices **(required)** | — |
| `--init-mode` | `scratch` or `continued` | scratch |
| `--ablations` | Ablation names (space-separated in quotes) | "full no_sinkhorn no_cls_loss no_koleo" |
| `--seeds` | Classification seeds | "0 1 42" |
| `--pretrain-epochs` | Pretraining epochs | 500 |
| `--pretrain-lr` | Pretraining learning rate | 1e-4 |
| `--classify-epochs` | Classification epochs | 100 |
| `--classify-lr` | Classification learning rate | 1e-4 |
| `--classify-mode` | `finetune` or `lineareval` | finetune |
| `--batch-size` | Batch size | 128 |
| `--num-prototypes` | Number of prototypes | 1024 |
| `--koleo-weight` | KoLeo loss weight | 0.1 |
| `--cls-weight` | CLS loss weight | 1.0 |
| `--output-dir` | Output directory | ablations |

**Ablation options:**

| Ablation | Description |
|----------|-------------|
| `full` | All components enabled (baseline) |
| `no_sinkhorn` | Softmax instead of Sinkhorn-Knopp normalization |
| `no_cls_loss` | Disable multi-crop CLS distillation loss |
| `no_koleo` | Disable KoLeo regularization |

### `run_lowshot.sh` — K-Shot Wrapper

Runs few-shot evaluation across multiple methods, datasets, and shot counts.

```bash
./scripts/run_lowshot.sh \
    --gpus 0 \
    --methods "kodiak dinov3" \
    --datasets "eurosat dtd oxford_pets" \
    --shots "1 2 4 8 16" \
    --label-seeds "0 1 42" \
    --train-seeds "42" \
    --mode finetune \
    --init-mode continued
```

| Flag | Description | Default |
|------|-------------|---------|
| `--gpus` | GPU index **(required)** | — |
| `--methods` | Methods to evaluate (space-separated) | "kodiak dinov3 mae ijepa moca" |
| `--datasets` | Datasets to evaluate (space-separated) | "eurosat dtd oxford_pets" |
| `--shots` | K-shot values (space-separated) | "1 2 4 8 16" |
| `--label-seeds` | Seeds for label subset selection | "0 1 42" |
| `--train-seeds` | Seeds for training | "42" |
| `--mode` | `finetune`, `lineareval`, or both (empty) | both |
| `--init-mode` | `scratch` or `continued` | continued |
| `--max-epochs` | Override max epochs | from config |
| `--batch-size` | Override batch size | from config |
| `--learning-rate` | Override learning rate | from config |
| `--output-dir` | Override output directory | auto |
| `--dry-run` | Print commands without executing | off |

---

## SLURM Scripts

All SLURM scripts are in `scripts/slurm/`. They wrap the local scripts and support optional Singularity containers.

### `run_experiment.sh` — Single SLURM Job

Runs one full pretrain + classify experiment as a SLURM job. All parameters are set via environment variables.

```bash
# Without container (uses conda/system Python)
NUM_PROTOTYPES=128 DATASET=dtd \
    sbatch scripts/slurm/run_experiment.sh

# With Singularity container
NUM_PROTOTYPES=256 DATASET=eurosat \
    USE_SINGULARITY=true SIF_IMAGE=/path/to/deeplearning.sif \
    sbatch scripts/slurm/run_experiment.sh
```

**SLURM defaults:** 1 node, 2 GPUs, 22 CPUs, 32GB RAM, 4 days, `gpu` partition.

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `NUM_PROTOTYPES` | Number of prototypes **(required)** | — |
| `DATASET` | Dataset name | dtd |
| `INIT_MODE` | `scratch` or `continued` | scratch |
| `SEEDS` | Classification seeds (space-separated) | "0 1 42" |
| `PRETRAIN_SEED` | Pretraining seed | 42 |
| `PRETRAIN_EPOCHS` | Pretraining epochs | 500 |
| `PRETRAIN_LR` | Pretraining learning rate | 0.0001 |
| `CLASSIFY_EPOCHS` | Classification epochs | 100 |
| `CLASSIFY_LR` | Classification learning rate | 0.0001 |
| `CLASSIFY_MODE` | `finetune` or `lineareval` | finetune |
| `BATCH_SIZE` | Batch size | 128 |
| `KOLEO_WEIGHT` | KoLeo loss weight | 0.1 |
| `CLS_WEIGHT` | CLS loss weight | 1.0 |
| `MULTI_CROP` | Enable multi-crop (`true`/`false`) | true |
| `COMPILE` | Enable `torch.compile()` (`true`/`false`) | false |
| `OUTPUT_DIR` | Output directory | output_proto_analysis |
| `SKIP_PRETRAIN` | Skip pretraining (`true`/`false`) | false |
| `SKIP_CLASSIFY` | Skip classification (`true`/`false`) | false |
| `PRETRAINED_PATH` | Path to pretrained checkpoint | — |
| `SAVE_EVERY_N_EPOCHS` | Checkpoint every N epochs | — |
| `USE_SINGULARITY` | Use Singularity container (`true`/`false`) | false |
| `SIF_IMAGE` | Path to Singularity `.sif` image | `$HOME/deeplearning.sif` |

### `run_proto_analysis.sh` — Prototype Sweep

Submits multiple SLURM jobs sweeping over prototype counts (128, 256, 512, 1024, 2048, 4096).

```bash
bash scripts/slurm/run_proto_analysis.sh
```

Edit the script to customize the sweep parameters (dataset, epochs, batch size, etc.).

### `run_sweep.sh` (SLURM)

SLURM wrapper for a single sweep job with configurable resources.

```bash
DATASET=eurosat PRETRAIN_EPOCHS=500 sbatch scripts/slurm/run_sweep.sh
```

Same environment variables as `run_experiment.sh`.

### `run_ablation.sh` (SLURM)

SLURM wrapper for ablation experiments. Requires `ABLATION` environment variable.

```bash
ABLATION=no_koleo DATASET=dtd sbatch scripts/slurm/run_ablation.sh
```

| Additional Variable | Description | Default |
|--------------------|-------------|---------|
| `ABLATION` | Ablation type **(required)**: `full`, `no_sinkhorn`, `no_cls_loss`, `no_koleo` | — |

---

## Supported Datasets

| Dataset | Config Directory | Type | Classes |
|---------|-----------------|------|---------|
| DTD | `configs/DTD/` | HuggingFace | 47 |
| EuroSAT | `configs/eurosat/` | HuggingFace | 10 |
| Oxford Pets | `configs/oxford_pets/` | HuggingFace | 37 |
| CIFAR-100 | `configs/cifar100/` | HuggingFace | 100 |
| NCTCRCHE100K | `configs/NCTCRCHE100K/` | HuggingFace | 9 |
| ImageNet-1K | `configs/imagenet1k/` | HuggingFace | 1000 |
| Tissue | `configs/tissue/` | Custom (CSV) | varies |

Each dataset directory contains:
- `pretrain.yaml` — Self-supervised pretraining config
- `pretrain_continued.yaml` — Continued pretraining from DINOv3
- `classify.yaml` — Classification config

---

## Configuration

Configs are YAML files with sections that can be overridden via CLI arguments.

### Key Hyperparameters

```yaml
model:
  num_prototypes: 4096              # Number of prototype vectors
  projector_dim: 256                # Projection head dimension
  mask_ratio_min_max: [0.1, 0.5]   # Masking ratio range
  mask_sample_probability: 0.5

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
  precision: bf16-mixed             # Mixed precision training
  gradient_clip_val: 3.0
  strategy: ddp                     # Distributed data parallel
```

---

## Output Structure

All outputs are organized under the output directory (default: `output/`):

```
output/
├── logs/
│   ├── pretraining/{dataset}/{experiment_name}/
│   │   ├── version_0/              # TensorBoard events or CSV metrics
│   │   ├── config.yaml             # Saved config snapshot
│   │   └── training_summary.json   # GPU hours, throughput, timing
│   └── classification/{dataset}/{pretrain_folder}/
│       └── finetune_seed_42/
│           ├── version_0/
│           ├── config.yaml
│           ├── training_summary.json
│           └── test_results.json   # Final test metrics
├── checkpoints/
│   ├── pretraining/{dataset}/{experiment_name}/
│   │   └── last.ckpt
│   └── classification/{dataset}/{pretrain_folder}/
│       └── finetune_seed_42/
│           ├── last.ckpt
│           └── classifier-epoch=XX-val_loss=X.XXXX.ckpt
└── knn/{dataset}/{pretrain_folder}/
    └── knn_results.json
```

### `training_summary.json`

Automatically saved at the end of each training run with timing information:

```json
{
  "total_wall_time_sec": 6150.3,
  "total_wall_time": "1h 42m 30s",
  "total_gpu_hours": 3.417,
  "num_gpus": 2,
  "num_epochs": 500,
  "avg_epoch_time_sec": 12.3,
  "throughput_samples_per_sec": 845.2,
  "gpu_model": "NVIDIA A100-SXM4-40GB"
}
```

### `test_results.json`

Saved after classification with all evaluation metrics:

```json
{
  "dataset": "dtd",
  "mode": "finetune",
  "seed": 42,
  "test_acc": 0.7523,
  "test_f1_macro": 0.7412,
  "test_auroc": 0.9801,
  "test_precision_macro": 0.7534,
  "test_recall_macro": 0.7412
}
```

---

## Logging

KODIAK supports three logger backends, selected with `--logger`:

| Logger | Flag | Viewer |
|--------|------|--------|
| CSV | `--logger csv` (default) | Open `metrics.csv` directly |
| TensorBoard | `--logger tensorboard` | `tensorboard --logdir output/logs` |
| Weights & Biases | `--logger wandb` | wandb.ai dashboard |

### TensorBoard

```bash
# Start TensorBoard (on a remote server, add --bind_all)
tensorboard --logdir output/logs --bind_all

# If behind a proxy, bypass it for localhost
no_proxy=127.0.0.0/8,localhost tensorboard --logdir output/logs --bind_all
```

Then SSH tunnel from your local machine and open `http://localhost:6006`:

```bash
ssh -L 6006:localhost:6006 user@server
```

### Logged Metrics

**Pretraining** (per step and per epoch):
- `train_loss` — total loss
- `train_mask` — masked image modeling loss
- `train_koleo` — KoLeo diversity loss
- `train_proto_cls` — prototype classification loss
- `train_cls_global` / `train_cls_local` — global/local CLS losses
- `train_util` — prototype utilization (unique prototypes used)
- `lr`, `last_layer_lr` — learning rates
- `wd`, `mom`, `teacher_temp` — schedule values
- `epoch_time_sec`, `gpu_hours` — timing

**Classification** (per step/epoch):
- `train_loss`, `train_acc`, `train_acc_top5`
- `val_loss`, `val_acc`, `val_acc_top5`
- `test_acc`, `test_f1_macro`, `test_auroc`, etc.
- `epoch_time_sec`, `gpu_hours`

---

## Project Structure

```
KODIAK/
├── src/
│   ├── models/                  # Model architectures
│   │   ├── kodiak.py            # Main KODIAK model (teacher-student)
│   │   ├── classifier.py        # Classification head
│   │   └── components.py        # ViT encoder, projection heads
│   ├── learners/                # PyTorch Lightning modules
│   │   ├── pretraining.py       # Self-supervised pretraining learner
│   │   └── classification.py    # Classification learner
│   ├── losses/                  # Loss functions
│   │   ├── losses.py            # MaskLoss, KoLeoLoss, Sinkhorn-Knopp
│   │   └── multicrop_prototype_cls_loss.py  # Multi-crop CLS distillation
│   ├── data/                    # Data loading
│   │   ├── pretraining/         # Pretraining DataModule, transforms, collate
│   │   ├── classification/      # Classification DataModule, transforms
│   │   ├── datasets/            # HuggingFace and custom dataset wrappers
│   │   └── utils/               # Samplers, registry
│   ├── utils/                   # Configuration and logging
│   │   ├── config.py            # YAML config loader with CLI overrides
│   │   ├── config_dataclasses.py # Typed config dataclasses
│   │   └── loggers.py           # CSV, TensorBoard, WandB logger setup
│   └── callbacks/               # Training callbacks
│       ├── timer.py             # TrainingTimer (GPU hours, epoch time)
│       └── progress_bar.py      # Custom progress bar
├── scripts/
│   ├── train.py                 # Pretraining entry point
│   ├── train_classifier.py      # Classification entry point
│   ├── eval_knn.py              # k-NN evaluation
│   ├── eval_lowshot.py          # K-shot evaluation
│   ├── run_sweep.sh             # Full pretrain + classify pipeline
│   ├── run_knn.sh               # k-NN evaluation wrapper
│   ├── run_ablation.sh          # Ablation experiments
│   ├── run_lowshot.sh           # K-shot evaluation wrapper
│   ├── submit_ablations.sh      # Submit all ablation SLURM jobs
│   ├── aggregate_lowshot.py     # Aggregate lowshot results into tables
│   └── slurm/                   # SLURM job scripts
│       ├── run_experiment.sh    # Single experiment (env var config)
│       ├── run_proto_analysis.sh # Prototype count sweep
│       ├── run_sweep.sh         # SLURM sweep wrapper
│       └── run_ablation.sh      # SLURM ablation wrapper
├── configs/                     # Per-dataset YAML configs
│   ├── DTD/
│   ├── eurosat/
│   ├── oxford_pets/
│   ├── cifar100/
│   ├── NCTCRCHE100K/
│   ├── imagenet1k/
│   └── tissue/
├── dinov3/                      # DINOv3 library (backbone, utilities)
└── requirements.txt
```

---

## Citation

```bibtex
@article{kodiak2025,
  title={KODIAK: Self-Supervised Learning with Masked Prototype Prediction},
  author={Stoimchev, Marjan and ...},
  year={2025}
}
```
