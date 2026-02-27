# KODIAK

PyTorch Lightning implementation of **KODIAK** — a self-supervised learning method combining masked image modeling with prototype learning using a teacher-student architecture.

## Architecture

Uses a **DINOv3 ViT-Small/16** backbone:
- Embedding dimension: 384
- Depth: 12 blocks
- Attention heads: 6
- Positional encoding: RoPE
- Register tokens: 4

## Supported Datasets

| Dataset | Type | Classes | Source |
|---------|------|---------|--------|
| **Pancreatic** | Custom folder | 7 | Local: `/home/marjans/Datasets/pancreatic/SLIDE-3210` |
| CIFAR-100 | HuggingFace | 100 | `uoft-cs/cifar100` |
| DTD | HuggingFace | 47 | `cansa/Describable-Textures-Dataset-DTD` |
| EuroSAT | HuggingFace | 10 | `blanchon/EuroSAT_RGB` |
| Oxford Pets | HuggingFace | 37 | `timm/oxford-iiit-pet` |
| NCT-CRC-HE-100K | HuggingFace | 9 | `DykeF/NCTCRCHE100K` |
| ImageNet-1K | HuggingFace | 1000 | `ILSVRC/imagenet-1k` |

See [DATASETS.md](DATASETS.md) for full dataset documentation and per-dataset copy-paste commands.

## Installation

```bash
git clone https://github.com/marjanstoimchev/KODIAK.git
cd KODIAK
pip install -r requirements.txt
```

---

## Quick Start (Copy-Paste Examples)

All examples below use SLURM with a Singularity container (`deeplearning.sif`). Remove `USE_SINGULARITY=true` to run without a container.

### Recommended Hyperparameters

| Setting | Value |
|---------|-------|
| Pretraining from scratch | 500 epochs |
| Pretraining continued (from DINOv3 weights) | 100 epochs |
| Classification fine-tuning (after scratch) | 100 epochs, LR = 1e-4 |
| Classification fine-tuning (after continued) | 50 epochs, LR = 1e-4 |
| Classification linear eval (any source) | 50 epochs, LR = 1e-3 |
| Batch size (HuggingFace datasets) | 128 |
| Batch size (Pancreatic, 256x256) | 64 |

---

### 1. Full Pipeline: Pretrain + Classify

The pipeline script runs pretraining followed by multi-seed classification in a single job.

#### From Scratch (500 pretrain epochs + 100 classify epochs)

**Fine-tuning:**

```bash
# Pancreatic — fine-tuning after scratch pretraining
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
CLASSIFY_MODE=finetune \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh
```

**Linear evaluation:**

```bash
# CIFAR-100 — linear eval after scratch pretraining
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=cifar100 \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=50 \
CLASSIFY_LR=1e-3 \
CLASSIFY_MODE=lineareval \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh
```

#### Continued Pretraining (100 pretrain epochs + 50 classify epochs)

Continued pretraining initializes from DINOv3 off-the-shelf weights.

**Fine-tuning:**

```bash
# Pancreatic — fine-tuning after continued pretraining
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=continued \
PRETRAIN_EPOCHS=100 \
CLASSIFY_EPOCHS=50 \
CLASSIFY_LR=1e-4 \
CLASSIFY_MODE=finetune \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh
```

**Linear evaluation:**

```bash
# DTD — linear eval after continued pretraining
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=dtd \
INIT_MODE=continued \
PRETRAIN_EPOCHS=100 \
CLASSIFY_EPOCHS=50 \
CLASSIFY_LR=1e-3 \
CLASSIFY_MODE=lineareval \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh
```

---

### 2. Pretraining Only

```bash
# From scratch — 500 epochs
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
SKIP_CLASSIFY=true \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh

# Continued from DINOv3 weights — 100 epochs
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=continued \
PRETRAIN_EPOCHS=100 \
SKIP_CLASSIFY=true \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh
```

---

### 3. Classification Only (Skip Pretraining)

Requires an existing pretrained checkpoint.

**Fine-tuning (from scratch checkpoint):**

```bash
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=scratch \
SKIP_PRETRAIN=true \
CLASSIFY_MODE=finetune \
CLASSIFY_LR=1e-4 \
CLASSIFY_EPOCHS=100 \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh
```

**Linear evaluation (from scratch checkpoint):**

```bash
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=scratch \
SKIP_PRETRAIN=true \
CLASSIFY_MODE=lineareval \
CLASSIFY_LR=1e-3 \
CLASSIFY_EPOCHS=50 \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh
```

**From continued checkpoint:**

```bash
# Fine-tuning from continued checkpoint
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=continued \
SKIP_PRETRAIN=true \
CLASSIFY_MODE=finetune \
CLASSIFY_LR=1e-4 \
CLASSIFY_EPOCHS=50 \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh

# Linear eval from continued checkpoint
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=continued \
SKIP_PRETRAIN=true \
CLASSIFY_MODE=lineareval \
CLASSIFY_LR=1e-3 \
CLASSIFY_EPOCHS=50 \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh
```

---

### 4. k-NN Evaluation

k-NN uses frozen features (no training). Evaluates representation quality directly.

```bash
# k-NN from scratch checkpoint
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=scratch \
OUTPUT_BASE_DIR=output_proto_analysis \
sbatch scripts/slurm/run_knn.sh

# k-NN from continued checkpoint
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=continued \
OUTPUT_BASE_DIR=output_proto_analysis \
sbatch scripts/slurm/run_knn.sh
```

---

### 5. Ablation Studies

See [ABLATIONS.md](ABLATIONS.md) for full ablation documentation.

```bash
# Run all ablations on Pancreatic
for ABL in full no_sinkhorn no_cls_loss no_koleo; do
    ABLATION=$ABL \
    DATASET=pancreatic \
    PRETRAIN_EPOCHS=500 \
    CLASSIFY_EPOCHS=100 \
    BATCH_SIZE=64 \
    NUM_PROTOTYPES=1024 \
    USE_SINGULARITY=true \
    sbatch scripts/slurm/run_ablation.sh
done
```

---

## Non-SLURM Usage

All scripts have non-SLURM equivalents. Use these on a machine with GPUs directly.

### 1. Full Pipeline: Pretrain + Classify

#### From Scratch (500 pretrain epochs + 100 classify epochs)

**Fine-tuning:**

```bash
# Pancreatic — fine-tuning after scratch pretraining
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode scratch --pretrain-epochs 500 \
    --classify-epochs 100 --classify-lr 1e-4 --classify-mode finetune \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"
```

**Linear evaluation:**

```bash
# CIFAR-100 — linear eval after scratch pretraining
./scripts/run_sweep.sh --dataset cifar100 --gpus 0,1 \
    --init-mode scratch --pretrain-epochs 500 \
    --classify-epochs 50 --classify-lr 1e-3 --classify-mode lineareval \
    --batch-size 128 --num-prototypes 128 --seeds "0 1 42"
```

#### Continued Pretraining (100 pretrain epochs + 50 classify epochs)

**Fine-tuning:**

```bash
# Pancreatic — fine-tuning after continued pretraining
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode continued --pretrain-epochs 100 \
    --classify-epochs 50 --classify-lr 1e-4 --classify-mode finetune \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"
```

**Linear evaluation:**

```bash
# DTD — linear eval after continued pretraining
./scripts/run_sweep.sh --dataset dtd --gpus 0,1 \
    --init-mode continued --pretrain-epochs 100 \
    --classify-epochs 50 --classify-lr 1e-3 --classify-mode lineareval \
    --batch-size 128 --num-prototypes 128 --seeds "0 1 42"
```

---

### 2. Pretraining Only

```bash
# From scratch — 500 epochs
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode scratch --pretrain-epochs 500 \
    --skip-classify --batch-size 64 --num-prototypes 128

# Continued from DINOv3 weights — 100 epochs
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode continued --pretrain-epochs 100 \
    --skip-classify --batch-size 64 --num-prototypes 128
```

---

### 3. Classification Only (Skip Pretraining)

Requires an existing pretrained checkpoint.

**Fine-tuning (from scratch checkpoint):**

```bash
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode scratch --skip-pretrain \
    --classify-mode finetune --classify-lr 1e-4 --classify-epochs 100 \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"
```

**Linear evaluation (from scratch checkpoint):**

```bash
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode scratch --skip-pretrain \
    --classify-mode lineareval --classify-lr 1e-3 --classify-epochs 50 \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"
```

**From continued checkpoint:**

```bash
# Fine-tuning from continued checkpoint
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode continued --skip-pretrain \
    --classify-mode finetune --classify-lr 1e-4 --classify-epochs 50 \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"

# Linear eval from continued checkpoint
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode continued --skip-pretrain \
    --classify-mode lineareval --classify-lr 1e-3 --classify-epochs 50 \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"
```

---

### 4. k-NN Evaluation

```bash
# k-NN from scratch checkpoint
./scripts/run_knn.sh --dataset pancreatic --gpus 0 \
    --init-mode scratch --num-prototypes 128

# k-NN from continued checkpoint
./scripts/run_knn.sh --dataset pancreatic --gpus 0 \
    --init-mode continued --num-prototypes 128
```

---

### 5. Ablation Studies

```bash
# Run all ablations on Pancreatic
./scripts/run_ablation.sh --dataset pancreatic --gpus 0,1 \
    --ablations "full no_sinkhorn no_cls_loss no_koleo" \
    --pretrain-epochs 500 --classify-epochs 100 \
    --batch-size 64 --num-prototypes 1024 --output-dir ablations
```

---

### 6. Few-Shot Evaluation

```bash
# Fine-tuning — pancreatic, continued, LR 1e-4
./scripts/run_lowshot.sh --gpus 0 --methods "kodiak" --datasets pancreatic \
    --shots "2 4 8 16" --mode finetune --init-mode continued \
    --learning-rate 1e-4 --max-epochs 50 --batch-size 64

# Linear eval — pancreatic, continued, LR 1e-3
./scripts/run_lowshot.sh --gpus 0 --methods "kodiak" --datasets pancreatic \
    --shots "2 4 8 16" --mode lineareval --init-mode continued \
    --learning-rate 1e-3 --max-epochs 50 --batch-size 64
```

---

## Output Structure

### Pretraining

```
output/
└── checkpoints/pretraining/{dataset}/{experiment_name}/
    └── last.ckpt
```

### Classification

```
output/
└── logs/classification/{dataset}/{pretrain_folder}/
    ├── finetune_seed_0/
    │   ├── test_results.json        # final test metrics
    │   └── training_summary.json
    ├── finetune_seed_1/
    ├── finetune_seed_42/
    ├── lineareval_seed_0/           # if classify-mode=lineareval
    ├── lineareval_seed_1/
    └── lineareval_seed_42/
```

### k-NN

```
output/knn/{dataset}/{pretrain_folder}/
└── knn_results.json

output/knn_from_continued/{dataset}/{pretrain_folder}/
└── knn_results.json
```

---

## Evaluation Metrics

Saved in `test_results.json` after each classification run:

| Metric | Description |
|--------|-------------|
| `test_acc` | Top-1 accuracy |
| `test_acc_top5` | Top-5 accuracy (when > 5 classes) |
| `test_f1_macro` | Macro-averaged F1 |
| `test_f1_weighted` | Weighted F1 |
| `test_precision_macro` | Macro-averaged precision |
| `test_recall_macro` | Macro-averaged recall |
| `test_auroc` | Area under ROC curve |

Model selection: best checkpoint by `val_loss` (min), early stopping on `val_loss`.

---

## Scripts Reference

### `train.py` — Pretraining

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
| `--num_prototypes` | Number of prototype vectors | 4096 |
| `--koleo_weight` | KoLeo regularization weight | 0.1 |
| `--cls_weight` | Multi-crop CLS distillation weight | from config |
| `--multi_crop` | Enable multi-crop (2 global + 8 local) | off |
| `--no_sinkhorn` | Use softmax instead of Sinkhorn-Knopp | off |
| `--compile` | Enable `torch.compile()` | off |
| `--precision` | Training precision | bf16-mixed |
| `--logger` | `csv`, `tensorboard`, or `wandb` | csv |
| `--name` | Experiment name | from config |
| `--resume` | Resume from checkpoint | — |
| `--pretrained_path` | Initialize from pretrained weights | — |
| `--save_every_n_epochs` | Save intermediate checkpoints | — |
| `--fast_dev_run` | Quick sanity check (1 batch) | off |

### `train_classifier.py` — Classification

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
| `--fast_dev_run` | Quick sanity check | off |

### `eval_knn.py` — k-NN Evaluation

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

### `eval_lowshot.py` — K-Shot Evaluation

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
| `--max_epochs` | Training epochs | from config |

---

## SLURM Scripts

All SLURM scripts are in `scripts/slurm/`. They wrap the local scripts and support optional Singularity containers.

**SLURM defaults:** 1 node, 2 GPUs, 22 CPUs, 64GB RAM, 4 days, `gpu` partition.

### `run_experiment.sh` — Single SLURM Job

Runs one full pretrain + classify experiment as a SLURM job. All parameters via environment variables.

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `NUM_PROTOTYPES` | Number of prototypes **(required)** | — |
| `DATASET` | Dataset name | dtd |
| `INIT_MODE` | `scratch` or `continued` | scratch |
| `SEEDS` | Classification seeds | "0 1 42" |
| `PRETRAIN_SEED` | Pretraining seed | 42 |
| `PRETRAIN_EPOCHS` | Pretraining epochs | 500 |
| `PRETRAIN_LR` | Pretraining learning rate | 0.0001 |
| `CLASSIFY_EPOCHS` | Classification epochs | 100 |
| `CLASSIFY_LR` | Classification learning rate | 0.0001 |
| `CLASSIFY_MODE` | `finetune` or `lineareval` | finetune |
| `BATCH_SIZE` | Batch size | 128 |
| `KOLEO_WEIGHT` | KoLeo loss weight | 0.1 |
| `CLS_WEIGHT` | CLS loss weight | 1.0 |
| `MULTI_CROP` | Enable multi-crop | true |
| `COMPILE` | Enable `torch.compile()` | false |
| `LOGGER` | `csv`, `tensorboard`, `wandb` | csv |
| `OUTPUT_DIR` | Output directory | output_proto_analysis |
| `SKIP_PRETRAIN` | Skip pretraining | false |
| `SKIP_CLASSIFY` | Skip classification | false |
| `PRETRAINED_PATH` | Checkpoint path (for skip-pretrain) | — |
| `SAVE_EVERY_N_EPOCHS` | Checkpoint every N epochs | — |
| `USE_SINGULARITY` | Use Singularity container | false |
| `SIF_IMAGE` | Singularity `.sif` image path | `$HOME/deeplearning.sif` |

### `run_knn.sh` — k-NN SLURM Job

Runs k-NN evaluation as a SLURM job (1 GPU, 1 hour).

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `NUM_PROTOTYPES` | Number of prototypes (or use `PRETRAINED_PATH`) | — |
| `DATASET` | Dataset name | dtd |
| `INIT_MODE` | `scratch` or `continued` | scratch |
| `CHECKPOINT_TYPE` | `last` or `best` | last |
| `K` | Number of nearest neighbors | 20 |
| `TEMPERATURE` | Softmax temperature | 0.07 |
| `BATCH_SIZE` | Feature extraction batch size | 256 |
| `KOLEO_WEIGHT` | KoLeo weight (for checkpoint path) | 0.1 |
| `CLS_WEIGHT` | CLS weight (for checkpoint path) | 1.0 |
| `MULTI_CROP` | Multi-crop flag | true |
| `OUTPUT_BASE_DIR` | Base output directory | output |
| `SEEDS` | Evaluation seeds | "0 1 42" |
| `USE_SINGULARITY` | Use Singularity container | false |
| `SIF_IMAGE` | Singularity `.sif` image path | `$HOME/deeplearning.sif` |

### `run_proto_analysis.sh` — Prototype Sweep

Submits multiple SLURM jobs sweeping prototype counts (128, 256, 512, 1024, 2048, 4096).

```bash
bash scripts/slurm/run_proto_analysis.sh
```

### `run_ablation.sh` (SLURM) — Ablation Experiments

See [ABLATIONS.md](ABLATIONS.md) for full ablation documentation.

| Additional Variable | Description | Default |
|--------------------|-------------|---------|
| `ABLATION` | **Required.** `full`, `no_sinkhorn`, `no_cls_loss`, `no_koleo` | — |

---

## Singularity Container

All SLURM scripts support optional Singularity container execution:

```bash
# Default container at $HOME/deeplearning.sif
USE_SINGULARITY=true sbatch scripts/slurm/run_experiment.sh

# Custom container path
USE_SINGULARITY=true SIF_IMAGE=/shared/containers/pytorch.sif \
    sbatch scripts/slurm/run_experiment.sh
```

The scripts automatically bind-mount the repo directory, HuggingFace/PyTorch caches, `/tmp`, and shared memory into the container.

---

## Configuration

Configs are YAML files with sections that can be overridden via CLI arguments.

### Key Hyperparameters

```yaml
model:
  num_prototypes: 4096              # Number of prototype vectors
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

## Logging

| Logger | Flag | Viewer |
|--------|------|--------|
| CSV | `--logger csv` (default) | Open `metrics.csv` directly |
| TensorBoard | `--logger tensorboard` | `tensorboard --logdir output/logs` |
| Weights & Biases | `--logger wandb` | wandb.ai dashboard |

### TensorBoard

```bash
tensorboard --logdir output/logs --bind_all
```

Then SSH tunnel and open `http://localhost:6006`:

```bash
ssh -L 6006:localhost:6006 user@server
```

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
│   ├── aggregate_lowshot.py     # Aggregate lowshot results into tables
│   └── slurm/                   # SLURM job scripts
│       ├── run_experiment.sh    # Single experiment (env var config)
│       ├── run_knn.sh           # k-NN evaluation SLURM job
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
│   └── pancreatic/
├── dinov3/                      # DINOv3 library (backbone, utilities)
├── DATASETS.md                  # Dataset documentation and per-dataset examples
├── ABLATIONS.md                 # Ablation study documentation
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
