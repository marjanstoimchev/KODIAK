# Experiment recipes

Copy-paste commands for every experiment in the paper, on SLURM (with an optional Singularity container) and on a plain multi-GPU machine. See the [README](README.md) for installation and the short version, [DATASETS.md](DATASETS.md) for per-dataset commands and [ABLATIONS.md](ABLATIONS.md) for ablations.

Set `KODIAK_DATA_DIR` (or pass `--root_dir`) before running anything on the Pancreatic dataset.

---

## SLURM usage

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

### 5. Few-Shot (Low-Shot) Evaluation

Evaluates SSL representations with limited labelled data (k examples per class). Runs 3 label seeds x 1 training seed = 3 runs per setting by default. Batch size is dynamically adjusted for very low-shot settings (k=1,2) to ensure enough gradient updates.

#### Fine-Tuning + Linear Eval on CIFAR-100 (continued)

**Fine-tuning (LR 1e-4, 128 prototypes):**

```bash
USE_SINGULARITY=true \
INIT_MODE=continued \
DATASETS=cifar100 \
SHOTS="1 2 4 8 16" \
MODE=finetune \
LEARNING_RATE=1e-4 \
MAX_EPOCHS=50 \
BATCH_SIZE=128 \
CKPT_BASE_DIR=output_proto_analysis \
PROTOTYPES=128 \
sbatch scripts/slurm/run_lowshot.sh
```

**Linear evaluation (LR 1e-3):**

```bash
USE_SINGULARITY=true \
INIT_MODE=continued \
DATASETS=cifar100 \
SHOTS="1 2 4 8 16" \
MODE=lineareval \
LEARNING_RATE=1e-3 \
MAX_EPOCHS=50 \
BATCH_SIZE=128 \
CKPT_BASE_DIR=output_proto_analysis \
PROTOTYPES=128 \
sbatch scripts/slurm/run_lowshot.sh
```

**Custom prototypes (e.g. 256):**

```bash
USE_SINGULARITY=true \
INIT_MODE=continued \
DATASETS=cifar100 \
SHOTS="1 2 4 8 16" \
MODE=finetune \
LEARNING_RATE=1e-4 \
MAX_EPOCHS=50 \
BATCH_SIZE=128 \
CKPT_BASE_DIR=output_proto_analysis \
PROTOTYPES=256 \
sbatch --constraint h100 scripts/slurm/run_lowshot.sh
```

#### Pancreatic (continued)

```bash
# Fine-tuning
USE_SINGULARITY=true \
INIT_MODE=continued \
DATASETS=pancreatic \
SHOTS="1 2 4 8 16" \
MODE=finetune \
LEARNING_RATE=1e-4 \
MAX_EPOCHS=50 \
BATCH_SIZE=64 \
CKPT_BASE_DIR=output_proto_analysis \
PROTOTYPES=128 \
sbatch scripts/slurm/run_lowshot.sh

# Linear eval
USE_SINGULARITY=true \
INIT_MODE=continued \
DATASETS=pancreatic \
SHOTS="1 2 4 8 16" \
MODE=lineareval \
LEARNING_RATE=1e-3 \
MAX_EPOCHS=50 \
BATCH_SIZE=64 \
CKPT_BASE_DIR=output_proto_analysis \
PROTOTYPES=128 \
sbatch scripts/slurm/run_lowshot.sh
```

#### From Scratch Checkpoints

```bash
# Fine-tuning on multiple datasets
USE_SINGULARITY=true \
INIT_MODE=scratch \
DATASETS="cifar100 dtd" \
SHOTS="1 2 4 8 16" \
MODE=finetune \
LEARNING_RATE=1e-4 \
MAX_EPOCHS=50 \
BATCH_SIZE=128 \
CKPT_BASE_DIR=output_proto_analysis \
PROTOTYPES=128 \
sbatch scripts/slurm/run_lowshot.sh
```

#### Few-Shot SLURM Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INIT_MODE` | continued | `scratch` or `continued` |
| `METHODS` | kodiak | Methods to evaluate (space-separated) |
| `DATASETS` | "cifar100 dtd" | Datasets to evaluate (space-separated) |
| `SHOTS` | "1 2 4 8 16" | K-shot values (space-separated) |
| `LABEL_SEEDS` | "0 1 42" | Label subset seeds |
| `TRAIN_SEEDS` | "42" | Training seeds |
| `MODE` | both | `finetune`, `lineareval`, or both |
| `MAX_EPOCHS` | 100 | Max training epochs |
| `BATCH_SIZE` | from config | Batch size |
| `LEARNING_RATE` | from config | Learning rate |
| `OUTPUT_DIR` | auto | Override output directory |
| `CKPT_BASE_DIR` | output_proto_analysis | Base directory for kodiak checkpoints |
| `PROTOTYPES` | 128 | Number of prototypes (64, 128, 256, 512, 1024, 2048, 4096) |
| `USE_SINGULARITY` | false | Use Singularity container |
| `SIF_IMAGE` | `$HOME/deeplearning.sif` | Container image path |

---

### 6. Ablation Studies

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

### 5. Few-Shot (Low-Shot) Evaluation

#### CIFAR-100 (continued, 128 prototypes)

```bash
# Fine-tuning (LR 1e-4, 50 epochs)
./scripts/run_lowshot.sh --gpus 0 --methods kodiak --datasets cifar100 \
    --shots "1 2 4 8 16" --mode finetune --init-mode continued \
    --learning-rate 1e-4 --max-epochs 50 --batch-size 128 \
    --ckpt-base-dir output_proto_analysis --prototypes 128

# Linear eval (LR 1e-3, 50 epochs)
./scripts/run_lowshot.sh --gpus 0 --methods kodiak --datasets cifar100 \
    --shots "1 2 4 8 16" --mode lineareval --init-mode continued \
    --learning-rate 1e-3 --max-epochs 50 --batch-size 128 \
    --ckpt-base-dir output_proto_analysis --prototypes 128

# Custom prototypes (e.g. 256)
./scripts/run_lowshot.sh --gpus 0 --methods kodiak --datasets cifar100 \
    --shots "1 2 4 8 16" --mode finetune --init-mode continued \
    --learning-rate 1e-4 --max-epochs 50 --batch-size 128 \
    --ckpt-base-dir output_proto_analysis --prototypes 256
```

#### Pancreatic (continued)

```bash
# Fine-tuning
./scripts/run_lowshot.sh --gpus 0 --methods kodiak --datasets pancreatic \
    --shots "1 2 4 8 16" --mode finetune --init-mode continued \
    --learning-rate 1e-4 --max-epochs 50 --batch-size 64 \
    --ckpt-base-dir output_proto_analysis --prototypes 128

# Linear eval
./scripts/run_lowshot.sh --gpus 0 --methods kodiak --datasets pancreatic \
    --shots "1 2 4 8 16" --mode lineareval --init-mode continued \
    --learning-rate 1e-3 --max-epochs 50 --batch-size 64 \
    --ckpt-base-dir output_proto_analysis --prototypes 128
```

#### From Scratch Checkpoints

```bash
# Fine-tuning on multiple datasets
./scripts/run_lowshot.sh --gpus 0 --methods kodiak --datasets "cifar100 dtd" \
    --shots "1 2 4 8 16" --mode finetune --init-mode scratch \
    --learning-rate 1e-4 --max-epochs 50 --batch-size 128 \
    --ckpt-base-dir output_proto_analysis --prototypes 128
```

---

### 6. Ablation Studies

```bash
# Run all ablations on Pancreatic
./scripts/run_ablation.sh --dataset pancreatic --gpus 0,1 \
    --ablations "full no_sinkhorn no_cls_loss no_koleo" \
    --pretrain-epochs 500 --classify-epochs 100 \
    --batch-size 64 --num-prototypes 1024 --output-dir ablations
```

---

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

### `run_lowshot.sh` — Few-Shot SLURM Job

Runs k-shot evaluation sweep as a SLURM job (1 GPU, 3 days).

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `INIT_MODE` | `scratch` or `continued` | continued |
| `METHODS` | Methods to evaluate (space-separated) | kodiak |
| `DATASETS` | Datasets to evaluate (space-separated) | "cifar100 dtd" |
| `SHOTS` | K-shot values (space-separated) | "1 2 4 8 16" |
| `LABEL_SEEDS` | Label subset seeds | "0 1 42" |
| `TRAIN_SEEDS` | Training seeds | "42" |
| `MODE` | `finetune`, `lineareval`, or both | both |
| `MAX_EPOCHS` | Max training epochs | 100 |
| `BATCH_SIZE` | Batch size | from config |
| `LEARNING_RATE` | Learning rate | from config |
| `OUTPUT_DIR` | Override output directory | auto |
| `CKPT_BASE_DIR` | Base directory for kodiak checkpoints | output_proto_analysis |
| `PROTOTYPES` | Number of prototypes (64, 128, 256, 512, 1024, 2048, 4096) | 128 |
| `USE_SINGULARITY` | Use Singularity container | false |
| `SIF_IMAGE` | Singularity `.sif` image path | `$HOME/deeplearning.sif` |

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

The scripts automatically bind-mount the repo directory, HuggingFace/PyTorch caches, `/tmp`, shared memory and
`$KODIAK_DATA_DIR` (when set) into the container. Building the image (`Singularity.def`) expects a local
`flash_attn` wheel next to the definition file; see the `%files` section.

---
