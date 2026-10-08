# Datasets

KODIAK supports 7 datasets: 6 from HuggingFace and 1 custom (Pancreatic). The **Pancreatic** dataset is the primary dataset for exploration; the HuggingFace datasets serve as benchmarks.

## Supported Datasets

| Dataset | Config Directory | Type | Classes | Source |
|---------|-----------------|------|---------|--------|
| **Pancreatic** | `configs/pancreatic/` | Custom (folder) | 12 | Local folder: `$KODIAK_DATA_DIR/pancreatic` |
| CIFAR-100 | `configs/cifar100/` | HuggingFace | 100 | `uoft-cs/cifar100` |
| DTD | `configs/DTD/` | HuggingFace | 47 | `cansa/Describable-Textures-Dataset-DTD` |
| EuroSAT | `configs/eurosat/` | HuggingFace | 10 | `blanchon/EuroSAT_RGB` |
| Oxford Pets | `configs/oxford_pets/` | HuggingFace | 37 | `timm/oxford-iiit-pet` |
| NCT-CRC-HE-100K | `configs/NCTCRCHE100K/` | HuggingFace | 9 | `DykeF/NCTCRCHE100K` |
| ImageNet-1K | `configs/imagenet1k/` | HuggingFace | 1000 | `ILSVRC/imagenet-1k` |

Each dataset has 3 config files:
- `pretrain.yaml` — from-scratch pretraining
- `pretrain_continued.yaml` — continued pretraining from DINOv3 weights
- `classify.yaml` — classification (fine-tuning / linear evaluation)

---

## Pancreatic Dataset (Primary)

The Pancreatic dataset is a custom medical imaging dataset of histopathology patches (12 tissue classes in the
shipped config; set `data.num_classes` in `configs/pancreatic/classify.yaml` to match your data). It uses a
folder-based structure and is the main dataset for exploring KODIAK. The dataset itself is not distributed with
this repository.

### Folder Structure

The configs reference `${KODIAK_DATA_DIR}/pancreatic`; set the variable (or pass `--root_dir` to any script).
Both a flat layout and a nested (one sub-folder per slide) layout are detected automatically:

```
$KODIAK_DATA_DIR/pancreatic/            # flat                $KODIAK_DATA_DIR/pancreatic/   # nested
├── class_0/                                                  ├── SLIDE-1/
│   ├── image1.png                                            │   ├── class_0/
│   └── ...                                                   │   └── class_1/
├── class_1/                                                  └── SLIDE-2/
└── ...                                                           ├── class_0/
                                                                  └── class_1/
```

Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`. Class names are the sub-folder names.

### Configuration

The Pancreatic configs use `dataset_type: "pancreatic"` and a local `root_dir`. Key differences from HuggingFace datasets:
- Image size: **256x256** (vs 224 for most HuggingFace datasets)
- Batch size: **64** (smaller due to larger images)
- Splits: 90% train / 10% val / 20% test (from config)

### Complete Copy-Paste Commands

Each command is shown in both SLURM and non-SLURM variants.

---

#### Pretraining

**From scratch (500 epochs):**

```bash
# SLURM
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
SKIP_CLASSIFY=true \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh

# Non-SLURM
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode scratch --pretrain-epochs 500 \
    --skip-classify --batch-size 64 --num-prototypes 128
```

**Continued from DINOv3 (100 epochs):**

```bash
# SLURM
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=continued \
PRETRAIN_EPOCHS=100 \
SKIP_CLASSIFY=true \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh

# Non-SLURM
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode continued --pretrain-epochs 100 \
    --skip-classify --batch-size 64 --num-prototypes 128
```

---

#### Fine-Tuning (unfrozen backbone, LR 1e-4)

**From scratch checkpoint:**

```bash
# SLURM
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

# Non-SLURM
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode scratch --skip-pretrain \
    --classify-mode finetune --classify-lr 1e-4 --classify-epochs 100 \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"
```

**From continued checkpoint:**

```bash
# SLURM
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

# Non-SLURM
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode continued --skip-pretrain \
    --classify-mode finetune --classify-lr 1e-4 --classify-epochs 50 \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"
```

---

#### Linear Evaluation (frozen backbone, LR 1e-3)

**From scratch checkpoint:**

```bash
# SLURM
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

# Non-SLURM
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode scratch --skip-pretrain \
    --classify-mode lineareval --classify-lr 1e-3 --classify-epochs 50 \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"
```

**From continued checkpoint:**

```bash
# SLURM
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

# Non-SLURM
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode continued --skip-pretrain \
    --classify-mode lineareval --classify-lr 1e-3 --classify-epochs 50 \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"
```

---

#### k-NN Evaluation

**From scratch checkpoint:**

```bash
# SLURM
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=scratch \
OUTPUT_BASE_DIR=output_proto_analysis \
sbatch scripts/slurm/run_knn.sh

# Non-SLURM
./scripts/run_knn.sh --dataset pancreatic --gpus 0 \
    --init-mode scratch --num-prototypes 128 \
    --output-base-dir output_proto_analysis
```

**From continued checkpoint:**

```bash
# SLURM
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=continued \
OUTPUT_BASE_DIR=output_proto_analysis \
sbatch scripts/slurm/run_knn.sh

# Non-SLURM
./scripts/run_knn.sh --dataset pancreatic --gpus 0 \
    --init-mode continued --num-prototypes 128 \
    --output-base-dir output_proto_analysis
```

---

#### Full Pipeline (Pretrain + Fine-Tune in One Job)

**From scratch:**

```bash
# SLURM
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

# Non-SLURM
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode scratch --pretrain-epochs 500 \
    --classify-epochs 100 --classify-lr 1e-4 --classify-mode finetune \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"
```

**Continued from DINOv3:**

```bash
# SLURM
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

# Non-SLURM
./scripts/run_sweep.sh --dataset pancreatic --gpus 0,1 \
    --init-mode continued --pretrain-epochs 100 \
    --classify-epochs 50 --classify-lr 1e-4 --classify-mode finetune \
    --batch-size 64 --num-prototypes 128 --seeds "0 1 42"
```

---

#### Output Locations

```
output_proto_analysis/
├── checkpoints/pretraining/pancreatic/
│   ├── kodiak_pancreatic_proto128_koleo0.1_cls1.0_mc/last.ckpt           # scratch
│   └── kodiak_pancreatic_continued_proto128_koleo0.1_cls1.0_mc/last.ckpt # continued
├── logs/classification/pancreatic/
│   └── proto128_koleo0.1_cls1.0_mc/
│       ├── finetune_seed_0/test_results.json
│       ├── finetune_seed_1/test_results.json
│       ├── finetune_seed_42/test_results.json
│       ├── lineareval_seed_0/test_results.json
│       ├── lineareval_seed_1/test_results.json
│       └── lineareval_seed_42/test_results.json
└── knn/pancreatic/
    └── proto128_koleo0.1_cls1.0_mc/knn_results.json
```

---

## HuggingFace Datasets

HuggingFace datasets are downloaded automatically on first run and cached in `$HF_HOME` (default: `$HOME/.cache/huggingface`).

### CIFAR-100

```bash
# SLURM — full pipeline, scratch, fine-tuning
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=cifar100 \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh

# Non-SLURM — full pipeline, scratch, fine-tuning
./scripts/run_sweep.sh --dataset cifar100 --gpus 0,1 \
    --init-mode scratch --pretrain-epochs 500 \
    --classify-epochs 100 --classify-lr 1e-4 --classify-mode finetune \
    --batch-size 128 --num-prototypes 128 --seeds "0 1 42"

# SLURM — linear eval only, from existing scratch checkpoint
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=cifar100 \
INIT_MODE=scratch \
SKIP_PRETRAIN=true \
CLASSIFY_MODE=lineareval \
CLASSIFY_LR=1e-3 \
CLASSIFY_EPOCHS=50 \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh

# Non-SLURM — linear eval only, from existing scratch checkpoint
./scripts/run_sweep.sh --dataset cifar100 --gpus 0,1 \
    --init-mode scratch --skip-pretrain \
    --classify-mode lineareval --classify-lr 1e-3 --classify-epochs 50 \
    --batch-size 128 --num-prototypes 128 --seeds "0 1 42"

# SLURM — k-NN, scratch
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=cifar100 \
INIT_MODE=scratch \
OUTPUT_BASE_DIR=output_proto_analysis \
sbatch scripts/slurm/run_knn.sh

# Non-SLURM — k-NN, scratch
./scripts/run_knn.sh --dataset cifar100 --gpus 0 \
    --init-mode scratch --num-prototypes 128 \
    --output-base-dir output_proto_analysis
```

### DTD

```bash
# SLURM — full pipeline, continued, fine-tuning
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=dtd \
INIT_MODE=continued \
PRETRAIN_EPOCHS=100 \
CLASSIFY_EPOCHS=50 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh

# Non-SLURM — full pipeline, continued, fine-tuning
./scripts/run_sweep.sh --dataset dtd --gpus 0,1 \
    --init-mode continued --pretrain-epochs 100 \
    --classify-epochs 50 --classify-lr 1e-4 --classify-mode finetune \
    --batch-size 128 --num-prototypes 128 --seeds "0 1 42"
```

### EuroSAT

```bash
# SLURM — full pipeline, scratch, fine-tuning
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=eurosat \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh

# Non-SLURM — full pipeline, scratch, fine-tuning
./scripts/run_sweep.sh --dataset eurosat --gpus 0,1 \
    --init-mode scratch --pretrain-epochs 500 \
    --classify-epochs 100 --classify-lr 1e-4 --classify-mode finetune \
    --batch-size 128 --num-prototypes 128 --seeds "0 1 42"
```

### Oxford Pets

```bash
# SLURM — full pipeline, scratch, fine-tuning
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=oxford_pets \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh

# Non-SLURM — full pipeline, scratch, fine-tuning
./scripts/run_sweep.sh --dataset oxford_pets --gpus 0,1 \
    --init-mode scratch --pretrain-epochs 500 \
    --classify-epochs 100 --classify-lr 1e-4 --classify-mode finetune \
    --batch-size 128 --num-prototypes 128 --seeds "0 1 42"
```

### NCT-CRC-HE-100K

```bash
# SLURM — full pipeline, scratch, fine-tuning
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=nctcrche100k \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh

# Non-SLURM — full pipeline, scratch, fine-tuning
./scripts/run_sweep.sh --dataset nctcrche100k --gpus 0,1 \
    --init-mode scratch --pretrain-epochs 500 \
    --classify-epochs 100 --classify-lr 1e-4 --classify-mode finetune \
    --batch-size 128 --num-prototypes 128 --seeds "0 1 42"
```

### ImageNet-1K

```bash
# SLURM — full pipeline, scratch, fine-tuning (needs more GPUs)
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=imagenet1k \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch --gres=gpu:4 scripts/slurm/run_experiment.sh

# Non-SLURM — full pipeline, scratch, fine-tuning (needs more GPUs)
./scripts/run_sweep.sh --dataset imagenet1k --gpus 0,1,2,3 \
    --init-mode scratch --pretrain-epochs 500 \
    --classify-epochs 100 --classify-lr 1e-4 --classify-mode finetune \
    --batch-size 128 --num-prototypes 128 --seeds "0 1 42"
```

---

## Adding a Custom Dataset

1. Create a folder structure with one subfolder per class:
   ```
   /path/to/my_dataset/
   ├── class_a/
   │   ├── img001.jpg
   │   └── ...
   ├── class_b/
   └── class_c/
   ```

2. Create config files in `configs/my_dataset/`:
   - Copy from `configs/pancreatic/` as a template
   - Update `root_dir`, `num_classes`, `image_size`, `batch_size`

3. Register the dataset name in `src/data/utils/registry.py` (if using the custom dataset loader)

4. Run with `--dataset my_dataset` or `DATASET=my_dataset`
