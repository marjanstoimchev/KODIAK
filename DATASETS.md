# Datasets

KODIAK supports 7 datasets: 6 from HuggingFace and 1 custom (Pancreatic). The **Pancreatic** dataset is the primary dataset for exploration; the HuggingFace datasets serve as benchmarks.

## Supported Datasets

| Dataset | Config Directory | Type | Classes | Source |
|---------|-----------------|------|---------|--------|
| **Pancreatic** | `configs/pancreatic/` | Custom (folder) | 7 | Local: `/home/marjans/Datasets/pancreatic/SLIDE-3210` |
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

The Pancreatic dataset is a custom medical imaging dataset with 7 tissue classes from histopathology slides. It uses a folder-based structure and is the main dataset for exploring KODIAK.

### Folder Structure

```
/home/marjans/Datasets/pancreatic/SLIDE-3210/
├── class_0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_1/
├── class_2/
├── class_3/
├── class_4/
├── class_5/
└── class_6/
```

### Configuration

The Pancreatic configs use `dataset_type: "pancreatic"` and a local `root_dir`. Key differences from HuggingFace datasets:
- Image size: **256x256** (vs 224 for most HuggingFace datasets)
- Batch size: **64** (smaller due to larger images)
- Splits: 90% train / 10% val / 20% test (from config)

### Complete Copy-Paste Commands

All examples use Singularity container + SLURM. Remove `USE_SINGULARITY=true` to run without a container.

---

#### Pretraining

**From scratch (500 epochs):**

```bash
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh
```

**Continued from DINOv3 (100 epochs):**

```bash
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=continued \
PRETRAIN_EPOCHS=100 \
BATCH_SIZE=64 \
sbatch scripts/slurm/run_experiment.sh
```

---

#### Fine-Tuning (unfrozen backbone, LR 1e-4)

**From scratch checkpoint:**

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

**From continued checkpoint:**

```bash
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
```

---

#### Linear Evaluation (frozen backbone, LR 1e-3)

**From scratch checkpoint:**

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

#### k-NN Evaluation

**From scratch checkpoint:**

```bash
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=scratch \
OUTPUT_BASE_DIR=output_proto_analysis \
sbatch scripts/slurm/run_knn.sh
```

**From continued checkpoint:**

```bash
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=pancreatic \
INIT_MODE=continued \
OUTPUT_BASE_DIR=output_proto_analysis \
sbatch scripts/slurm/run_knn.sh
```

---

#### Full Pipeline (Pretrain + Fine-Tune in One Job)

**From scratch:**

```bash
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

**Continued from DINOv3:**

```bash
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
# Full pipeline — scratch, fine-tuning
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=cifar100 \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh

# Linear eval only — from existing scratch checkpoint
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

# k-NN — scratch
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=cifar100 \
INIT_MODE=scratch \
OUTPUT_BASE_DIR=output_proto_analysis \
sbatch scripts/slurm/run_knn.sh
```

### DTD

```bash
# Full pipeline — continued, fine-tuning
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=dtd \
INIT_MODE=continued \
PRETRAIN_EPOCHS=100 \
CLASSIFY_EPOCHS=50 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh
```

### EuroSAT

```bash
# Full pipeline — scratch, fine-tuning
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=eurosat \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh
```

### Oxford Pets

```bash
# Full pipeline — scratch, fine-tuning
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=oxford_pets \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh
```

### NCT-CRC-HE-100K

```bash
# Full pipeline — scratch, fine-tuning
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=nctcrche100k \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch scripts/slurm/run_experiment.sh
```

### ImageNet-1K

```bash
# Full pipeline — scratch, fine-tuning (needs more GPUs)
USE_SINGULARITY=true \
NUM_PROTOTYPES=128 \
DATASET=imagenet1k \
INIT_MODE=scratch \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=128 \
sbatch --gres=gpu:4 scripts/slurm/run_experiment.sh
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
