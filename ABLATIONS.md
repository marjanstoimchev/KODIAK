# Ablation Studies

KODIAK's ablation framework tests the contribution of each key component by selectively disabling it. All ablations share the same pretraining + classification pipeline — only the loss configuration changes.

## Ablation Options

| Ablation | Description | What Changes |
|----------|-------------|--------------|
| `full` | All components enabled (baseline) | Nothing disabled |
| `no_sinkhorn` | Softmax instead of Sinkhorn-Knopp | Prototype normalization changes |
| `no_cls_loss` | Disable multi-crop CLS distillation | `cls_weight = 0` |
| `no_koleo` | Disable KoLeo regularization | `koleo_weight = 0` |

## What Each Component Does

### Sinkhorn-Knopp Normalization
Balances prototype assignments across the batch to prevent collapse (all samples mapping to the same prototype). The alternative (softmax) has no such constraint. Disabling Sinkhorn tests whether the model can avoid collapse without explicit balancing.

### Multi-Crop CLS Distillation Loss
Forces CLS tokens to match across different crops of the same image in prototype space. Teacher produces soft targets from global crops; student matches them from both global and local crops. Disabling this removes cross-crop consistency learning.

### KoLeo Regularization
Encourages uniform spreading of features in the embedding space by maximizing the distance to the nearest neighbor. Prevents representation collapse where features cluster in a small region. Disabling this tests whether the model naturally spreads features.

---

## Running Ablations

### SLURM (with Singularity)

**Single ablation:**

```bash
ABLATION=no_koleo \
DATASET=pancreatic \
PRETRAIN_EPOCHS=500 \
CLASSIFY_EPOCHS=100 \
CLASSIFY_LR=1e-4 \
BATCH_SIZE=64 \
NUM_PROTOTYPES=1024 \
USE_SINGULARITY=true \
LOGGER=tensorboard \
sbatch scripts/slurm/run_ablation.sh
```

**All ablations on Pancreatic:**

```bash
for ABL in full no_sinkhorn no_cls_loss no_koleo; do
    ABLATION=$ABL \
    DATASET=pancreatic \
    PRETRAIN_EPOCHS=500 \
    CLASSIFY_EPOCHS=100 \
    CLASSIFY_LR=1e-4 \
    BATCH_SIZE=64 \
    NUM_PROTOTYPES=1024 \
    USE_SINGULARITY=true \
    LOGGER=tensorboard \
    sbatch scripts/slurm/run_ablation.sh
done
```

**All ablations on CIFAR-100:**

```bash
for ABL in full no_sinkhorn no_cls_loss no_koleo; do
    ABLATION=$ABL \
    DATASET=cifar100 \
    PRETRAIN_EPOCHS=500 \
    CLASSIFY_EPOCHS=100 \
    CLASSIFY_LR=1e-4 \
    BATCH_SIZE=128 \
    NUM_PROTOTYPES=1024 \
    USE_SINGULARITY=true \
    LOGGER=tensorboard \
    sbatch scripts/slurm/run_ablation.sh
done
```

**All ablations on DTD:**

```bash
for ABL in full no_sinkhorn no_cls_loss no_koleo; do
    ABLATION=$ABL \
    DATASET=dtd \
    PRETRAIN_EPOCHS=500 \
    CLASSIFY_EPOCHS=100 \
    CLASSIFY_LR=1e-4 \
    BATCH_SIZE=128 \
    NUM_PROTOTYPES=1024 \
    USE_SINGULARITY=true \
    LOGGER=tensorboard \
    sbatch scripts/slurm/run_ablation.sh
done
```

### Local (no SLURM)

```bash
./scripts/run_ablation.sh \
    --dataset pancreatic \
    --gpus 0,1 \
    --ablations "full no_sinkhorn no_cls_loss no_koleo" \
    --pretrain-epochs 500 \
    --classify-epochs 100 \
    --batch-size 64 \
    --num-prototypes 1024 \
    --output-dir ablations
```

---

## SLURM Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ABLATION` | — | **Required.** One of: `full`, `no_sinkhorn`, `no_cls_loss`, `no_koleo` |
| `DATASET` | dtd | Dataset name |
| `NUM_PROTOTYPES` | 1024 | Number of prototypes |
| `PRETRAIN_EPOCHS` | 500 | Pretraining epochs |
| `CLASSIFY_EPOCHS` | 100 | Classification epochs |
| `CLASSIFY_LR` | 1e-4 | Classification learning rate |
| `BATCH_SIZE` | 128 | Batch size |
| `LOGGER` | csv | `csv`, `tensorboard`, `wandb` |
| `USE_SINGULARITY` | false | Use Singularity container |
| `SIF_IMAGE` | `$HOME/deeplearning.sif` | Container image path |

---

## Output Structure

Ablation results are saved under the `ablations/` directory (or `--output-dir`):

```
ablations/
├── checkpoints/pretraining/{dataset}/
│   ├── kodiak_{dataset}_full_proto1024_koleo0.1_cls1.0_mc/last.ckpt
│   ├── kodiak_{dataset}_no_sinkhorn_proto1024_koleo0.1_cls1.0_mc/last.ckpt
│   ├── kodiak_{dataset}_no_cls_loss_proto1024_koleo0.0_cls0.0_mc/last.ckpt
│   └── kodiak_{dataset}_no_koleo_proto1024_koleo0.0_cls1.0_mc/last.ckpt
└── logs/
    ├── pretraining/{dataset}/
    │   └── {ablation_experiment_name}/
    │       ├── training_summary.json
    │       └── version_0/    # TensorBoard events or CSV metrics
    └── classification/{dataset}/
        └── {pretrain_folder}/
            ├── finetune_seed_0/test_results.json
            ├── finetune_seed_1/test_results.json
            └── finetune_seed_42/test_results.json
```

---

## Comparing Results

After running all ablations, compare `test_results.json` across ablation folders:

```bash
# Quick comparison of test accuracy across ablations
for abl in full no_sinkhorn no_cls_loss no_koleo; do
    echo "=== $abl ==="
    cat ablations/logs/classification/pancreatic/*${abl}*/finetune_seed_0/test_results.json 2>/dev/null | python -m json.tool | grep test_acc
done
```

---

## Prototype Count Sweep (Separate from Ablations)

The prototype count sweep is a separate analysis that tests different numbers of prototypes (64, 128, 256, 512, 1024, 2048, 4096). It uses the standard pipeline (`run_experiment.sh`), not the ablation script.

```bash
# Submit prototype sweep for Pancreatic
for NPROTO in 64 128 256 512 1024 2048 4096; do
    USE_SINGULARITY=true \
    NUM_PROTOTYPES=$NPROTO \
    DATASET=pancreatic \
    INIT_MODE=scratch \
    PRETRAIN_EPOCHS=500 \
    CLASSIFY_EPOCHS=100 \
    CLASSIFY_LR=1e-4 \
    BATCH_SIZE=64 \
    OUTPUT_DIR=output_proto_analysis \
    sbatch scripts/slurm/run_experiment.sh
done
```

See also: `scripts/slurm/run_proto_analysis.sh` for a pre-configured sweep script.
