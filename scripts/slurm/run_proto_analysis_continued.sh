#!/bin/bash
# =============================================================================
# Submit multiple KODIAK continued pretraining experiments as separate SLURM jobs.
# Uses DINOv2 pretrained weights as initialization (init-mode=continued).
# Edit the loop variables below to sweep over any parameter.
# =============================================================================

for PROTO in 4096 2048 1024 512 256 128 64; do
  DATASET=cifar100 NUM_PROTOTYPES=$PROTO KOLEO_WEIGHT=0.1 CLS_WEIGHT=1.0 \
  PRETRAIN_EPOCHS=100 PRETRAIN_LR=0.00001 CLASSIFY_EPOCHS=50 \
  CLASSIFY_LR=0.0001 CLASSIFY_MODE=finetune BATCH_SIZE=128 \
  SEEDS="0 1 42" PRETRAIN_SEED=42 INIT_MODE=continued \
  MULTI_CROP=true COMPILE=false OUTPUT_DIR=output_proto_analysis \
  LOGGER=tensorboard \
  USE_SINGULARITY=true SIF_IMAGE="${SIF_IMAGE:-$HOME/deeplearning.sif}" \
  sbatch scripts/slurm/run_experiment.sh
done
