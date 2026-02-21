#!/bin/bash
# =============================================================================
# Submit Local Crops Sweep: One SLURM job per local crop value
# =============================================================================
# Submits separate SLURM jobs for each local crop count, running in parallel.
#
# Environment variable overrides:
#   LOCAL_CROPS_VALUES  Space-separated list (default: "0 2 4 8")
#   DATASET             Dataset name (default: dtd)
#   INIT_MODE           scratch or continued (default: scratch)
#   PRETRAIN_EPOCHS     Pretraining epochs (default: 500 scratch, 100 continued)
#   PRETRAIN_LR         Pretraining learning rate (default: 0.0001)
#   CLASSIFY_EPOCHS     Classification epochs (default: 100)
#   CLASSIFY_LR         Classification learning rate (default: 0.0001)
#   BATCH_SIZE          Batch size (default: 128)
#   NUM_PROTOTYPES      Number of prototypes (default: 128)
#   KOLEO_WEIGHT        KoLeo loss weight (default: 0.1)
#   CLS_WEIGHT          CLS loss weight (default: 1.0)
#   SEEDS               Classification seeds (default: "0 1 42")
#   OUTPUT_DIR          Output directory (default: output_local_crops_sweep)
#   SKIP_PRETRAIN       Set to "true" to skip pretraining
#   SKIP_CLASSIFY       Set to "true" to skip classification
#   COMPILE             Set to "true" to enable torch.compile
#   LOGGER              Logger type: csv, tensorboard, wandb (default: csv)
#   USE_SINGULARITY     Set to "true" to run inside Singularity container
#   SIF_IMAGE           Path to .sif image (default: $HOME/deeplearning.sif)
#   SBATCH_ARGS         Extra sbatch flags (e.g., "--time=2-00:00:00")
#
# Examples:
#   # Default: submit 4 parallel jobs for 0, 2, 4, 8 local crops
#   ./scripts/slurm/submit_local_crops_sweep.sh
#
#   # Custom values with TensorBoard
#   LOCAL_CROPS_VALUES="0 4 8 12" LOGGER=tensorboard \
#       ./scripts/slurm/submit_local_crops_sweep.sh
#
#   # With Singularity, custom LR
#   PRETRAIN_LR=1e-4 CLASSIFY_LR=1e-4 \
#       USE_SINGULARITY=true SIF_IMAGE=/path/to/deeplearning.sif \
#       ./scripts/slurm/submit_local_crops_sweep.sh
# =============================================================================

set -e

LOCAL_CROPS_VALUES="${LOCAL_CROPS_VALUES:-0 2 4 8}"
SBATCH_ARGS="${SBATCH_ARGS:-}"

echo "============================================================"
echo "Submitting Local Crops Sweep"
echo "============================================================"
echo "Local crops values: $LOCAL_CROPS_VALUES"
[ -n "${DATASET}" ] && echo "Dataset:           $DATASET"
[ -n "${PRETRAIN_EPOCHS}" ] && echo "Pretrain epochs:   $PRETRAIN_EPOCHS"
[ -n "${PRETRAIN_LR}" ] && echo "Pretrain LR:       $PRETRAIN_LR"
[ -n "${CLASSIFY_EPOCHS}" ] && echo "Classify epochs:   $CLASSIFY_EPOCHS"
[ -n "${CLASSIFY_LR}" ] && echo "Classify LR:       $CLASSIFY_LR"
[ -n "${BATCH_SIZE}" ] && echo "Batch size:        $BATCH_SIZE"
[ -n "${NUM_PROTOTYPES}" ] && echo "Num prototypes:    $NUM_PROTOTYPES"
[ -n "${LOGGER}" ] && echo "Logger:            $LOGGER"
[ -n "$SBATCH_ARGS" ] && echo "SBATCH args:       $SBATCH_ARGS"
echo "============================================================"

# Build env var prefix to forward to each job
ENV_VARS=""
[ -n "${DATASET}" ] && ENV_VARS="$ENV_VARS DATASET=$DATASET"
[ -n "${INIT_MODE}" ] && ENV_VARS="$ENV_VARS INIT_MODE=$INIT_MODE"
[ -n "${PRETRAIN_EPOCHS}" ] && ENV_VARS="$ENV_VARS PRETRAIN_EPOCHS=$PRETRAIN_EPOCHS"
[ -n "${PRETRAIN_LR}" ] && ENV_VARS="$ENV_VARS PRETRAIN_LR=$PRETRAIN_LR"
[ -n "${CLASSIFY_EPOCHS}" ] && ENV_VARS="$ENV_VARS CLASSIFY_EPOCHS=$CLASSIFY_EPOCHS"
[ -n "${CLASSIFY_LR}" ] && ENV_VARS="$ENV_VARS CLASSIFY_LR=$CLASSIFY_LR"
[ -n "${BATCH_SIZE}" ] && ENV_VARS="$ENV_VARS BATCH_SIZE=$BATCH_SIZE"
[ -n "${NUM_PROTOTYPES}" ] && ENV_VARS="$ENV_VARS NUM_PROTOTYPES=$NUM_PROTOTYPES"
[ -n "${KOLEO_WEIGHT}" ] && ENV_VARS="$ENV_VARS KOLEO_WEIGHT=$KOLEO_WEIGHT"
[ -n "${CLS_WEIGHT}" ] && ENV_VARS="$ENV_VARS CLS_WEIGHT=$CLS_WEIGHT"
[ -n "${SEEDS}" ] && ENV_VARS="$ENV_VARS SEEDS=\"$SEEDS\""
[ -n "${OUTPUT_DIR}" ] && ENV_VARS="$ENV_VARS OUTPUT_DIR=$OUTPUT_DIR"
[ "${SKIP_PRETRAIN:-false}" = "true" ] && ENV_VARS="$ENV_VARS SKIP_PRETRAIN=true"
[ "${SKIP_CLASSIFY:-false}" = "true" ] && ENV_VARS="$ENV_VARS SKIP_CLASSIFY=true"
[ "${COMPILE:-false}" = "true" ] && ENV_VARS="$ENV_VARS COMPILE=true"
[ -n "${LOGGER}" ] && ENV_VARS="$ENV_VARS LOGGER=$LOGGER"
[ "${USE_SINGULARITY:-false}" = "true" ] && ENV_VARS="$ENV_VARS USE_SINGULARITY=true"
[ -n "${SIF_IMAGE}" ] && ENV_VARS="$ENV_VARS SIF_IMAGE=$SIF_IMAGE"

# Submit one job per local crops value
for N_LOCAL in $LOCAL_CROPS_VALUES; do
    CMD="$ENV_VARS LOCAL_CROPS_VALUES=$N_LOCAL sbatch $SBATCH_ARGS scripts/slurm/run_local_crops_sweep.sh"
    echo ""
    echo ">>> $CMD"
    eval $CMD
done

echo ""
echo "============================================================"
echo "All jobs submitted. Check status with: squeue -u \$USER"
echo "============================================================"
