#!/bin/bash

#SBATCH --job-name=lc-sweep
#SBATCH --output=logs/slurm-lc-sweep-%j.out
#SBATCH --error=logs/slurm-lc-sweep-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --constraint=h100

# =============================================================================
# KODIAK Local Crops Sweep - SLURM Script
# =============================================================================
# Investigates the impact of local crop count on training speed and accuracy.
# Runs one pretraining + classification pipeline per local crops value.
#
# Environment variable overrides:
#   DATASET             Dataset name (default: dtd)
#   LOCAL_CROPS_VALUES  Space-separated list of local crop counts to test
#                       (default: "0 2 4 8")
#   INIT_MODE           scratch or continued (default: scratch)
#   PRETRAIN_EPOCHS     Pretraining epochs (default: 500 for scratch, 100 continued)
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
#
# Usage:
#   # Default: test 0, 2, 4, 8 local crops on DTD from scratch
#   sbatch scripts/slurm/run_local_crops_sweep.sh
#
#   # Custom local crop values
#   LOCAL_CROPS_VALUES="0 4 8 12" sbatch scripts/slurm/run_local_crops_sweep.sh
#
#   # Different dataset, continued pretraining
#   DATASET=cifar100 INIT_MODE=continued \
#       sbatch scripts/slurm/run_local_crops_sweep.sh
#
#   # Only 0 vs 8 local crops, with Singularity container
#   LOCAL_CROPS_VALUES="0 8" USE_SINGULARITY=true \
#       sbatch scripts/slurm/run_local_crops_sweep.sh
#
#   # Skip pretraining, only classify from existing checkpoints
#   SKIP_PRETRAIN=true LOCAL_CROPS_VALUES="0 2 4 8" \
#       sbatch scripts/slurm/run_local_crops_sweep.sh
# =============================================================================

set -e

# Navigate to repo root
cd "$SLURM_SUBMIT_DIR" || exit 1

# Create logs directory
mkdir -p logs

# =============================================================================
# Configuration
# =============================================================================
DATASET="${DATASET:-dtd}"
LOCAL_CROPS_VALUES="${LOCAL_CROPS_VALUES:-0 2 4 8}"
INIT_MODE="${INIT_MODE:-scratch}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-}"
PRETRAIN_LR="${PRETRAIN_LR:-0.0001}"
CLASSIFY_EPOCHS="${CLASSIFY_EPOCHS:-100}"
CLASSIFY_LR="${CLASSIFY_LR:-0.0001}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_PROTOTYPES="${NUM_PROTOTYPES:-128}"
KOLEO_WEIGHT="${KOLEO_WEIGHT:-0.1}"
CLS_WEIGHT="${CLS_WEIGHT:-1.0}"
SEEDS="${SEEDS:-0 1 42}"
OUTPUT_DIR="${OUTPUT_DIR:-output_local_crops_sweep}"
SKIP_PRETRAIN="${SKIP_PRETRAIN:-false}"
SKIP_CLASSIFY="${SKIP_CLASSIFY:-false}"
COMPILE="${COMPILE:-false}"
LOGGER="${LOGGER:-csv}"

# Singularity container setup
SIF_IMAGE="${SIF_IMAGE:-$HOME/deeplearning.sif}"
USE_SINGULARITY="${USE_SINGULARITY:-false}"

if [[ "$USE_SINGULARITY" == "true" ]] && [[ ! -f "$SIF_IMAGE" ]]; then
    echo "WARNING: Singularity image not found at $SIF_IMAGE, running without container"
    USE_SINGULARITY=false
fi

# Set default pretrain epochs based on init mode
if [[ -z "$PRETRAIN_EPOCHS" ]]; then
    if [[ "$INIT_MODE" == "continued" ]]; then
        PRETRAIN_EPOCHS="100"
    else
        PRETRAIN_EPOCHS="500"
    fi
fi

# Detect GPUs from SLURM allocation
if [[ -n "$SLURM_GPUS_ON_NODE" ]]; then
    NUM_GPUS="$SLURM_GPUS_ON_NODE"
elif [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=1
fi
GPUS=$(seq -s',' 0 $((NUM_GPUS - 1)))

# =============================================================================
# Print configuration
# =============================================================================
echo "============================================================"
echo "KODIAK Local Crops Sweep"
echo "============================================================"
echo "SLURM Job ID:      $SLURM_JOB_ID"
echo "Node:              $SLURM_NODELIST"
echo "GPUs:              $NUM_GPUS ($GPUS)"
echo "Dataset:           $DATASET"
echo "Local crops sweep: $LOCAL_CROPS_VALUES"
echo "Init mode:         $INIT_MODE"
echo "Pretrain epochs:   $PRETRAIN_EPOCHS"
echo "Pretrain LR:       $PRETRAIN_LR"
echo "Classify epochs:   $CLASSIFY_EPOCHS"
echo "Classify LR:       $CLASSIFY_LR"
echo "Batch size:        $BATCH_SIZE"
echo "Num prototypes:    $NUM_PROTOTYPES"
echo "KoLeo weight:      $KOLEO_WEIGHT"
echo "CLS weight:        $CLS_WEIGHT"
echo "Seeds:             $SEEDS"
echo "Output dir:        $OUTPUT_DIR"
echo "Skip pretrain:     $SKIP_PRETRAIN"
echo "Skip classify:     $SKIP_CLASSIFY"
echo "torch.compile:     $COMPILE"
echo "Logger:            $LOGGER"
[[ "$USE_SINGULARITY" == "true" ]] && echo "Container:         $SIF_IMAGE"
echo "Start time:        $(date)"
echo "============================================================"
echo ""

# =============================================================================
# Environment setup
# =============================================================================
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
mkdir -p "$HF_HOME" "$TORCH_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Shared memory for DataLoader workers
SHM_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
mkdir -p "$SHM_DIR"

# Disable PyTorch Lightning's SLURM auto-detection
unset SLURM_NTASKS
unset SLURM_PROCID
unset SLURM_LOCALID
unset SLURM_NODEID
export SLURM_JOB_NAME="bash"

# =============================================================================
# Run sweep over local crop values
# =============================================================================
TOTAL=$(echo $LOCAL_CROPS_VALUES | wc -w)
CURRENT=0

for N_LOCAL in $LOCAL_CROPS_VALUES; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "============================================================"
    echo "[$CURRENT/$TOTAL] Local crops: $N_LOCAL"
    echo "============================================================"

    # Build sweep arguments
    SWEEP_ARGS=(
        --dataset "$DATASET"
        --gpus "$GPUS"
        --init-mode "$INIT_MODE"
        --seeds "$SEEDS"
        --pretrain-epochs "$PRETRAIN_EPOCHS"
        --pretrain-lr "$PRETRAIN_LR"
        --classify-epochs "$CLASSIFY_EPOCHS"
        --classify-lr "$CLASSIFY_LR"
        --batch-size "$BATCH_SIZE"
        --num-prototypes "$NUM_PROTOTYPES"
        --koleo-weight "$KOLEO_WEIGHT"
        --cls-weight "$CLS_WEIGHT"
        --output-dir "$OUTPUT_DIR"
        --logger "$LOGGER"
    )

    # Multi-crop: enable if local crops > 0, disable if 0
    if [[ "$N_LOCAL" -gt 0 ]]; then
        SWEEP_ARGS+=(--multi-crop --local-crops-number "$N_LOCAL")
    else
        SWEEP_ARGS+=(--no-multi-crop)
    fi

    if [[ "$SKIP_PRETRAIN" == "true" ]]; then
        SWEEP_ARGS+=(--skip-pretrain)
    fi

    if [[ "$SKIP_CLASSIFY" == "true" ]]; then
        SWEEP_ARGS+=(--skip-classify)
    fi

    if [[ "$COMPILE" == "true" ]]; then
        SWEEP_ARGS+=(--compile)
    else
        SWEEP_ARGS+=(--no-compile)
    fi

    # Run
    if [[ "$USE_SINGULARITY" == "true" ]]; then
        export SINGULARITYENV_HF_HOME="$HF_HOME"
        export SINGULARITYENV_TORCH_HOME="$TORCH_HOME"
        export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
        export SINGULARITYENV_SLURM_JOB_NAME="bash"

        srun singularity exec --nv \
            --bind "$SLURM_SUBMIT_DIR":"$SLURM_SUBMIT_DIR" \
            --bind /tmp:/tmp \
            --bind "$HF_HOME":"$HF_HOME" \
            --bind "$TORCH_HOME":"$TORCH_HOME" \
            --bind "$SHM_DIR":/dev/shm \
            "$SIF_IMAGE" \
            ./scripts/run_sweep.sh "${SWEEP_ARGS[@]}"
    else
        srun ./scripts/run_sweep.sh "${SWEEP_ARGS[@]}"
    fi

    echo ""
    echo "[$CURRENT/$TOTAL] Local crops=$N_LOCAL complete at $(date)"
done

echo ""
echo "============================================================"
echo "Local Crops Sweep COMPLETE"
echo "============================================================"
echo "Tested: $LOCAL_CROPS_VALUES"
echo "Results in: $OUTPUT_DIR/"
echo "End time: $(date)"
echo "============================================================"

# Clean up shared memory
rm -rf "$SHM_DIR"
