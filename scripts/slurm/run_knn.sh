#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --output=logs/slurm-knn-%j.out
#SBATCH --error=logs/slurm-knn-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu

# =============================================================================
# KODIAK k-NN Evaluation - SLURM Script
# =============================================================================
# Runs k-NN evaluation on pretrained SSL checkpoints via SLURM.
# All parameters are configurable via environment variables.
#
# Usage:
#   # Scratch mode
#   NUM_PROTOTYPES=128 DATASET=cifar100 \
#       sbatch scripts/slurm/run_knn.sh
#
#   # Continued mode
#   NUM_PROTOTYPES=128 DATASET=cifar100 INIT_MODE=continued \
#       sbatch scripts/slurm/run_knn.sh
#
#   # With Singularity container
#   USE_SINGULARITY=true NUM_PROTOTYPES=128 DATASET=cifar100 \
#       sbatch scripts/slurm/run_knn.sh
#
#   # Custom checkpoint path
#   DATASET=cifar100 PRETRAINED_PATH=/path/to/checkpoint.ckpt \
#       sbatch scripts/slurm/run_knn.sh
# =============================================================================

# Navigate to repo root (where sbatch was submitted from)
cd "$SLURM_SUBMIT_DIR" || exit 1

# Create logs directory if needed
mkdir -p logs

# =============================================================================
# Singularity container setup (optional)
# =============================================================================
SIF_IMAGE="${SIF_IMAGE:-$HOME/deeplearning.sif}"
USE_SINGULARITY="${USE_SINGULARITY:-false}"

if [[ "$USE_SINGULARITY" == "true" ]] && [[ ! -f "$SIF_IMAGE" ]]; then
    echo "WARNING: Singularity image not found at $SIF_IMAGE, running without container"
    USE_SINGULARITY=false
fi

# =============================================================================
# Print job info
# =============================================================================
echo "============================================================"
echo "KODIAK k-NN Evaluation (SLURM)"
echo "============================================================"
echo "SLURM Job ID:    $SLURM_JOB_ID"
echo "Node:            $SLURM_NODELIST"
echo "GPUs:            $CUDA_VISIBLE_DEVICES"
if [[ "$USE_SINGULARITY" == "true" ]]; then
    echo "Container:       $SIF_IMAGE"
fi
echo "Working dir:     $(pwd)"
echo "Start time:      $(date)"
echo "============================================================"
echo ""

# =============================================================================
# Configuration (override via environment variables)
# =============================================================================
DATASET="${DATASET:-dtd}"
INIT_MODE="${INIT_MODE:-scratch}"
NUM_PROTOTYPES="${NUM_PROTOTYPES:-}"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"
CHECKPOINT_TYPE="${CHECKPOINT_TYPE:-last}"
K="${K:-20}"
TEMPERATURE="${TEMPERATURE:-0.07}"
BATCH_SIZE="${BATCH_SIZE:-256}"
KOLEO_WEIGHT="${KOLEO_WEIGHT:-0.1}"
CLS_WEIGHT="${CLS_WEIGHT:-1.0}"
MULTI_CROP="${MULTI_CROP:-true}"
LOCAL_CROPS_NUMBER="${LOCAL_CROPS_NUMBER:-8}"
CONCAT_CLS_PATCH="${CONCAT_CLS_PATCH:-false}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-output}"
SEEDS="${SEEDS:-0 1 42}"

# Require either NUM_PROTOTYPES or PRETRAINED_PATH
if [[ -z "$NUM_PROTOTYPES" && -z "$PRETRAINED_PATH" ]]; then
    echo "ERROR: NUM_PROTOTYPES or PRETRAINED_PATH environment variable is required."
    echo "Usage: NUM_PROTOTYPES=128 sbatch scripts/slurm/run_knn.sh"
    exit 1
fi

# Detect GPU index (k-NN uses single GPU)
GPUS="0"

echo "Dataset:          $DATASET"
echo "Init mode:        $INIT_MODE"
echo "GPU:              $GPUS"
echo "Num prototypes:   $NUM_PROTOTYPES"
echo "Checkpoint type:  $CHECKPOINT_TYPE"
echo "k:                $K"
echo "Temperature:      $TEMPERATURE"
echo "Batch size:       $BATCH_SIZE"
echo "KoLeo weight:     $KOLEO_WEIGHT"
echo "CLS weight:       $CLS_WEIGHT"
echo "Multi-crop:       $MULTI_CROP"
echo "Concat CLS+Patch: $CONCAT_CLS_PATCH"
echo "Output base dir:  $OUTPUT_BASE_DIR"
echo "Seeds:            $SEEDS"
if [[ -n "$PRETRAINED_PATH" ]]; then
    echo "Pretrained path:  $PRETRAINED_PATH"
fi
echo "============================================================"
echo ""

# =============================================================================
# Environment setup
# =============================================================================

# --- Writable cache directories ---
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# --- CUDA memory allocation ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Shared memory for DataLoader workers ---
SHM_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
mkdir -p "$SHM_DIR"

# =============================================================================
# Build command arguments
# =============================================================================
KNN_ARGS=(
    --dataset "$DATASET"
    --gpus "$GPUS"
    --init-mode "$INIT_MODE"
    --checkpoint-type "$CHECKPOINT_TYPE"
    --k "$K"
    --temperature "$TEMPERATURE"
    --batch-size "$BATCH_SIZE"
    --koleo-weight "$KOLEO_WEIGHT"
    --cls-weight "$CLS_WEIGHT"
    --output-base-dir "$OUTPUT_BASE_DIR"
    --seeds "$SEEDS"
)

if [[ -n "$NUM_PROTOTYPES" ]]; then
    KNN_ARGS+=(--num-prototypes "$NUM_PROTOTYPES")
fi

if [[ -n "$PRETRAINED_PATH" ]]; then
    KNN_ARGS+=(--pretrained-path "$PRETRAINED_PATH")
fi

if [[ "$MULTI_CROP" == "true" ]]; then
    KNN_ARGS+=(--multi-crop --local-crops-number "$LOCAL_CROPS_NUMBER")
else
    KNN_ARGS+=(--no-multi-crop)
fi

if [[ "$CONCAT_CLS_PATCH" == "true" ]]; then
    KNN_ARGS+=(--concat-cls-patch)
fi

# =============================================================================
# Run k-NN evaluation
# =============================================================================

# IMPORTANT: Disable PyTorch Lightning's SLURM auto-detection
unset SLURM_NTASKS
unset SLURM_PROCID
unset SLURM_LOCALID
unset SLURM_NODEID
export SLURM_JOB_NAME="bash"

if [[ "$USE_SINGULARITY" == "true" ]]; then
    # Export environment variables for Singularity
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
        ./scripts/run_knn.sh "${KNN_ARGS[@]}"
else
    srun ./scripts/run_knn.sh "${KNN_ARGS[@]}"
fi

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "KODIAK k-NN evaluation finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

# Clean up shared memory
rm -rf "$SHM_DIR"

exit $EXIT_CODE
