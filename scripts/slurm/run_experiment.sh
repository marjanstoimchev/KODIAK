#!/bin/bash

#SBATCH --job-name=proto
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu

# =============================================================================
# KODIAK Experiment Runner - SLURM Script
# =============================================================================
# Runs the full KODIAK pipeline (pretrain + classify) as a single SLURM job.
# All parameters are configurable via environment variables.
#
# Usage:
#   NUM_PROTOTYPES=128 sbatch scripts/slurm/run_experiment.sh
#
#   # Override any parameter:
#   DATASET=eurosat NUM_PROTOTYPES=256 PRETRAIN_EPOCHS=300 \
#       sbatch scripts/slurm/run_experiment.sh
#
#   # Use Singularity container:
#   USE_SINGULARITY=true SIF_IMAGE=/path/to/deeplearning.sif \
#       NUM_PROTOTYPES=128 sbatch scripts/slurm/run_experiment.sh
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
echo "KODIAK Experiment Runner (SLURM)"
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

# Single prototype count per job (required)
NUM_PROTOTYPES="${NUM_PROTOTYPES:-}"
if [[ -z "$NUM_PROTOTYPES" ]]; then
    echo "ERROR: NUM_PROTOTYPES environment variable is required."
    echo "Usage: NUM_PROTOTYPES=128 sbatch scripts/slurm/run_experiment.sh"
    exit 1
fi

# Detect number of GPUs from SLURM allocation
if [[ -n "$SLURM_GPUS_ON_NODE" ]]; then
    NUM_GPUS="$SLURM_GPUS_ON_NODE"
elif [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=1
fi

# Build GPU list as 0,1,2... (local indices after SLURM remapping)
GPUS=$(seq -s',' 0 $((NUM_GPUS - 1)))
echo "Detected $NUM_GPUS GPUs, using indices: $GPUS"

# Configuration
INIT_MODE="${INIT_MODE:-scratch}"
SEEDS="${SEEDS:-0 1 42}"
PRETRAIN_SEED="${PRETRAIN_SEED:-42}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-500}"
PRETRAIN_LR="${PRETRAIN_LR:-0.0001}"
CLASSIFY_EPOCHS="${CLASSIFY_EPOCHS:-100}"
CLASSIFY_LR="${CLASSIFY_LR:-0.0001}"
CLASSIFY_MODE="${CLASSIFY_MODE:-finetune}"
BATCH_SIZE="${BATCH_SIZE:-128}"
KOLEO_WEIGHT="${KOLEO_WEIGHT:-0.1}"
CLS_WEIGHT="${CLS_WEIGHT:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-output_proto_analysis}"
MULTI_CROP="${MULTI_CROP:-true}"
COMPILE="${COMPILE:-false}"
SKIP_PRETRAIN="${SKIP_PRETRAIN:-false}"
SKIP_CLASSIFY="${SKIP_CLASSIFY:-false}"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-}"
LOGGER="${LOGGER:-csv}"

echo "Dataset:          $DATASET"
echo "GPUs:             $GPUS"
echo "Num prototypes:   $NUM_PROTOTYPES"
echo "Init mode:        $INIT_MODE"
echo "Seeds:            $SEEDS"
echo "Pretrain seed:    $PRETRAIN_SEED"
echo "Pretrain epochs:  $PRETRAIN_EPOCHS"
echo "Pretrain LR:      $PRETRAIN_LR"
echo "Classify epochs:  $CLASSIFY_EPOCHS"
echo "Classify LR:      $CLASSIFY_LR"
echo "Classify mode:    $CLASSIFY_MODE"
echo "Batch size:       $BATCH_SIZE"
echo "KoLeo weight:     $KOLEO_WEIGHT"
echo "CLS weight:       $CLS_WEIGHT"
echo "Multi-crop:       $MULTI_CROP"
echo "torch.compile:    $COMPILE"
echo "Output dir:       $OUTPUT_DIR"
echo "Skip pretrain:    $SKIP_PRETRAIN"
echo "Skip classify:    $SKIP_CLASSIFY"
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
SWEEP_ARGS=(
    --dataset "$DATASET"
    --gpus "$GPUS"
    --init-mode "$INIT_MODE"
    --seeds "$SEEDS"
    --pretrain-seed "$PRETRAIN_SEED"
    --pretrain-epochs "$PRETRAIN_EPOCHS"
    --pretrain-lr "$PRETRAIN_LR"
    --classify-epochs "$CLASSIFY_EPOCHS"
    --classify-lr "$CLASSIFY_LR"
    --classify-mode "$CLASSIFY_MODE"
    --batch-size "$BATCH_SIZE"
    --num-prototypes "$NUM_PROTOTYPES"
    --koleo-weight "$KOLEO_WEIGHT"
    --cls-weight "$CLS_WEIGHT"
    --output-dir "$OUTPUT_DIR"
)

if [[ "$SKIP_PRETRAIN" == "true" ]]; then
    SWEEP_ARGS+=(--skip-pretrain)
fi

if [[ "$SKIP_CLASSIFY" == "true" ]]; then
    SWEEP_ARGS+=(--skip-classify)
fi

if [[ -n "$PRETRAINED_PATH" ]]; then
    SWEEP_ARGS+=(--pretrained-path "$PRETRAINED_PATH")
fi

if [[ "$MULTI_CROP" == "true" ]]; then
    SWEEP_ARGS+=(--multi-crop)
fi

if [[ "$COMPILE" == "true" ]]; then
    SWEEP_ARGS+=(--compile)
else
    SWEEP_ARGS+=(--no-compile)
fi

if [[ -n "$SAVE_EVERY_N_EPOCHS" ]]; then
    SWEEP_ARGS+=(--save-every "$SAVE_EVERY_N_EPOCHS")
fi

SWEEP_ARGS+=(--logger "$LOGGER")

# =============================================================================
# Run sweep
# =============================================================================

# IMPORTANT: Disable PyTorch Lightning's SLURM auto-detection
# Lightning expects ntasks-per-node == num_gpus, but we want Lightning to handle
# multi-GPU DDP internally within a single SLURM task
unset SLURM_NTASKS
unset SLURM_PROCID
unset SLURM_LOCALID
unset SLURM_NODEID
export SLURM_JOB_NAME="bash"  # Trick Lightning into thinking this isn't a SLURM job

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
        ./scripts/run_sweep.sh "${SWEEP_ARGS[@]}"
else
    # Run with srun (inherits GPU allocation from SLURM)
    srun ./scripts/run_sweep.sh "${SWEEP_ARGS[@]}"
fi

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "KODIAK experiment (proto=$NUM_PROTOTYPES) finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

# Clean up shared memory
rm -rf "$SHM_DIR"

exit $EXIT_CODE
