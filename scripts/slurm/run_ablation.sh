#!/bin/bash

#SBATCH --job-name=kodiak-ablation
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu

# =============================================================================
# KODIAK Ablation Study - SLURM Script
# =============================================================================
# Runs ablation experiments to measure component contributions.
#
# Usage (one ablation per job):
#   ABLATION=full sbatch scripts/slurm/run_ablation.sh
#   ABLATION=no_sinkhorn sbatch scripts/slurm/run_ablation.sh
#   ABLATION=no_cls_loss sbatch scripts/slurm/run_ablation.sh
#   ABLATION=no_koleo sbatch scripts/slurm/run_ablation.sh
#
#   # Submit all four at once:
#   for ab in full no_sinkhorn no_cls_loss no_koleo; do
#       ABLATION=$ab sbatch scripts/slurm/run_ablation.sh
#   done
#
#   # Override defaults with environment variables:
#   DATASET=dtd PRETRAIN_EPOCHS=500 ABLATION=full sbatch scripts/slurm/run_ablation.sh
#
#   # Use Singularity container:
#   USE_SINGULARITY=true SIF_IMAGE=/path/to/deeplearning.sif \
#       ABLATION=full sbatch scripts/slurm/run_ablation.sh
#
# Ablation configurations:
#   - full:         All components enabled (baseline)
#   - no_sinkhorn:  Softmax instead of Sinkhorn-Knopp optimal transport
#   - no_cls_loss:  Disable multi-crop CLS distillation loss
#   - no_koleo:     Disable KoLeo regularization
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
echo "KODIAK Ablation Study (SLURM)"
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

# Single ablation per job (required)
ABLATION="${ABLATION:-}"
if [[ -z "$ABLATION" ]]; then
    echo "ERROR: ABLATION environment variable is required."
    echo "Usage: ABLATION=full sbatch scripts/slurm/run_ablation.sh"
    echo "Valid values: full, no_sinkhorn, no_cls_loss, no_koleo"
    exit 1
fi

VALID_ABLATIONS="full no_sinkhorn no_cls_loss no_koleo"
if [[ ! " $VALID_ABLATIONS " =~ " $ABLATION " ]]; then
    echo "ERROR: Invalid ablation '$ABLATION'. Valid: $VALID_ABLATIONS"
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
SEEDS="${SEEDS:-0 1 42}"
INIT_MODE="${INIT_MODE:-scratch}"

# Training configuration
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-500}"
PRETRAIN_LR="${PRETRAIN_LR:-0.0001}"
CLASSIFY_EPOCHS="${CLASSIFY_EPOCHS:-100}"
CLASSIFY_LR="${CLASSIFY_LR:-0.0001}"
CLASSIFY_MODE="${CLASSIFY_MODE:-finetune}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# Model configuration
NUM_PROTOTYPES="${NUM_PROTOTYPES:-128}"
KOLEO_WEIGHT="${KOLEO_WEIGHT:-0.1}"
CLS_WEIGHT="${CLS_WEIGHT:-1.0}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-ablations}"
LOGGER="${LOGGER:-csv}"

echo "Dataset:          $DATASET"
echo "GPUs:             $GPUS"
echo "Ablation:         $ABLATION"
echo "Seeds:            $SEEDS"
echo "Init mode:        $INIT_MODE"
echo "Pretrain epochs:  $PRETRAIN_EPOCHS"
echo "Pretrain LR:      $PRETRAIN_LR"
echo "Classify epochs:  $CLASSIFY_EPOCHS"
echo "Classify LR:      $CLASSIFY_LR"
echo "Classify mode:    $CLASSIFY_MODE"
echo "Batch size:       $BATCH_SIZE"
echo "Num prototypes:   $NUM_PROTOTYPES"
echo "KoLeo weight:     $KOLEO_WEIGHT"
echo "CLS weight:       $CLS_WEIGHT"
echo "Output dir:       $OUTPUT_DIR"
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
ABLATION_ARGS=(
    --dataset "$DATASET"
    --gpus "$GPUS"
    --ablations "$ABLATION"
    --seeds "$SEEDS"
    --init-mode "$INIT_MODE"
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
    --logger "$LOGGER"
)

# =============================================================================
# Run ablation
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
    # Forward the dataset root (used by configs/pancreatic/*.yaml) into the container
    EXTRA_BINDS=()
    if [[ -n "$KODIAK_DATA_DIR" ]]; then
        export SINGULARITYENV_KODIAK_DATA_DIR="$KODIAK_DATA_DIR"
        EXTRA_BINDS+=(--bind "$KODIAK_DATA_DIR":"$KODIAK_DATA_DIR")
    fi

    srun singularity exec --nv \
        --bind "$SLURM_SUBMIT_DIR":"$SLURM_SUBMIT_DIR" \
        "${EXTRA_BINDS[@]}" \
        --bind /tmp:/tmp \
        --bind "$HF_HOME":"$HF_HOME" \
        --bind "$TORCH_HOME":"$TORCH_HOME" \
        --bind "$SHM_DIR":/dev/shm \
        "$SIF_IMAGE" \
        ./scripts/run_ablation.sh "${ABLATION_ARGS[@]}"
else
    # Run with srun (inherits GPU allocation from SLURM)
    srun ./scripts/run_ablation.sh "${ABLATION_ARGS[@]}"
fi

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "KODIAK ablation finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

# Clean up shared memory
rm -rf "$SHM_DIR"

exit $EXIT_CODE
