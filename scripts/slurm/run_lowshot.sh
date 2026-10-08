#!/bin/bash
# =============================================================================
# SLURM K-Shot Evaluation Script
# =============================================================================
# Submits k-shot (few-shot) evaluation sweep as a SLURM job.
# Wraps scripts/run_lowshot.sh with SLURM resource allocation.
#
# Environment variable overrides (set before sbatch):
#   INIT_MODE         scratch or continued (default: continued)
#   METHODS           Space-separated methods (default: "kodiak")
#   DATASETS          Space-separated datasets (default: "cifar100 dtd")
#   SHOTS             K-shot values (default: "1 2 4 8 16")
#   LABEL_SEEDS       Label subset seeds (default: "0 1 42")
#   TRAIN_SEEDS       Training seeds (default: "42")
#   MODE              finetune, lineareval, or both (default: both)
#   MAX_EPOCHS        Override max training epochs
#   BATCH_SIZE        Override batch size
#   LEARNING_RATE     Override learning rate
#   OUTPUT_DIR        Override output directory
#   CKPT_BASE_DIR     Base dir for kodiak checkpoints (default: output_proto_analysis)
#   PROTOTYPES        Number of prototypes (default: 128)
#   USE_SINGULARITY   Set to "true" to run inside a Singularity container
#   SIF_IMAGE         Path to .sif image (default: $HOME/deeplearning.sif)
#
# SBATCH defaults (override via sbatch flags):
#   --gres=gpu:1      1 GPU
#   --cpus-per-task=22
#   --mem=64G
#   --time=3-00:00:00 3 days wall time (many sequential runs)
#   --partition=gpu
#
# Examples:
#   # KODIAK on CIFAR-100, continued, fine-tuning + linear eval
#   USE_SINGULARITY=true \
#   DATASETS=cifar100 SHOTS="1 2 4 8 16" \
#       sbatch scripts/slurm/run_lowshot.sh
#
#   # KODIAK on Pancreatic, scratch, fine-tuning only
#   USE_SINGULARITY=true \
#   INIT_MODE=scratch DATASETS=pancreatic MODE=finetune \
#   LEARNING_RATE=1e-4 MAX_EPOCHS=50 BATCH_SIZE=64 \
#       sbatch scripts/slurm/run_lowshot.sh
#
#   # Multiple datasets, custom shots, with constraint
#   USE_SINGULARITY=true \
#   DATASETS="cifar100 dtd eurosat" SHOTS="1 4 16" \
#   MAX_EPOCHS=50 BATCH_SIZE=128 \
#       sbatch --constraint=h100 scripts/slurm/run_lowshot.sh
# =============================================================================

#SBATCH --job-name=kodiak-lowshot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=22
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm-lowshot-%j.out
#SBATCH --error=logs/slurm-lowshot-%j.err

set -e

# Navigate to repo root
cd "$SLURM_SUBMIT_DIR" || exit 1

# Create log directory
mkdir -p logs

# =============================================================================
# Configuration (override via environment variables)
# =============================================================================
INIT_MODE="${INIT_MODE:-continued}"
METHODS="${METHODS:-kodiak}"
DATASETS="${DATASETS:-cifar100 dtd}"
SHOTS="${SHOTS:-1 2 4 8 16}"
LABEL_SEEDS="${LABEL_SEEDS:-0 1 42}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
MODE="${MODE:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-output_proto_analysis}"
PROTOTYPES="${PROTOTYPES:-128}"
LOCAL_CROPS_NUMBER="${LOCAL_CROPS_NUMBER:-8}"

# Singularity container setup (optional)
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
echo "KODIAK K-Shot Evaluation (SLURM)"
echo "============================================================"
echo "SLURM Job ID:    $SLURM_JOB_ID"
echo "Node:            $SLURM_NODELIST"
echo "GPUs:            $CUDA_VISIBLE_DEVICES"
echo "Init mode:       $INIT_MODE"
echo "Methods:         $METHODS"
echo "Datasets:        $DATASETS"
echo "Shots:           $SHOTS"
echo "Label seeds:     $LABEL_SEEDS"
echo "Train seeds:     $TRAIN_SEEDS"
echo "Mode:            ${MODE:-both}"
if [[ "$USE_SINGULARITY" == "true" ]]; then
    echo "Container:       $SIF_IMAGE"
fi
[ -n "$MAX_EPOCHS" ] && echo "Max epochs:      $MAX_EPOCHS"
[ -n "$BATCH_SIZE" ] && echo "Batch size:      $BATCH_SIZE"
[ -n "$LEARNING_RATE" ] && echo "Learning rate:   $LEARNING_RATE"
[ -n "$OUTPUT_DIR" ] && echo "Output dir:      $OUTPUT_DIR"
echo "Ckpt base dir:   $CKPT_BASE_DIR"
echo "Prototypes:      $PROTOTYPES"
echo "Start time:      $(date)"
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

# CRITICAL: Disable PyTorch Lightning's SLURM auto-detection
unset SLURM_NTASKS
unset SLURM_PROCID
unset SLURM_LOCALID
unset SLURM_NODEID
export SLURM_JOB_NAME="bash"

# =============================================================================
# Build command
# =============================================================================
CMD="./scripts/run_lowshot.sh --gpus 0 --init-mode $INIT_MODE --methods '$METHODS' --datasets '$DATASETS' --shots '$SHOTS' --label-seeds '$LABEL_SEEDS' --train-seeds '$TRAIN_SEEDS' --ckpt-base-dir $CKPT_BASE_DIR --prototypes $PROTOTYPES --local-crops-number $LOCAL_CROPS_NUMBER"
[ -n "$MODE" ] && CMD="$CMD --mode $MODE"
[ -n "$MAX_EPOCHS" ] && CMD="$CMD --max-epochs $MAX_EPOCHS"
[ -n "$BATCH_SIZE" ] && CMD="$CMD --batch-size $BATCH_SIZE"
[ -n "$LEARNING_RATE" ] && CMD="$CMD --learning-rate $LEARNING_RATE"
[ -n "$OUTPUT_DIR" ] && CMD="$CMD --output-dir $OUTPUT_DIR"

# =============================================================================
# Run
# =============================================================================
if [[ "$USE_SINGULARITY" == "true" ]]; then
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
        bash -c "$CMD"
else
    srun bash -c "$CMD"
fi

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "KODIAK K-Shot evaluation finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

# Clean up shared memory
rm -rf "$SHM_DIR"

exit $EXIT_CODE
