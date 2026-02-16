#!/bin/bash
# =============================================================================
# KODIAK Sweep Script
# =============================================================================
# Runs the full KODIAK pipeline: pretrain → classify
# Supports training from scratch and continued pretraining.
#
# Usage:
#   ./scripts/run_sweep.sh \
#       --dataset dtd \
#       --gpus 1,2,3 \
#       --pretrain-epochs 500 \
#       --classify-epochs 100 \
#       --batch-size 128
#
#   # With continued pretraining from DINOv3:
#   ./scripts/run_sweep.sh \
#       --dataset dtd \
#       --gpus 1,2,3 \
#       --init-mode continued \
#       --pretrain-epochs 100
#
#   # Prototype analysis sweep:
#   ./scripts/run_sweep.sh \
#       --dataset dtd \
#       --gpus 0,1 \
#       --num-prototypes 256 \
#       --pretrain-epochs 300
#
# Output structure (scratch):
#   output/
#   ├── checkpoints/pretraining/{dataset}/kodiak_{dataset}_proto{N}_koleo{W}_cls{C}_mc/
#   └── classification/{dataset}/proto{N}_koleo{W}_cls{C}_mc/{mode}_seed_{seed}/
#
# Output structure (continued):
#   output/
#   ├── checkpoints/pretraining/{dataset}/kodiak_{dataset}_continued_proto{N}_koleo{W}_cls{C}_mc/
#   └── classification_from_continued/{dataset}/proto{N}_koleo{W}_cls{C}_mc/{mode}_seed_{seed}/
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --dataset DATASET        Dataset name (dtd, eurosat, oxford_pets, nctcrche100k, imagenet1k, cifar100)"
    echo "  --gpus GPUS              GPU indices, comma-separated (e.g., 0,1,2,3)"
    echo ""
    echo "Optional:"
    echo "  --init-mode MODE         Initialization: scratch or continued (default: scratch)"
    echo "  --seeds \"S1 S2 ..\"       Classification seeds (default: 0 1 42)"
    echo "  --pretrain-seed SEED     Pretraining seed (default: 42)"
    echo "  --pretrain-epochs N      Pretraining epochs (default: 300 for scratch, 100 for continued)"
    echo "  --pretrain-lr LR         Pretraining learning rate (default: 0.001)"
    echo "  --classify-epochs N      Classification epochs (default: 100)"
    echo "  --classify-lr LR         Classification learning rate (default: 0.001)"
    echo "  --classify-mode MODE     finetune or lineareval (default: finetune)"
    echo "  --batch-size N           Batch size (default: 128)"
    echo "  --num-prototypes N       Number of prototypes (default: 4096)"
    echo "  --koleo-weight W         KoLeo loss weight (default: 0.1)"
    echo "  --cls-weight W           CLS loss weight (default: 1.0)"
    echo "  --output-dir DIR         Output directory (default: output)"
    echo "  --skip-pretrain          Skip pretraining, use existing checkpoint"
    echo "  --skip-classify          Skip classification"
    echo "  --multi-crop             Enable multi-crop training"
    echo "  --compile                Enable torch.compile() for ~15-30% speedup"
    echo "  --no-compile             Disable torch.compile()"
    echo "  --concat-cls-patch       Concatenate CLS + mean patch tokens for classification (2x features)"
    echo "  --save-every N           Save checkpoint every N epochs (default: from config)"
    echo "  --logger TYPE            Logger type: csv, tensorboard, wandb (default: csv)"
    echo "  --help                   Show this help"
    echo ""
    echo "Examples:"
    echo "  # Pretrain 500 epochs + finetune 100 epochs on GPUs 1,2,3"
    echo "  $0 --dataset dtd --gpus 1,2,3 --pretrain-epochs 500 --classify-epochs 100 --batch-size 128"
    echo ""
    echo "  # Continued pretraining from DINOv3"
    echo "  $0 --dataset dtd --gpus 0,1 --init-mode continued --pretrain-epochs 100"
    echo ""
    echo "  # Prototype analysis (different prototype counts)"
    echo "  $0 --dataset dtd --gpus 0,1 --num-prototypes 256 --pretrain-epochs 300"
    echo ""
    echo "  # Only classification (skip pretrain)"
    echo "  $0 --dataset dtd --gpus 0 --skip-pretrain --pretrained-path /path/to/checkpoint.ckpt"
    exit 1
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
DATASET=""
GPUS=""
INIT_MODE="scratch"
SEEDS="0 1 42"
PRETRAIN_SEED="42"
PRETRAIN_EPOCHS=""
PRETRAIN_LR="0.0001"
CLASSIFY_EPOCHS="100"
CLASSIFY_LR="0.0001"
CLASSIFY_MODE="finetune"
BATCH_SIZE="128"
NUM_PROTOTYPES="4096"
KOLEO_WEIGHT="0.1"
CLS_WEIGHT="1.0"
OUTPUT_DIR="output"
SKIP_PRETRAIN=false
SKIP_CLASSIFY=false
PRETRAINED_PATH=""
MULTI_CROP=true
COMPILE=false
CONCAT_CLS_PATCH=false
SAVE_EVERY_N_EPOCHS=""
LOGGER="csv"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --init-mode)
            INIT_MODE="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --pretrain-seed)
            PRETRAIN_SEED="$2"
            shift 2
            ;;
        --pretrain-epochs)
            PRETRAIN_EPOCHS="$2"
            shift 2
            ;;
        --pretrain-lr)
            PRETRAIN_LR="$2"
            shift 2
            ;;
        --classify-epochs)
            CLASSIFY_EPOCHS="$2"
            shift 2
            ;;
        --classify-lr)
            CLASSIFY_LR="$2"
            shift 2
            ;;
        --classify-mode)
            CLASSIFY_MODE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-prototypes)
            NUM_PROTOTYPES="$2"
            shift 2
            ;;
        --koleo-weight)
            KOLEO_WEIGHT="$2"
            shift 2
            ;;
        --cls-weight)
            CLS_WEIGHT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-pretrain)
            SKIP_PRETRAIN=true
            shift
            ;;
        --skip-classify)
            SKIP_CLASSIFY=true
            shift
            ;;
        --pretrained-path)
            PRETRAINED_PATH="$2"
            shift 2
            ;;
        --multi-crop)
            MULTI_CROP=true
            shift
            ;;
        --compile)
            COMPILE=true
            shift
            ;;
        --no-compile)
            COMPILE=false
            shift
            ;;
        --concat-cls-patch)
            CONCAT_CLS_PATCH=true
            shift
            ;;
        --save-every)
            SAVE_EVERY_N_EPOCHS="$2"
            shift 2
            ;;
        --logger)
            LOGGER="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASET" ]]; then
    echo "ERROR: --dataset is required"
    usage
fi

if [[ -z "$GPUS" ]]; then
    echo "ERROR: --gpus is required"
    usage
fi

# Validate init mode
if [[ "$INIT_MODE" != "scratch" && "$INIT_MODE" != "continued" ]]; then
    echo "ERROR: --init-mode must be 'scratch' or 'continued'"
    exit 1
fi

# Set default pretrain epochs based on init mode
if [[ -z "$PRETRAIN_EPOCHS" ]]; then
    if [[ "$INIT_MODE" == "continued" ]]; then
        PRETRAIN_EPOCHS="100"
    else
        PRETRAIN_EPOCHS="300"
    fi
fi

# -----------------------------------------------------------------------------
# Paths and configs
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Map dataset to config directory
declare -A CONFIG_DIRS=(
    ["dtd"]="DTD"
    ["eurosat"]="eurosat"
    ["oxford_pets"]="oxford_pets"
    ["nctcrche100k"]="NCTCRCHE100K"
    ["imagenet1k"]="imagenet1k"
    ["cifar100"]="cifar100"
    ["pancreatic"]="pancreatic"
)

CONFIG_DIR="${CONFIG_DIRS[$DATASET]}"
if [[ -z "$CONFIG_DIR" ]]; then
    echo "ERROR: Unknown dataset: $DATASET"
    echo "Available: ${!CONFIG_DIRS[@]}"
    exit 1
fi

# Select pretrain config based on init mode
if [[ "$INIT_MODE" == "continued" ]]; then
    PRETRAIN_CONFIG="configs/${CONFIG_DIR}/pretrain_continued.yaml"
    EXPERIMENT_SUFFIX="_continued"
    CLASSIFICATION_DIR="classification_from_continued"
else
    PRETRAIN_CONFIG="configs/${CONFIG_DIR}/pretrain.yaml"
    EXPERIMENT_SUFFIX=""
    CLASSIFICATION_DIR="classification"
fi

CLASSIFY_CONFIG="configs/${CONFIG_DIR}/classify.yaml"

if [[ ! -f "$PRETRAIN_CONFIG" ]]; then
    echo "ERROR: Pretrain config not found: $PRETRAIN_CONFIG"
    exit 1
fi

# Compute devices string (0-indexed for PyTorch)
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
if [[ $NUM_GPUS -eq 1 ]]; then
    DEVICES="0"
else
    DEVICES=$(seq -s',' 0 $((NUM_GPUS - 1)))
fi

# Build experiment name (for pretraining)
# Format: kodiak_{dataset}_proto{N}_koleo{W}_cls{C}[_mc]
EXP_NAME="kodiak_${DATASET}${EXPERIMENT_SUFFIX}_proto${NUM_PROTOTYPES}_koleo${KOLEO_WEIGHT}_cls${CLS_WEIGHT}"
if [[ "$MULTI_CROP" == true ]]; then
    EXP_NAME="${EXP_NAME}_mc"
fi

# Build classification folder name (matches train_classifier.py auto-generated structure)
# Format: proto{N}_koleo{W}_cls{C}[_mc]
PRETRAIN_FOLDER="proto${NUM_PROTOTYPES}_koleo${KOLEO_WEIGHT}_cls${CLS_WEIGHT}"
if [[ "$MULTI_CROP" == true ]]; then
    PRETRAIN_FOLDER="${PRETRAIN_FOLDER}_mc"
fi

# Output directories
# Note: train.py constructs checkpoint_dir as: base_dir/dataset/exp_name
PRETRAIN_CKPT_DIR="$OUTPUT_DIR/checkpoints/pretraining/$DATASET/$EXP_NAME"
PRETRAIN_LOG_DIR="$OUTPUT_DIR/logs/pretraining/$DATASET"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

separator() {
    echo "========================================================================"
}

find_checkpoint() {
    local ckpt_dir="$1"
    if [[ -f "$ckpt_dir/last.ckpt" ]]; then
        echo "$ckpt_dir/last.ckpt"
    else
        find "$ckpt_dir" -name "*.ckpt" -type f 2>/dev/null | sort | tail -1
    fi
}

# -----------------------------------------------------------------------------
# Print configuration
# -----------------------------------------------------------------------------
separator
echo "KODIAK SWEEP"
separator
echo "Dataset:          $DATASET"
echo "GPUs:             $GPUS (devices: $DEVICES)"
echo "Init mode:        $INIT_MODE"
echo "Batch size:       $BATCH_SIZE"
echo "Num prototypes:   $NUM_PROTOTYPES"
echo "KoLeo weight:     $KOLEO_WEIGHT"
echo "Multi-crop:       $MULTI_CROP"
echo "torch.compile:    $COMPILE"
echo "Logger:           $LOGGER"
echo ""
echo "Pretraining:"
echo "  Config:         $PRETRAIN_CONFIG"
echo "  Epochs:         $PRETRAIN_EPOCHS"
echo "  LR:             $PRETRAIN_LR"
echo "  Seed:           $PRETRAIN_SEED"
echo "  Checkpoints:    $PRETRAIN_CKPT_DIR"
echo ""
echo "Classification:"
echo "  Config:         $CLASSIFY_CONFIG"
echo "  Mode:           $CLASSIFY_MODE"
echo "  Epochs:         $CLASSIFY_EPOCHS"
echo "  LR:             $CLASSIFY_LR"
echo "  Seeds:          $SEEDS"
echo "  Output folder:  $CLASSIFICATION_DIR/$DATASET/$PRETRAIN_FOLDER/"
separator
echo ""

read -ra SEED_ARRAY <<< "$SEEDS"

# -----------------------------------------------------------------------------
# PRETRAINING
# -----------------------------------------------------------------------------
if [[ "$SKIP_PRETRAIN" == false ]]; then
    separator
    log "PHASE 1: PRETRAINING"
    separator

    # Check if checkpoint already exists
    EXISTING_CKPT=$(find_checkpoint "$PRETRAIN_CKPT_DIR" 2>/dev/null || echo "")

    if [[ -n "$EXISTING_CKPT" ]]; then
        log "Found existing checkpoint: $EXISTING_CKPT"
        log "Skipping pretraining (use --skip-pretrain to explicitly skip)"
        PRETRAIN_CKPT="$EXISTING_CKPT"
    else
        log "Starting pretraining..."
        mkdir -p "$PRETRAIN_CKPT_DIR" "$PRETRAIN_LOG_DIR"

        # Build pretrain command
        # Pass base experiment name so train.py constructs same path as run_sweep.sh expects
        BASE_EXP_NAME="kodiak_${DATASET}${EXPERIMENT_SUFFIX}"
        PRETRAIN_CMD=(
            python scripts/train.py
            --config "$PRETRAIN_CONFIG"
            --name "$BASE_EXP_NAME"
            --devices "$DEVICES"
            --max_epochs "$PRETRAIN_EPOCHS"
            --learning_rate "$PRETRAIN_LR"
            --batch_size "$BATCH_SIZE"
            --num_prototypes "$NUM_PROTOTYPES"
            --koleo_weight "$KOLEO_WEIGHT"
            --cls_weight "$CLS_WEIGHT"
            --log_base_dir "$OUTPUT_DIR/logs/pretraining"
            --checkpoint_base_dir "$OUTPUT_DIR/checkpoints/pretraining"
            --logger "$LOGGER"
        )

        if [[ "$MULTI_CROP" == true ]]; then
            PRETRAIN_CMD+=(--multi_crop)
        fi

        if [[ "$COMPILE" == true ]]; then
            PRETRAIN_CMD+=(--compile)
        else
            PRETRAIN_CMD+=(--no_compile)
        fi

        if [[ -n "$SAVE_EVERY_N_EPOCHS" ]]; then
            PRETRAIN_CMD+=(--save_every_n_epochs "$SAVE_EVERY_N_EPOCHS")
        fi

        CUDA_VISIBLE_DEVICES="$GPUS" "${PRETRAIN_CMD[@]}"

        # Find the checkpoint
        PRETRAIN_CKPT=$(find "$PRETRAIN_CKPT_DIR" -name "last.ckpt" -type f 2>/dev/null | head -1)

        if [[ -z "$PRETRAIN_CKPT" ]]; then
            PRETRAIN_CKPT=$(find "$PRETRAIN_CKPT_DIR" -name "*.ckpt" -type f 2>/dev/null | sort | tail -1)
        fi

        if [[ -z "$PRETRAIN_CKPT" ]]; then
            log "ERROR: Pretraining failed, no checkpoint found"
            exit 1
        fi

        log "Pretraining complete: $PRETRAIN_CKPT"
    fi
else
    log "Skipping pretraining (--skip-pretrain)"
    if [[ -n "$PRETRAINED_PATH" ]]; then
        PRETRAIN_CKPT="$PRETRAINED_PATH"
    else
        PRETRAIN_CKPT=$(find_checkpoint "$PRETRAIN_CKPT_DIR" 2>/dev/null || echo "")
    fi

    if [[ -z "$PRETRAIN_CKPT" ]]; then
        log "ERROR: No pretrained checkpoint found. Provide --pretrained-path or run pretraining first."
        exit 1
    fi
    log "Using checkpoint: $PRETRAIN_CKPT"
fi

echo ""

# -----------------------------------------------------------------------------
# CLASSIFICATION
# -----------------------------------------------------------------------------
if [[ "$SKIP_CLASSIFY" == false ]]; then
    separator
    log "PHASE 2: CLASSIFICATION"
    separator

    for SEED in "${SEED_ARRAY[@]}"; do
        log "Classification with seed $SEED..."

        # Structure mirrors pretraining:
        #   logs/classification/{dataset}/{pretrain_folder}/{mode}_seed_{seed}/
        #   checkpoints/classification/{dataset}/{pretrain_folder}/{mode}_seed_{seed}/
        CLASS_RUN="${CLASSIFY_MODE}_seed_${SEED}"
        CLASS_LOG_DIR="$OUTPUT_DIR/logs/$CLASSIFICATION_DIR/$DATASET/$PRETRAIN_FOLDER/$CLASS_RUN"
        CLASS_CKPT_DIR="$OUTPUT_DIR/checkpoints/$CLASSIFICATION_DIR/$DATASET/$PRETRAIN_FOLDER/$CLASS_RUN"

        # Check if checkpoint exists - skip if so
        EXISTING_CLASS_CKPT=$(find_checkpoint "$CLASS_CKPT_DIR" 2>/dev/null || echo "")
        if [[ -n "$EXISTING_CLASS_CKPT" ]]; then
            log "Checkpoint already exists for seed $SEED: $EXISTING_CLASS_CKPT"
            log "Skipping seed $SEED"
            continue
        fi

        # No checkpoint exists - run full training
        mkdir -p "$CLASS_CKPT_DIR" "$CLASS_LOG_DIR"

        # Build classification command
        CLASS_CMD=(
            python scripts/train_classifier.py
            --config "$CLASSIFY_CONFIG"
            --pretrained_path "$PRETRAIN_CKPT"
            --devices "$DEVICES"
            --max_epochs "$CLASSIFY_EPOCHS"
            --learning_rate "$CLASSIFY_LR"
            --batch_size "$BATCH_SIZE"
            --seed "$SEED"
            --log_dir "$CLASS_LOG_DIR"
            --checkpoint_dir "$CLASS_CKPT_DIR"
            --encoder_type "teacher"
            --logger "$LOGGER"
        )

        # Add freeze_backbone for linear evaluation
        if [[ "$CLASSIFY_MODE" == "lineareval" ]]; then
            CLASS_CMD+=(--freeze_backbone)
        fi

        # Add concat_cls_patch if enabled
        if [[ "$CONCAT_CLS_PATCH" == true ]]; then
            CLASS_CMD+=(--concat_cls_patch)
        fi

        CUDA_VISIBLE_DEVICES="$GPUS" "${CLASS_CMD[@]}"

        log "Classification seed $SEED complete"
    done
else
    log "Skipping classification (--skip-classify)"
fi

echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
separator
log "SWEEP COMPLETE"
separator
echo ""
echo "Output structure:"
echo "  $OUTPUT_DIR/"
echo "  ├── logs/"
echo "  │   ├── pretraining/$DATASET/$EXP_NAME/"
echo "  │   └── $CLASSIFICATION_DIR/$DATASET/$PRETRAIN_FOLDER/"
for SEED in "${SEED_ARRAY[@]}"; do
echo "  │       └── ${CLASSIFY_MODE}_seed_${SEED}/"
done
echo "  └── checkpoints/"
echo "      ├── pretraining/$DATASET/$EXP_NAME/"
echo "      └── $CLASSIFICATION_DIR/$DATASET/$PRETRAIN_FOLDER/"
for SEED in "${SEED_ARRAY[@]}"; do
echo "          └── ${CLASSIFY_MODE}_seed_${SEED}/"
done
echo ""
echo "Pretrain checkpoint: $PRETRAIN_CKPT"
separator
