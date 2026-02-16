#!/bin/bash
# =============================================================================
# KODIAK Ablation Analysis Sweep
# =============================================================================
# Runs component ablation experiments to measure the contribution of each
# part of the KODIAK (Masked Prototype Prediction) framework.
#
# Ablation configurations:
#   - full:         All components enabled (baseline)
#   - no_sinkhorn:  Softmax instead of Sinkhorn-Knopp optimal transport
#   - no_cls_loss:  Disable multi-crop CLS distillation loss
#   - no_koleo:     Disable KoLeo regularization
#
# Usage:
#   ./scripts/run_ablation.sh \
#       --dataset dtd \
#       --gpus 1,2,3 \
#       --ablations "full no_sinkhorn no_cls_loss no_koleo"
#
#   # Quick test
#   ./scripts/run_ablation.sh \
#       --dataset dtd \
#       --gpus 0 \
#       --ablations "full no_sinkhorn" \
#       --seeds "0" \
#       --pretrain-epochs 10 \
#       --classify-epochs 5
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --dataset DATASET        Dataset to run (dtd, eurosat, oxford_pets, etc.)"
    echo "  --gpus GPUS              GPU indices, comma-separated (e.g., 0,1,2,3)"
    echo ""
    echo "Optional:"
    echo "  --ablations \"A1 A2 ..\"   Ablations to run (default: full no_sinkhorn no_cls_loss no_koleo)"
    echo "  --seeds \"S1 S2 ..\"       Classification seeds (default: 0 1 42)"
    echo "  --pretrain-epochs N      Pretraining epochs (default: 500)"
    echo "  --pretrain-lr LR         Pretraining learning rate (default: 0.0001)"
    echo "  --classify-epochs N      Classification epochs (default: 100)"
    echo "  --classify-lr LR         Classification learning rate (default: 0.0001)"
    echo "  --classify-mode MODE     finetune or lineareval (default: finetune)"
    echo "  --batch-size N           Batch size (default: 128)"
    echo "  --num-prototypes N       Number of prototypes (default: 1024)"
    echo "  --koleo-weight W         KoLeo loss weight for baseline (default: 0.1)"
    echo "  --cls-weight W           CLS loss weight for baseline (default: 1.0)"
    echo "  --output-dir DIR         Output directory (default: ablations)"
    echo "  --init-mode MODE         scratch or continued (default: scratch)"
    echo "  --logger TYPE            Logger type: csv, tensorboard, wandb (default: csv)"
    echo "  --help                   Show this help message"
    echo ""
    echo "Available ablations:"
    echo "  full         All components enabled (baseline)"
    echo "  no_sinkhorn  Use softmax instead of Sinkhorn-Knopp"
    echo "  no_cls_loss  Disable multi-crop CLS distillation"
    echo "  no_koleo     Disable KoLeo regularization"
    exit 1
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
DATASET=""
GPUS=""
INIT_MODE="scratch"
ABLATIONS="full no_sinkhorn no_cls_loss no_koleo"
SEEDS="0 1 42"
PRETRAIN_EPOCHS="500"
PRETRAIN_LR="0.0001"
CLASSIFY_EPOCHS="100"
CLASSIFY_LR="0.0001"
CLASSIFY_MODE="finetune"
BATCH_SIZE="128"
NUM_PROTOTYPES="1024"
KOLEO_WEIGHT="0.1"
CLS_WEIGHT="1.0"
OUTPUT_DIR="ablations"
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
        --ablations)
            ABLATIONS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
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
    ["pancreatic"]="pancreatic"
)

CONFIG_DIR="${CONFIG_DIRS[$DATASET]}"
if [[ -z "$CONFIG_DIR" ]]; then
    echo "ERROR: Unknown dataset: $DATASET"
    echo "Available: ${!CONFIG_DIRS[@]}"
    exit 1
fi

# Select config based on init mode
if [[ "$INIT_MODE" == "continued" ]]; then
    PRETRAIN_CONFIG="configs/${CONFIG_DIR}/pretrain_continued.yaml"
else
    PRETRAIN_CONFIG="configs/${CONFIG_DIR}/pretrain.yaml"
fi
CLASSIFY_CONFIG="configs/${CONFIG_DIR}/classify.yaml"

# Compute devices string
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
if [[ $NUM_GPUS -eq 1 ]]; then
    DEVICES="0"
else
    DEVICES=$(seq -s',' 0 $((NUM_GPUS - 1)))
fi

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

# Get effective koleo weight for ablation
get_koleo_weight() {
    local ablation="$1"
    if [[ "$ablation" == "no_koleo" ]]; then
        echo "0.0"
    else
        echo "$KOLEO_WEIGHT"
    fi
}

# Get effective cls weight for ablation
get_cls_weight() {
    local ablation="$1"
    if [[ "$ablation" == "no_cls_loss" ]]; then
        echo "0.0"
    else
        echo "$CLS_WEIGHT"
    fi
}

# Check if ablation uses sinkhorn
use_sinkhorn() {
    local ablation="$1"
    if [[ "$ablation" == "no_sinkhorn" ]]; then
        echo "false"
    else
        echo "true"
    fi
}

# -----------------------------------------------------------------------------
# Print configuration
# -----------------------------------------------------------------------------
separator
echo "KODIAK ABLATION ANALYSIS"
separator
echo "Dataset:          $DATASET"
echo "GPUs:             $GPUS (devices: $DEVICES)"
echo "Init mode:        $INIT_MODE"
echo "Ablations:        $ABLATIONS"
echo "Seeds:            $SEEDS"
echo "Num prototypes:   $NUM_PROTOTYPES"
echo "Pretrain epochs:  $PRETRAIN_EPOCHS"
echo "Classify epochs:  $CLASSIFY_EPOCHS"
echo "Classify mode:    $CLASSIFY_MODE"
echo "Output:           $OUTPUT_DIR"
separator
echo ""

read -ra ABLATION_ARRAY <<< "$ABLATIONS"
read -ra SEED_ARRAY <<< "$SEEDS"

# -----------------------------------------------------------------------------
# Main loop - iterate over ablations and seeds
# -----------------------------------------------------------------------------
for ABLATION in "${ABLATION_ARRAY[@]}"; do
    separator
    log "ABLATION: $ABLATION"
    separator

    # Directories
    ABLATION_DIR="$OUTPUT_DIR/pretraining/$DATASET/${ABLATION}"
    CKPT_DIR="$ABLATION_DIR/checkpoints"
    LOG_DIR="$ABLATION_DIR/logs"

    # Get ablation-specific settings
    EFFECTIVE_KOLEO=$(get_koleo_weight "$ABLATION")
    EFFECTIVE_CLS=$(get_cls_weight "$ABLATION")

    # -------------------------------------------------------------
    # PRETRAINING (once per ablation)
    # -------------------------------------------------------------
    PRETRAIN_CKPT=$(find_checkpoint "$CKPT_DIR" 2>/dev/null || echo "")

    if [[ -z "$PRETRAIN_CKPT" ]]; then
        log "Pretraining ${ABLATION}..."

        mkdir -p "$CKPT_DIR" "$LOG_DIR"

        PRETRAIN_CMD=(
            python scripts/train.py
            --config "$PRETRAIN_CONFIG"
            --name "$ABLATION"
            --num_prototypes "$NUM_PROTOTYPES"
            --koleo_weight "$EFFECTIVE_KOLEO"
            --cls_weight "$EFFECTIVE_CLS"
            --devices "$DEVICES"
            --max_epochs "$PRETRAIN_EPOCHS"
            --learning_rate "$PRETRAIN_LR"
            --batch_size "$BATCH_SIZE"
            --log_dir "$LOG_DIR"
            --checkpoint_dir "$CKPT_DIR"
            --logger "$LOGGER"
        )

        # Add no_sinkhorn flag if needed
        if [[ "$ABLATION" == "no_sinkhorn" ]]; then
            PRETRAIN_CMD+=(--no_sinkhorn)
        fi

        CUDA_VISIBLE_DEVICES="$GPUS" "${PRETRAIN_CMD[@]}"

        PRETRAIN_CKPT=$(find_checkpoint "$CKPT_DIR")

        if [[ -z "$PRETRAIN_CKPT" ]]; then
            log "ERROR: Pretraining failed for ${ABLATION}"
            continue
        fi

        log "Pretraining complete: $PRETRAIN_CKPT"
    else
        log "Using existing checkpoint: $PRETRAIN_CKPT"
    fi

    # -------------------------------------------------------------
    # CLASSIFICATION (for each seed)
    # -------------------------------------------------------------
    for SEED in "${SEED_ARRAY[@]}"; do
        log "Classification ${ABLATION} seed_${SEED}..."

        # Unified structure: {output}/classification/{dataset}/{ablation}/{mode}_seed_{S}/
        # Matches sweep structure: classification/{dataset}/proto{N}_koleo{W}_cls{C}_mc/{mode}_seed_{S}/
        CLASS_DIR="$OUTPUT_DIR/classification/$DATASET/${ABLATION}/${CLASSIFY_MODE}_seed_${SEED}"
        CLASS_CKPT_DIR="$CLASS_DIR/checkpoints"
        CLASS_LOG_DIR="$CLASS_DIR/logs"

        # Check if already done (look for test_results.json)
        if [[ -f "$CLASS_LOG_DIR/test_results.json" ]]; then
            log "Results already exist: $CLASS_LOG_DIR/test_results.json"
            continue
        fi

        # Also check for checkpoint
        CLASS_EXISTING=$(find_checkpoint "$CLASS_CKPT_DIR" 2>/dev/null || echo "")
        if [[ -n "$CLASS_EXISTING" ]]; then
            log "Checkpoint exists but no results, re-running evaluation..."
        fi

        mkdir -p "$CLASS_CKPT_DIR" "$CLASS_LOG_DIR"

        CLASSIFY_CMD=(
            python scripts/train_classifier.py
            --config "$CLASSIFY_CONFIG"
            --pretrained_path "$PRETRAIN_CKPT"
            --max_epochs "$CLASSIFY_EPOCHS"
            --learning_rate "$CLASSIFY_LR"
            --devices "$DEVICES"
            --batch_size "$BATCH_SIZE"
            --encoder_type "teacher"
            --checkpoint_dir "$CLASS_CKPT_DIR"
            --log_dir "$CLASS_LOG_DIR"
            --seed "$SEED"
            --logger "$LOGGER"
        )

        [[ "$CLASSIFY_MODE" == "lineareval" ]] && CLASSIFY_CMD+=(--freeze_backbone)

        CUDA_VISIBLE_DEVICES="$GPUS" "${CLASSIFY_CMD[@]}"

        log "Classification complete: ${ABLATION} ${CLASSIFY_MODE}_seed_${SEED}"
        log "Results saved to: $CLASS_LOG_DIR/test_results.json"
    done
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
separator
log "ABLATION SWEEP COMPLETE"
separator
echo "Results location: $OUTPUT_DIR/"
echo ""
echo "Structure (unified with sweep):"
echo "  $OUTPUT_DIR/"
echo "  ├── pretraining/$DATASET/"
for ABLATION in "${ABLATION_ARRAY[@]}"; do
echo "  │   └── ${ABLATION}/checkpoints/"
done
echo "  └── classification/$DATASET/"
for ABLATION in "${ABLATION_ARRAY[@]}"; do
echo "      └── ${ABLATION}/"
    for SEED in "${SEED_ARRAY[@]}"; do
echo "          └── ${CLASSIFY_MODE}_seed_${SEED}/logs/test_results.json"
    done
done
separator
