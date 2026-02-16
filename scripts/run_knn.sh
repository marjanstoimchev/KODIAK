#!/bin/bash
# =============================================================================
# KODIAK k-NN Evaluation Script
# =============================================================================
# Runs k-NN evaluation on pretrained SSL checkpoints.
# Supports both scratch and continued pretraining modes.
# Can evaluate multiple prototype configurations in one go.
#
# Usage:
#   # Single prototype count
#   ./scripts/run_knn.sh --dataset dtd --gpus 0 --num-prototypes 4096
#
#   # Multiple prototype counts
#   ./scripts/run_knn.sh --dataset dtd --gpus 0 --num-prototypes "256 1024 4096"
#
#   # Continued pretraining
#   ./scripts/run_knn.sh --dataset dtd --gpus 0 --init-mode continued --num-prototypes "256 4096"
#
#   # Custom checkpoint (bypasses auto-path construction)
#   ./scripts/run_knn.sh --dataset dtd --gpus 0 --pretrained-path /path/to/checkpoint.ckpt
#
# Output structure:
#   output/knn/{dataset}/{pretrain_folder}/knn_results.json
#   output/knn_from_continued/{dataset}/{pretrain_folder}/knn_results.json
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --dataset DATASET            Dataset name (dtd, eurosat, oxford_pets, nctcrche100k, imagenet1k, cifar100)"
    echo "  --gpus GPU                   GPU index (single GPU, e.g., 0)"
    echo ""
    echo "Optional:"
    echo "  --init-mode MODE             scratch or continued (default: scratch)"
    echo "  --pretrained-path PATH       Override checkpoint path (skips auto-path, single run)"
    echo "  --checkpoint-type TYPE       last or best (default: last)"
    echo "  --k K                        Number of nearest neighbors (default: 20)"
    echo "  --temperature T              Softmax temperature (default: 0.07)"
    echo "  --batch-size N               Batch size for feature extraction (default: 256)"
    echo "  --num-prototypes \"N1 N2 ..\"  Prototype counts, space-separated (default: 4096)"
    echo "  --koleo-weight W             KoLeo weight (default: 0.1)"
    echo "  --cls-weight W               CLS weight (default: 1.0)"
    echo "  --multi-crop                 Multi-crop flag (default: on)"
    echo "  --no-multi-crop              Disable multi-crop flag"
    echo "  --concat-cls-patch           Concatenate CLS + mean patch tokens for k-NN features (2x embed_dim)"
    echo "  --output-base-dir DIR        Base output directory (default: output)"
    echo "  --seeds \"S1 S2 ..\"           Evaluation seeds (default: \"0 1 42\")"
    echo "  --help                       Show this help"
    echo ""
    echo "Examples:"
    echo "  # Evaluate single config"
    echo "  $0 --dataset dtd --gpus 0"
    echo ""
    echo "  # Evaluate multiple prototype counts"
    echo "  $0 --dataset dtd --gpus 0 --num-prototypes \"256 1024 4096\""
    echo ""
    echo "  # Continued pretraining mode"
    echo "  $0 --dataset dtd --gpus 0 --init-mode continued --num-prototypes \"256 4096\""
    echo ""
    echo "  # Direct checkpoint path"
    echo "  $0 --dataset dtd --gpus 0 --pretrained-path /path/to/checkpoint.ckpt"
    exit 1
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
DATASET=""
GPUS=""
INIT_MODE="scratch"
PRETRAINED_PATH=""
CHECKPOINT_TYPE="last"
K="20"
TEMPERATURE="0.07"
BATCH_SIZE="256"
NUM_PROTOTYPES="4096"
KOLEO_WEIGHT="0.1"
CLS_WEIGHT="1.0"
MULTI_CROP=true
CONCAT_CLS_PATCH=false
OUTPUT_BASE_DIR="output"
SEEDS="0 1 42"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)         DATASET="$2";           shift 2 ;;
        --gpus)            GPUS="$2";              shift 2 ;;
        --init-mode)       INIT_MODE="$2";         shift 2 ;;
        --pretrained-path) PRETRAINED_PATH="$2";   shift 2 ;;
        --checkpoint-type) CHECKPOINT_TYPE="$2";   shift 2 ;;
        --k)               K="$2";                 shift 2 ;;
        --temperature)     TEMPERATURE="$2";       shift 2 ;;
        --batch-size)      BATCH_SIZE="$2";        shift 2 ;;
        --num-prototypes)  NUM_PROTOTYPES="$2";    shift 2 ;;
        --koleo-weight)    KOLEO_WEIGHT="$2";      shift 2 ;;
        --cls-weight)      CLS_WEIGHT="$2";        shift 2 ;;
        --multi-crop)      MULTI_CROP=true;        shift ;;
        --no-multi-crop)   MULTI_CROP=false;       shift ;;
        --concat-cls-patch) CONCAT_CLS_PATCH=true; shift ;;
        --output-base-dir) OUTPUT_BASE_DIR="$2";   shift 2 ;;
        --seeds)           SEEDS="$2";             shift 2 ;;
        --help|-h)         usage ;;
        *)                 echo "Unknown option: $1"; usage ;;
    esac
done

# Validate
if [[ -z "$DATASET" ]]; then
    echo "ERROR: --dataset is required"
    usage
fi

if [[ -z "$GPUS" ]]; then
    echo "ERROR: --gpus is required"
    usage
fi

if [[ "$INIT_MODE" != "scratch" && "$INIT_MODE" != "continued" ]]; then
    echo "ERROR: --init-mode must be 'scratch' or 'continued'"
    exit 1
fi

# -----------------------------------------------------------------------------
# Paths and configs
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

declare -A CONFIG_DIRS=(
    ["dtd"]="DTD"
    ["eurosat"]="eurosat"
    ["oxford_pets"]="oxford_pets"
    ["nctcrche100k"]="NCTCRCHE100K"
    ["imagenet1k"]="imagenet1k"
    ["cifar100"]="cifar100"
    ["tissue"]="tissue"
)

CONFIG_DIR="${CONFIG_DIRS[$DATASET]}"
if [[ -z "$CONFIG_DIR" ]]; then
    echo "ERROR: Unknown dataset: $DATASET"
    echo "Available: ${!CONFIG_DIRS[@]}"
    exit 1
fi

CLASSIFY_CONFIG="configs/${CONFIG_DIR}/classify.yaml"
if [[ ! -f "$CLASSIFY_CONFIG" ]]; then
    echo "ERROR: Config not found: $CLASSIFY_CONFIG"
    exit 1
fi

if [[ "$INIT_MODE" == "continued" ]]; then
    EXPERIMENT_SUFFIX="_continued"
else
    EXPERIMENT_SUFFIX=""
fi

MC_SUFFIX=""
if [[ "$MULTI_CROP" == true ]]; then
    MC_SUFFIX="_mc"
fi

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# -----------------------------------------------------------------------------
# Run k-NN evaluation
# -----------------------------------------------------------------------------

# If a direct checkpoint path is given, run once and exit
if [[ -n "$PRETRAINED_PATH" ]]; then
    echo "========================================================================"
    echo "k-NN EVALUATION"
    echo "========================================================================"
    echo "Dataset:         $DATASET"
    echo "Init mode:       $INIT_MODE"
    echo "GPU:             $GPUS"
    echo "Checkpoint:      $PRETRAINED_PATH"
    echo "k:               $K"
    echo "Temperature:     $TEMPERATURE"
    echo "Concat CLS+Patch: $CONCAT_CLS_PATCH"
    echo "Seeds:           $SEEDS"
    echo "========================================================================"
    echo ""

    KNN_CMD=(
        python scripts/eval_knn.py
        --config "$CLASSIFY_CONFIG"
        --pretrained_path "$PRETRAINED_PATH"
        --checkpoint_type "$CHECKPOINT_TYPE"
        --init_mode "$INIT_MODE"
        --k "$K"
        --temperature "$TEMPERATURE"
        --batch_size "$BATCH_SIZE"
        --devices "$GPUS"
        --output_base_dir "$OUTPUT_BASE_DIR"
        --seeds "$SEEDS"
    )
    if [[ "$CONCAT_CLS_PATCH" == true ]]; then
        KNN_CMD+=(--concat_cls_patch)
    fi

    CUDA_VISIBLE_DEVICES="$GPUS" "${KNN_CMD[@]}"

    echo ""
    log "k-NN EVALUATION COMPLETE"
    exit 0
fi

# Otherwise, loop over prototype counts
read -ra PROTO_ARRAY <<< "$NUM_PROTOTYPES"

echo "========================================================================"
echo "k-NN EVALUATION"
echo "========================================================================"
echo "Dataset:         $DATASET"
echo "Init mode:       $INIT_MODE"
echo "GPU:             $GPUS"
echo "Prototypes:      ${PROTO_ARRAY[*]}"
echo "KoLeo weight:    $KOLEO_WEIGHT"
echo "CLS weight:      $CLS_WEIGHT"
echo "Multi-crop:      $MULTI_CROP"
echo "k:               $K"
echo "Temperature:     $TEMPERATURE"
echo "Concat CLS+Patch: $CONCAT_CLS_PATCH"
echo "Seeds:           $SEEDS"
echo "========================================================================"
echo ""

for NPROTO in "${PROTO_ARRAY[@]}"; do
    EXP_NAME="kodiak_${DATASET}${EXPERIMENT_SUFFIX}_proto${NPROTO}_koleo${KOLEO_WEIGHT}_cls${CLS_WEIGHT}${MC_SUFFIX}"
    CKPT_DIR="$OUTPUT_BASE_DIR/checkpoints/pretraining/$DATASET/$EXP_NAME"

    if [[ ! -d "$CKPT_DIR" && ! -f "$CKPT_DIR" ]]; then
        log "SKIP proto=${NPROTO}: no checkpoint at $CKPT_DIR"
        continue
    fi

    log "Running k-NN for proto=${NPROTO} ..."
    log "  Checkpoint dir: $CKPT_DIR"

    KNN_CMD=(
        python scripts/eval_knn.py
        --config "$CLASSIFY_CONFIG"
        --pretrained_path "$CKPT_DIR"
        --checkpoint_type "$CHECKPOINT_TYPE"
        --init_mode "$INIT_MODE"
        --k "$K"
        --temperature "$TEMPERATURE"
        --batch_size "$BATCH_SIZE"
        --devices "$GPUS"
        --output_base_dir "$OUTPUT_BASE_DIR"
        --seeds "$SEEDS"
    )
    if [[ "$CONCAT_CLS_PATCH" == true ]]; then
        KNN_CMD+=(--concat_cls_patch)
    fi

    CUDA_VISIBLE_DEVICES="$GPUS" "${KNN_CMD[@]}"

    log "Done proto=${NPROTO}"
    echo ""
done

echo "========================================================================"
log "ALL k-NN EVALUATIONS COMPLETE"
echo "========================================================================"
