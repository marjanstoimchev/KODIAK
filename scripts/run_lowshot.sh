#!/bin/bash
# =============================================================================
# K-Shot Evaluation Runner
# =============================================================================
# Runs eval_lowshot.py across all combinations of:
#   methods × datasets × k-shots × label seeds × training seeds
#
# Protocol:
#   3 label subsets (label_seeds) × 3 training seeds = 9 runs per setting.
#   Each class gets exactly k labelled examples.
#
# Usage:
#   # All methods, all datasets, both FT and LP
#   ./scripts/run_lowshot.sh --gpus 0
#
#   # Single method + dataset (quick test)
#   ./scripts/run_lowshot.sh --gpus 0 --methods kodiak --datasets dtd --mode finetune
#
#   # Continued pretraining checkpoints
#   ./scripts/run_lowshot.sh --gpus 0 --init-mode continued
#
#   # Custom shots
#   ./scripts/run_lowshot.sh --gpus 0 --shots "1 5 20"
# =============================================================================

set -e

# ─────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────
GPUS=""
METHODS="kodiak dinov3 mae ijepa moca"
DATASETS="eurosat dtd oxford_pets"
SHOTS="1 2 4 8 16"
LABEL_SEEDS="0 1 42"
TRAIN_SEEDS="42"
MODE=""          # empty = both FT and LP
INIT_MODE="continued"   # scratch or continued
MAX_EPOCHS=""    # empty = use low-shot default (100)
BATCH_SIZE=""
LEARNING_RATE=""
OUTPUT_DIR=""
CKPT_BASE_DIR="output_proto_analysis"  # base dir for kodiak checkpoints
PROTOTYPES="128"                        # number of prototypes (e.g. 64, 128, 256, 512, 1024, 2048, 4096)
DRY_RUN=false

# ─────────────────────────────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 --gpus GPU_ID [OPTIONS]

Required:
  --gpus GPU              GPU index (e.g. 0, or 0,1)

Optional:
  --methods "m1 m2 .."    Methods to evaluate (default: "$METHODS")
  --datasets "d1 d2 .."   Datasets (default: "$DATASETS")
  --shots "k1 k2 .."      Examples per class (default: "$SHOTS")
  --label-seeds "s1 .."   Label subset seeds (default: "$LABEL_SEEDS")
  --train-seeds "s1 .."   Training seeds (default: "$TRAIN_SEEDS")
  --mode MODE             finetune, lineareval, or both (default: both)
  --init-mode MODE        scratch or continued (default: $INIT_MODE)
  --max-epochs N          Override max training epochs
  --batch-size N          Override batch size
  --learning-rate LR      Override learning rate
  --output-dir DIR        Override output directory
  --ckpt-base-dir DIR     Base dir for kodiak checkpoints (default: $CKPT_BASE_DIR)
  --prototypes N          Number of prototypes (default: $PROTOTYPES)
  --dry-run               Print commands without executing
  --help                  Show this help

Examples:
  # Full sweep (continued PT checkpoints)
  $0 --gpus 0

  # Quick single-method test
  $0 --gpus 0 --methods kodiak --datasets dtd --mode finetune --shots "5"

  # From-scratch checkpoints
  $0 --gpus 0 --init-mode scratch

  # Custom prototypes and checkpoint dir
  $0 --gpus 0 --methods kodiak --datasets cifar100 --prototypes 256 --ckpt-base-dir output_proto_analysis
EOF
    exit 1
}

# ─────────────────────────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)           GPUS="$2";          shift 2 ;;
        --methods)        METHODS="$2";       shift 2 ;;
        --datasets)       DATASETS="$2";      shift 2 ;;
        --shots)          SHOTS="$2";         shift 2 ;;
        --label-seeds)    LABEL_SEEDS="$2";   shift 2 ;;
        --train-seeds)    TRAIN_SEEDS="$2";   shift 2 ;;
        --mode)           MODE="$2";          shift 2 ;;
        --init-mode)      INIT_MODE="$2";     shift 2 ;;
        --max-epochs)     MAX_EPOCHS="$2";    shift 2 ;;
        --batch-size)     BATCH_SIZE="$2";    shift 2 ;;
        --learning-rate)  LEARNING_RATE="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";    shift 2 ;;
        --ckpt-base-dir)  CKPT_BASE_DIR="$2"; shift 2 ;;
        --prototypes)     PROTOTYPES="$2";    shift 2 ;;
        --dry-run)        DRY_RUN=true;       shift   ;;
        --help|-h)        usage ;;
        *)                echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$GPUS" ]]; then
    echo "ERROR: --gpus is required"
    usage
fi

# ─────────────────────────────────────────────────────────────────────
# Paths and mappings
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Dataset → config directory
declare -A CONFIG_DIRS=(
    ["dtd"]="DTD"
    ["eurosat"]="eurosat"
    ["oxford_pets"]="oxford_pets"
    ["cifar100"]="cifar100"
    ["nctcrche100k"]="NCTCRCHE100K"
    ["imagenet1k"]="imagenet1k"
    ["pancreatic"]="pancreatic"
)

# ── Checkpoint finder ────────────────────────────────────────────────
# Maps each (method, dataset, init_mode) to its SSL pretraining checkpoint.
# These are the actual pretrained encoder weights, NOT trained classifiers.

find_ssl_checkpoint() {
    local method="$1"
    local dataset="$2"
    local ckpt=""

    case "$method" in
        kodiak)
            if [[ "$INIT_MODE" == "continued" ]]; then
                ckpt="${CKPT_BASE_DIR}/checkpoints/pretraining/${dataset}/kodiak_${dataset}_continued_proto${PROTOTYPES}_koleo0.1_cls1.0_mc/last.ckpt"
            else
                ckpt="${CKPT_BASE_DIR}/checkpoints/pretraining/${dataset}/kodiak_${dataset}_proto${PROTOTYPES}_koleo0.1_cls1.0_mc/last.ckpt"
            fi
            ;;
        dinov3)
            if [[ "$INIT_MODE" == "continued" ]]; then
                ckpt="/home/marjans/DinoV3LightningTraining/prototype_analysis_dinov3_continued/pretraining/${dataset}/proto_4096/checkpoints/last.ckpt"
            else
                # DINOv3 official pretrained weights (from-scratch baseline)
                ckpt="/home/marjans/DinoV3LightningTraining/dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            fi
            ;;
        mae)
            if [[ "$INIT_MODE" == "continued" ]]; then
                ckpt="/home/marjans/mae_lightning/output/pretraining/${dataset}/mae_${dataset}_continued/checkpoints/last.ckpt"
            else
                ckpt="/home/marjans/mae_lightning/output/pretraining/${dataset}/mae_${dataset}/checkpoints/last.ckpt"
            fi
            ;;
        ijepa)
            if [[ "$INIT_MODE" == "continued" ]]; then
                ckpt="/home/marjans/ijepa_lightning/output/pretraining/${dataset}/ijepa_${dataset}_continued/checkpoints/last.ckpt"
            else
                ckpt="/home/marjans/ijepa_lightning/output/pretraining/${dataset}/ijepa_${dataset}/checkpoints/last.ckpt"
            fi
            ;;
        moca)
            if [[ "$INIT_MODE" == "continued" ]]; then
                ckpt="/home/marjans/MOCA/output/checkpoints/pretraining/${dataset}/moca_${dataset}_continued/last.ckpt"
            else
                ckpt="/home/marjans/MOCA/output/checkpoints/pretraining/${dataset}/moca_${dataset}/last.ckpt"
            fi
            ;;
        foundation)
            # DINOv3 foundation model — no domain adaptation, same weights for all datasets
            ckpt="/home/marjans/DinoV3LightningTraining/dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            ;;
    esac

    # Verify file exists
    if [[ -n "$ckpt" && -f "$ckpt" ]]; then
        echo "$ckpt"
    else
        echo ""
    fi
}

# ─────────────────────────────────────────────────────────────────────
# Determine modes to run
# ─────────────────────────────────────────────────────────────────────
if [[ -z "$MODE" ]]; then
    MODES="finetune lineareval"
elif [[ "$MODE" == "both" ]]; then
    MODES="finetune lineareval"
else
    MODES="$MODE"
fi

# ─────────────────────────────────────────────────────────────────────
# Print plan
# ─────────────────────────────────────────────────────────────────────
separator() { echo "========================================================================"; }

separator
echo "K-SHOT EVALUATION SWEEP"
separator
echo "  GPU:            $GPUS (via CUDA_VISIBLE_DEVICES)"
echo "  Init mode:      $INIT_MODE"
echo "  Methods:        $METHODS"
echo "  Datasets:       $DATASETS"
echo "  Shots:          $SHOTS"
echo "  Label seeds:    $LABEL_SEEDS"
echo "  Train seeds:    $TRAIN_SEEDS"
echo "  Modes:          $MODES"

# Count total runs
N_METHODS=$(echo $METHODS | wc -w)
N_DATASETS=$(echo $DATASETS | wc -w)
N_SHOTS=$(echo $SHOTS | wc -w)
N_LSEEDS=$(echo $LABEL_SEEDS | wc -w)
N_TSEEDS=$(echo $TRAIN_SEEDS | wc -w)
N_MODES=$(echo $MODES | wc -w)
TOTAL=$((N_METHODS * N_DATASETS * N_SHOTS * N_LSEEDS * N_TSEEDS * N_MODES))
echo "  Total runs:     $TOTAL"

if [[ -n "$OUTPUT_DIR" ]]; then
    echo "  Output dir:     $OUTPUT_DIR"
fi
if [[ "$DRY_RUN" == true ]]; then
    echo "  *** DRY RUN — commands will be printed, not executed ***"
fi
separator
echo ""

# ─────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────
RUN_IDX=0
SKIPPED=0
FAILED=0

for method in $METHODS; do
    for dataset in $DATASETS; do

        # Resolve config
        config_dir="${CONFIG_DIRS[$dataset]}"
        if [[ -z "$config_dir" ]]; then
            echo "WARNING: Unknown dataset '$dataset', skipping"
            continue
        fi
        config="configs/${config_dir}/classify.yaml"
        if [[ ! -f "$config" ]]; then
            echo "WARNING: Config not found: $config, skipping"
            continue
        fi

        # Resolve SSL checkpoint
        ssl_ckpt=$(find_ssl_checkpoint "$method" "$dataset")
        if [[ -z "$ssl_ckpt" ]]; then
            echo "WARNING: No SSL checkpoint found for $method/$dataset ($INIT_MODE), skipping"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        for eval_mode in $MODES; do
            for k in $SHOTS; do
                for ls in $LABEL_SEEDS; do
                    for ts in $TRAIN_SEEDS; do
                        RUN_IDX=$((RUN_IDX + 1))

                        # Build output dir
                        # Structure: {ckpt_base}/lowshot_from_{init_mode}/{dataset}/proto{N}_koleo0.1_cls1/{run_tag}
                        # Matches existing folder convention (knn_from_continued, checkpoints, etc.)
                        out_base="${OUTPUT_DIR:-${CKPT_BASE_DIR}/lowshot_from_${INIT_MODE}}"
                        run_tag="${eval_mode}_${k}shot_ls${ls}_ts${ts}"
                        run_dir="${out_base}/${dataset}/proto${PROTOTYPES}_koleo0.1_cls1/${run_tag}"

                        # Skip if results already exist
                        if [[ -f "${run_dir}/test_results.json" ]]; then
                            echo "[$RUN_IDX/$TOTAL] SKIP  $method/$dataset  ${eval_mode}  ${k}-shot  ls=$ls  ts=$ts  (results exist)"
                            SKIPPED=$((SKIPPED + 1))
                            continue
                        fi

                        echo "[$RUN_IDX/$TOTAL] RUN   $method/$dataset  ${eval_mode}  ${k}-shot  ls=$ls  ts=$ts"

                        # Build command (single GPU via CUDA_VISIBLE_DEVICES)
                        # method_tag controls the subfolder under dataset/
                        # e.g. proto128_koleo0.1_cls1 → matches knn & checkpoint naming
                        method_subtag="proto${PROTOTYPES}_koleo0.1_cls1"
                        CMD=(
                            python scripts/eval_lowshot.py
                            --config "$config"
                            --pretrained_path "$ssl_ckpt"
                            --k_shot "$k"
                            --label_seed "$ls"
                            --train_seed "$ts"
                            --method_tag "$method_subtag"
                            --output_dir "$out_base"
                        )

                        if [[ "$eval_mode" == "lineareval" ]]; then
                            CMD+=(--freeze_backbone)
                        fi

                        if [[ -n "$MAX_EPOCHS" ]]; then
                            CMD+=(--max_epochs "$MAX_EPOCHS")
                        fi
                        if [[ -n "$BATCH_SIZE" ]]; then
                            CMD+=(--batch_size "$BATCH_SIZE")
                        fi
                        if [[ -n "$LEARNING_RATE" ]]; then
                            CMD+=(--learning_rate "$LEARNING_RATE")
                        fi

                        if [[ "$DRY_RUN" == true ]]; then
                            echo "  CUDA_VISIBLE_DEVICES=$GPUS ${CMD[*]}"
                        else
                            if CUDA_VISIBLE_DEVICES="$GPUS" "${CMD[@]}"; then
                                echo "  ✓ Complete"
                            else
                                echo "  ✗ FAILED"
                                FAILED=$((FAILED + 1))
                            fi
                        fi

                    done
                done
            done
        done
    done
done

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
echo ""
separator
echo "K-SHOT SWEEP COMPLETE"
separator
echo "  Total:    $TOTAL"
echo "  Ran:      $((TOTAL - SKIPPED - FAILED))"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo ""
echo "Results in: ${OUTPUT_DIR:-${CKPT_BASE_DIR}/lowshot_from_${INIT_MODE}}/"
echo "  Aggregate with:  python scripts/aggregate_lowshot.py --input_dir ${OUTPUT_DIR:-${CKPT_BASE_DIR}/lowshot_from_${INIT_MODE}}"
separator
