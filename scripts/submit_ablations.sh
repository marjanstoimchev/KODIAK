#!/bin/bash
# Submit all ablation jobs to SLURM (one per configuration)
#
# Usage:
#   ./scripts/submit_ablations.sh
#   ./scripts/submit_ablations.sh --dataset eurosat --pretrain-epochs 200
#   ./scripts/submit_ablations.sh --ablations "full no_koleo" --batch-size 64

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p logs

ABLATIONS="full no_sinkhorn no_cls_loss no_koleo"
EXPORT_VARS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)          EXPORT_VARS+=(DATASET="$2");          shift 2 ;;
        --ablations)        ABLATIONS="$2";                       shift 2 ;;
        --seeds)            EXPORT_VARS+=(SEEDS="$2");            shift 2 ;;
        --pretrain-epochs)  EXPORT_VARS+=(PRETRAIN_EPOCHS="$2");  shift 2 ;;
        --pretrain-lr)      EXPORT_VARS+=(PRETRAIN_LR="$2");      shift 2 ;;
        --classify-epochs)  EXPORT_VARS+=(CLASSIFY_EPOCHS="$2");  shift 2 ;;
        --classify-lr)      EXPORT_VARS+=(CLASSIFY_LR="$2");      shift 2 ;;
        --classify-mode)    EXPORT_VARS+=(CLASSIFY_MODE="$2");    shift 2 ;;
        --batch-size)       EXPORT_VARS+=(BATCH_SIZE="$2");       shift 2 ;;
        --num-prototypes)   EXPORT_VARS+=(NUM_PROTOTYPES="$2");   shift 2 ;;
        --koleo-weight)     EXPORT_VARS+=(KOLEO_WEIGHT="$2");     shift 2 ;;
        --cls-weight)       EXPORT_VARS+=(CLS_WEIGHT="$2");       shift 2 ;;
        --init-mode)        EXPORT_VARS+=(INIT_MODE="$2");        shift 2 ;;
        --output-dir)       EXPORT_VARS+=(OUTPUT_DIR="$2");       shift 2 ;;
        --sif)              EXPORT_VARS+=(USE_SINGULARITY=true SIF_IMAGE="$2"); shift 2 ;;
        --help|-h)
            sed -n '2,6p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

for ab in $ABLATIONS; do
    env ABLATION="$ab" "${EXPORT_VARS[@]}" sbatch --job-name="${ab}" scripts/slurm/run_ablation.sh
    echo "Submitted: $ab"
done
