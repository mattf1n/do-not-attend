#!/bin/bash
#SBATCH --job-name=do-not-attend-qkv
#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=677G
#SBATCH --time=7-00:00:00
#SBATCH --account=swabhas_1625
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Defaults (override from the terminal — see usage below)
TOKENS=16000
MAX_SUBTOKENS=2
COMPONENTS="all"   # "all", or comma-separated e.g. "Wikipedia (en),PubMed Abstracts"
OVERWRITE=false

# Usage:
#   sbatch scripts/run_qkv_cache.sh --tokens 16000
#   sbatch scripts/run_qkv_cache.sh --tokens 26000 --components all
#   sbatch scripts/run_qkv_cache.sh --tokens 500 --components "PubMed Abstracts" --overwrite
#
# Output: output/qkv_cache/{TOKENS}_tokens/{component}_{TOKENS}tokens/{q0..v1}.pt
# Runs on CPU (no GPU). Filtered Q/K/V cache only; still slower than GPU.

set -euo pipefail
 
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokens)
            TOKENS="$2"
            shift 2
            ;;
        --max-subtokens)
            MAX_SUBTOKENS="$2"
            shift 2
            ;;
        --components)
            COMPONENTS="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        -h|--help)
            echo "Usage: sbatch $0 --tokens N [--components NAME|all] [--max-subtokens N] [--overwrite]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: sbatch $0 --tokens N [--components NAME|all] [--max-subtokens N] [--overwrite]" >&2
            exit 1
            ;;
    esac
done

PROJECT=/home1/calebtal/projects/do-not-attend
cd "$PROJECT"

mkdir -p logs

# Force CPU even if the node has idle GPUs visible.
export CUDA_VISIBLE_DEVICES=""

echo "=== Job started: $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}"
echo "TOKENS=$TOKENS  COMPONENTS=$COMPONENTS  MAX_SUBTOKENS=$MAX_SUBTOKENS  OVERWRITE=$OVERWRITE"
echo "Device: CPU (CUDA_VISIBLE_DEVICES empty)"

source "$PROJECT/.venv/bin/activate"

OVERWRITE_FLAG=""
if [ "$OVERWRITE" = "true" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

uv run main.py \
    --batch \
    --mode qkv \
    --tokens "$TOKENS" \
    --max-subtokens "$MAX_SUBTOKENS" \
    --components "$COMPONENTS" \
    $OVERWRITE_FLAG

echo "=== Job finished: $(date) ==="
