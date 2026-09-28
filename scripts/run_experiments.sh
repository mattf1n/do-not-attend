#!/bin/bash
#SBATCH --job-name=do-not-attend-exps
#SBATCH --partition=nlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=swabhas_1625
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Defaults (override from the terminal — see usage below)
FOLDER=output/16000_tokens

# Usage:
#   sbatch scripts/run_experiments.sh
#   sbatch scripts/run_experiments.sh --folder output/500_tokens/
#
# Runs experiments 4–7 (hypothesis rate, Michelson contrast, and pooled variants)
# over all component JSONs in FOLDER, once per word-category filter (not just "all").
# Output figures under figures/.

set -euo pipefail

while [[ $# -gt 0 ]]; do
    case "$1" in
        --folder)
            FOLDER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: sbatch $0 [--folder PATH]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: sbatch $0 [--folder PATH]" >&2
            exit 1
            ;;
    esac
done

PROJECT=/home1/calebtal/projects/do-not-attend
cd "$PROJECT"

mkdir -p logs

echo "=== Job started: $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}"
echo "FOLDER=$FOLDER"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [[ ! -d "$FOLDER" ]]; then
    echo "ERROR: folder not found: $FOLDER" >&2
    exit 1
fi

source "$PROJECT/.venv/bin/activate"

uv run run_experiments.py --folder "$FOLDER" --exp 4 5 6 7 --all-filters

echo "=== Job finished: $(date) ==="
