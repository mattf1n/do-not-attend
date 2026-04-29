#!/bin/bash
#SBATCH --job-name=do-not-attend
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=185G
#SBATCH --time=2-00:00:00
#SBATCH --account=swabhas_1625
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Edit these to control the run ────────────────────────────────────────────
TOKENS=30000
MAX_SUBTOKENS=2
COMPONENTS="all"   # "all", "Pile-CC", or comma-separated e.g. "Wikipedia (en),HackerNews"
OVERWRITE=false    # set to true to overwrite an existing output folder; false creates a duplicate e.g. 30000_tokens(1)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT=/home1/calebtal/projects/do-not-attend
cd "$PROJECT"

mkdir -p logs

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

source "$PROJECT/.venv/bin/activate"

OVERWRITE_FLAG=""
if [ "$OVERWRITE" = "true" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

uv run main.py \
    --batch \
    --tokens "$TOKENS" \
    --max-subtokens "$MAX_SUBTOKENS" \
    --components "$COMPONENTS" \
    $OVERWRITE_FLAG

echo "=== Job finished: $(date) ==="
