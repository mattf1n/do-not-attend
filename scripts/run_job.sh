#!/bin/bash
#SBATCH --job-name=do-not-attend
#SBATCH --partition=oneweek
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --account=swabhas_1625
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
``
# ── Edit these to control the run ────────────────────────────────────────────
TOKENS=30000
MAX_SUBTOKENS=2
COMPONENTS="all"   # "all", "Pile-CC", or comma-separated e.g. "Wikipedia (en),HackerNews"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT=/home1/calebtal/projects/do-not-attend
cd "$PROJECT"

mkdir -p logs

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

source "$PROJECT/.venv/bin/activate"

python main.py \
    --batch \
    --tokens "$TOKENS" \
    --max-subtokens "$MAX_SUBTOKENS" \
    --components "$COMPONENTS"

echo "=== Job finished: $(date) ==="
