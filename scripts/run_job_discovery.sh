#!/bin/bash
#SBATCH --job-name=do-not-attend
#SBATCH --partition=largemem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1498G
#SBATCH --time=5-00:00:00
#SBATCH --account=swabhas_1625
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Edit these to control the run ────────────────────────────────────────────
TOKENS=26000
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
  test_tokens = tl_model.to_tokens("Hello world")
  _, cache = tl_model.run_with_cache(test_tokens)
  print("Cache keys (first 10):", list(cache.keys())[:10])
  # specifically check for q and k hooks
  has_q = any("hook_q" in k for k in cache.keys())
  has_k = any("hook_k" in k for k in cache.keys())
  print("hook_q available:", has_q)
  print("hook_k available:", has_k)