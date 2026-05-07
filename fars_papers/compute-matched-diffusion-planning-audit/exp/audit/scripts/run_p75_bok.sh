#!/bin/bash
# Run best-of-k with p75-derived k values for sensitivity analysis.
# Countdown: k_p75=34 (vs k_median=35). Sudoku: k_p75=39 = k_median, skip.
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/compute-matched-diffusion-planning-audit/exp
source .venv/bin/activate

export $(grep -v '^#' .env | xargs)
export WANDB_MODE=offline

echo "=== p75 Sensitivity: Best-of-k with k_p75 ==="

echo ">>> Countdown k_p75=34 (3 seeds)"
python audit/inference/qwen_best_of_k_optimized.py \
    --task countdown \
    --seeds 42 123 456 \
    --data_split test \
    --temperature 1.2 \
    --presence_penalty 0.0 \
    --k_override 34 \
    --output_prefix "qwen_bok_p75"

echo ""
echo "=== p75 best-of-k completed ==="
