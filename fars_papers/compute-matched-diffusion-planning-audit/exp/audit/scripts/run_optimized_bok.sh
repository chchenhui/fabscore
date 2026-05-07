#!/bin/bash
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/compute-matched-diffusion-planning-audit/exp
source .venv/bin/activate

export $(grep -v '^#' .env | xargs)
export WANDB_MODE=offline

echo "=== Optimized Best-of-k Run (temp=1.2, stop=\\n\\n for both tasks) ==="

for TASK in countdown sudoku; do
    echo ""
    echo ">>> Running Task=$TASK with temp=1.2, 3 seeds"
    python audit/inference/qwen_best_of_k_optimized.py \
        --task "$TASK" \
        --seeds 42 123 456 \
        --data_split test \
        --temperature 1.2 \
        --presence_penalty 0.0 \
        --output_prefix "qwen_bok_opt"
done

echo ""
echo "=== Optimized best-of-k completed ==="
