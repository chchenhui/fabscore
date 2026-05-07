#!/bin/bash
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/compute-matched-diffusion-planning-audit/exp
source .venv/bin/activate

export $(grep -v '^#' .env | xargs)
export WANDB_MODE=offline

echo "=== Temperature Sweep on Calibration Set ==="

for TASK in countdown sudoku; do
    for TEMP in 0.8 1.0 1.2; do
        echo ""
        echo ">>> Task=$TASK, temp=$TEMP"
        python audit/inference/qwen_best_of_k_optimized.py \
            --task "$TASK" \
            --seeds 42 \
            --data_split cal \
            --temperature "$TEMP" \
            --presence_penalty 0.0 \
            --output_prefix "sweep_t${TEMP}"
    done
done

echo ""
echo "=== Also test presence_penalty=0.3 with best temp candidates ==="
for TASK in countdown sudoku; do
    for TEMP in 1.0 1.2; do
        echo ""
        echo ">>> Task=$TASK, temp=$TEMP, presence_penalty=0.3"
        python audit/inference/qwen_best_of_k_optimized.py \
            --task "$TASK" \
            --seeds 42 \
            --data_split cal \
            --temperature "$TEMP" \
            --presence_penalty 0.3 \
            --output_prefix "sweep_t${TEMP}_pp0.3"
    done
done

echo ""
echo "=== Temperature sweep completed ==="
