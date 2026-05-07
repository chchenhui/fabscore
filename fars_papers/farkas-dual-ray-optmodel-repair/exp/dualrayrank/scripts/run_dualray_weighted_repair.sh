#!/bin/bash
# Optimized DualRay-TopK+Weights Repair (Condition C): multi-attempt iterative
# repair with normalized Farkas multiplier weights and enhanced prompts
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/farkas-dual-ray-optmodel-repair/exp
cd "$PROJ_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

# Load env vars for wandb
set -a
source .env
set +a
export WANDB_MODE=offline

python -m dualrayrank.scripts.run_dualray_weighted_repair \
    --model Qwen/Qwen2.5-7B-Instruct \
    --attempt0-dir dualrayrank/outputs/attempt0 \
    --output-dir dualrayrank/outputs/repair_dualray_weighted \
    --results-dir dualrayrank/results \
    --iis-results dualrayrank/results/iis_topk_results.json \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --k 10 \
    --max-attempts 3

echo ""
echo "=== DualRay-TopK+Weights optimized repair experiment complete ==="
