#!/bin/bash
# Best-of-N DualRay-TopK+Weights Repair (Condition C+): generate N diverse repair
# candidates per instance and pick the best one
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/farkas-dual-ray-optmodel-repair/exp
cd "$PROJ_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

set -a
source .env
set +a
export WANDB_MODE=offline

python -m dualrayrank.scripts.run_dualray_weighted_bestofn \
    --model Qwen/Qwen2.5-7B-Instruct \
    --attempt0-dir dualrayrank/outputs/attempt0 \
    --output-dir dualrayrank/outputs/repair_dualray_weighted_bestofn \
    --results-dir dualrayrank/results \
    --iis-results dualrayrank/results/iis_topk_results.json \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --k 10 \
    --n 5 \
    --temperature 0.7 \
    --seed 42

echo ""
echo "=== DualRay-TopK+Weights best-of-N repair experiment complete ==="
