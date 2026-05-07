#!/bin/bash
# Best-of-2 inference scaling control: 2 stochastic samples per instance, 3 seeds
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/farkas-dual-ray-optmodel-repair/exp
cd "$PROJ_DIR"

source .venv/bin/activate
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

python -m dualrayrank.scripts.run_best_of_2 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output-dir dualrayrank/outputs/best_of_2 \
    --results-dir dualrayrank/results \
    --tensor-parallel-size 1

echo ""
echo "=== Best-of-2 experiment complete ==="
