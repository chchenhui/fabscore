#!/bin/bash
# DualRay-TopK Repair (Condition B): repair infeasible attempt-0 instances with dual ray ranking feedback
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/farkas-dual-ray-optmodel-repair/exp
cd "$PROJ_DIR"

source .venv/bin/activate
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

python -m dualrayrank.scripts.run_dualray_topk_repair \
    --model Qwen/Qwen2.5-7B-Instruct \
    --attempt0-dir dualrayrank/outputs/attempt0 \
    --output-dir dualrayrank/outputs/repair_dualray_topk \
    --results-dir dualrayrank/results \
    --iis-results dualrayrank/results/iis_topk_results.json \
    --tensor-parallel-size 1 \
    --k 5

echo ""
echo "=== DualRay-TopK repair experiment complete ==="
