#!/bin/bash
# IIS-TopK Repair (Condition A): repair infeasible attempt-0 instances with IIS feedback
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/farkas-dual-ray-optmodel-repair/exp
cd "$PROJ_DIR"

source .venv/bin/activate
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

python -m dualrayrank.scripts.run_iis_topk_repair \
    --model Qwen/Qwen2.5-7B-Instruct \
    --attempt0-dir dualrayrank/outputs/attempt0 \
    --output-dir dualrayrank/outputs/repair_iis_topk \
    --results-dir dualrayrank/results \
    --tensor-parallel-size 1 \
    --k 5

echo ""
echo "=== IIS-TopK repair experiment complete ==="
