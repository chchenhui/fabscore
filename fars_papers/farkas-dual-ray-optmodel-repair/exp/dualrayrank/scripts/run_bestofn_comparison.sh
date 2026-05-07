#!/bin/bash
# Run both DualRay+Weights and IIS best-of-N experiments for fair comparison
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/farkas-dual-ray-optmodel-repair/exp
cd "$PROJ_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

set -a
source .env
set +a
export WANDB_MODE=offline

echo "=========================================="
echo "Part 1: DualRay-TopK+Weights Best-of-N"
echo "=========================================="
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
echo "=========================================="
echo "Part 2: IIS-TopK Best-of-N (control)"
echo "=========================================="
python -m dualrayrank.scripts.run_iis_bestofn_repair \
    --model Qwen/Qwen2.5-7B-Instruct \
    --attempt0-dir dualrayrank/outputs/attempt0 \
    --output-dir dualrayrank/outputs/repair_iis_bestofn \
    --results-dir dualrayrank/results \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --k 10 \
    --n 5 \
    --temperature 0.7 \
    --seed 42

echo ""
echo "=== Both best-of-N experiments complete ==="
