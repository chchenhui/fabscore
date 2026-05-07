#!/bin/bash
# Attempt-0: Generate LPs with Qwen2.5-7B-Instruct and evaluate on MAMO benchmark
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/farkas-dual-ray-optmodel-repair/exp
cd "$PROJ_DIR"

source .venv/bin/activate
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

echo "=== Step 1: Generation ==="
python -m dualrayrank.scripts.run_attempt0 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output-dir dualrayrank/outputs/attempt0 \
    --tensor-parallel-size 1

echo ""
echo "=== Step 2: Evaluation ==="
python -m dualrayrank.evaluation.evaluate \
    --attempt-dir dualrayrank/outputs/attempt0 \
    --results-dir dualrayrank/results

echo ""
echo "=== Done ==="
