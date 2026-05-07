#!/bin/bash
# Debug run: attempt-0 with 10 instances only
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/farkas-dual-ray-optmodel-repair/exp
cd "$PROJ_DIR"

source .venv/bin/activate
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

echo "=== Debug: Generation (10 instances) ==="
python -m dualrayrank.scripts.run_attempt0 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output-dir dualrayrank/outputs/attempt0_debug \
    --tensor-parallel-size 1 \
    --limit 10

echo ""
echo "=== Debug: Checking outputs ==="
ls -la dualrayrank/outputs/attempt0_debug/ | head -20
echo ""
echo "=== First .lp file ==="
head -30 dualrayrank/outputs/attempt0_debug/*.lp 2>/dev/null | head -50

echo ""
echo "=== Done ==="
