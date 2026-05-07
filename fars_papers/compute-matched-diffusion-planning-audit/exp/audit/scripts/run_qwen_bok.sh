#!/bin/bash
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/compute-matched-diffusion-planning-audit/exp
source .venv/bin/activate

export $(grep -v '^#' .env | xargs)

TASK=${1:-countdown}
LIMIT=${2:-}

EXTRA_ARGS=""
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS="--limit $LIMIT"
fi

python audit/inference/qwen_best_of_k.py --task "$TASK" --seeds 42 123 456 $EXTRA_ARGS

echo "=== Best-of-k inference for $TASK completed ==="
