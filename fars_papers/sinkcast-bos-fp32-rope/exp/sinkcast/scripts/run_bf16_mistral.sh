#!/bin/bash
# Full BF16 FlashAttention shift microbenchmark for Mistral-7B-v0.3.

set -e

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sinkcast-bos-fp32-rope/exp"
source "$PROJECT_ROOT/.venv/bin/activate"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

python "$PROJECT_ROOT/sinkcast/benchmarks/shift_microbench.py" \
    --model mistral-7b-v0.3 \
    --output-dir "$PROJECT_ROOT/sinkcast/results/microbench/bf16_flash"
