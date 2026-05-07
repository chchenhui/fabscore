#!/bin/bash
# Full BF16 FlashAttention shift microbenchmark for Llama-3.1-8B.

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
    --model llama-3.1-8b \
    --output-dir "$PROJECT_ROOT/sinkcast/results/microbench/bf16_flash"
