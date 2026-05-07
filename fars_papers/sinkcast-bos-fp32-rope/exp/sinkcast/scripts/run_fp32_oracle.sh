#!/bin/bash
# Full FP32 oracle shift microbenchmark for both models.
# Runs Llama-3.1-8B then Mistral-7B-v0.3, seq_lengths 512/1024/2048.
set -e
PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sinkcast-bos-fp32-rope/exp"
source "$PROJECT_ROOT/.venv/bin/activate"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== Llama-3.1-8B FP32 Oracle ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/fp32_oracle_microbench.py" \
    --model llama-3.1-8b \
    --output-dir "$PROJECT_ROOT/sinkcast/results/microbench/fp32_oracle"

echo "=== Mistral-7B-v0.3 FP32 Oracle ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/fp32_oracle_microbench.py" \
    --model mistral-7b-v0.3 \
    --output-dir "$PROJECT_ROOT/sinkcast/results/microbench/fp32_oracle"

echo "=== All FP32 Oracle runs complete ==="
