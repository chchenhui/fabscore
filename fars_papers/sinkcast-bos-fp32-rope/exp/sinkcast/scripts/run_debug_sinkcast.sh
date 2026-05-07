#!/bin/bash
# Debug sanity check: run SinkCast with K=1 and K=4 on Llama with --debug flag
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== Debug: SinkCast K=1 (fixed BF16 logit) ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model llama-3.1-8b \
    --K 1 \
    --debug \
    --output-dir "$PROJECT_ROOT/sinkcast/results/debug/sinkcast_k1"

echo ""
echo "=== Debug: SinkCast K=4 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model llama-3.1-8b \
    --K 4 \
    --debug \
    --output-dir "$PROJECT_ROOT/sinkcast/results/debug/sinkcast_k4"

echo ""
echo "=== Debug: SinkCast K=8 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model llama-3.1-8b \
    --K 8 \
    --debug \
    --output-dir "$PROJECT_ROOT/sinkcast/results/debug/sinkcast_k8"

echo ""
echo "=== Debug: SinkCast K=16 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model llama-3.1-8b \
    --K 16 \
    --debug \
    --output-dir "$PROJECT_ROOT/sinkcast/results/debug/sinkcast_k16"

echo ""
echo "=== Debug: SinkCast K=64 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model llama-3.1-8b \
    --K 64 \
    --debug \
    --output-dir "$PROJECT_ROOT/sinkcast/results/debug/sinkcast_k64"

echo "Debug complete."
