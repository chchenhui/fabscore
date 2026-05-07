#!/bin/bash
# Full SinkCast benchmark for Llama-3.1-8B with K=1, K=4, K=8, K=16
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

OUTDIR="$PROJECT_ROOT/sinkcast/results/microbench_opt"

echo "=== Llama-3.1-8B: SinkCast K=1 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model llama-3.1-8b --K 1 \
    --output-dir "$OUTDIR/sinkcast_k1"

echo ""
echo "=== Llama-3.1-8B: SinkCast K=4 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model llama-3.1-8b --K 4 \
    --output-dir "$OUTDIR/sinkcast_k4"

echo ""
echo "=== Llama-3.1-8B: SinkCast K=8 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model llama-3.1-8b --K 8 \
    --output-dir "$OUTDIR/sinkcast_k8"

echo ""
echo "=== Llama-3.1-8B: SinkCast K=16 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model llama-3.1-8b --K 16 \
    --output-dir "$OUTDIR/sinkcast_k16"

echo "Llama runs complete."
