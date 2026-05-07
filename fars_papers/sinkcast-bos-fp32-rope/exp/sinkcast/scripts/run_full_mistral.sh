#!/bin/bash
# Full SinkCast benchmark for Mistral-7B-v0.3 with K=1, K=4, K=8, K=16
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

OUTDIR="$PROJECT_ROOT/sinkcast/results/microbench_opt"

echo "=== Mistral-7B-v0.3: SinkCast K=1 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model mistral-7b-v0.3 --K 1 \
    --output-dir "$OUTDIR/sinkcast_k1"

echo ""
echo "=== Mistral-7B-v0.3: SinkCast K=4 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model mistral-7b-v0.3 --K 4 \
    --output-dir "$OUTDIR/sinkcast_k4"

echo ""
echo "=== Mistral-7B-v0.3: SinkCast K=8 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model mistral-7b-v0.3 --K 8 \
    --output-dir "$OUTDIR/sinkcast_k8"

echo ""
echo "=== Mistral-7B-v0.3: SinkCast K=16 ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model mistral-7b-v0.3 --K 16 \
    --output-dir "$OUTDIR/sinkcast_k16"

echo "Mistral runs complete."
