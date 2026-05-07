#!/bin/bash
# Full downstream position-shift evaluation with SinkCast K=1 correction.
# Usage: bash run_downstream_sinkcast.sh <model-alias> [K]
# e.g.:  bash run_downstream_sinkcast.sh llama-3.1-8b 1
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"
set -a
source "$PROJECT_ROOT/.env"
set +a
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

MODEL=${1:?"Usage: $0 <model-alias> [K]"}
K=${2:-1}
OUTPUT_DIR="$PROJECT_ROOT/sinkcast/results/downstream/sinkcast_k${K}"

echo "========================================="
echo "SinkCast K=$K downstream eval: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "========================================="

echo ""
echo "=== RULER Position-Shift Evaluation (SinkCast K=$K) ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/ruler_shift_sinkcast.py" \
    --model "$MODEL" \
    --K "$K" \
    --seq_lengths 4096 8192 \
    --shift_M 4096 \
    --n_samples 50 \
    --max_new_tokens 128 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== LongBench Position-Shift Evaluation (SinkCast K=$K) ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/longbench_shift_sinkcast.py" \
    --model "$MODEL" \
    --K "$K" \
    --shift_M 4096 \
    --max_context 8192 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== All SinkCast downstream evaluations complete for $MODEL ==="
