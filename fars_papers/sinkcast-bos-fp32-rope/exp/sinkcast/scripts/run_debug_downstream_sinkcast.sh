#!/bin/bash
# Debug dry-run: 2 samples per task, single seq_length, RULER only with SinkCast K=1.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"
set -a
source "$PROJECT_ROOT/.env"
set +a
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

MODEL=${1:-"llama-3.1-8b"}
K=${2:-1}
OUTPUT_DIR="$PROJECT_ROOT/sinkcast/results/downstream/sinkcast_k${K}_debug"

echo "========================================="
echo "DEBUG: SinkCast K=$K downstream on $MODEL"
echo "Output: $OUTPUT_DIR"
echo "========================================="

python "$PROJECT_ROOT/sinkcast/benchmarks/ruler_shift_sinkcast.py" \
    --model "$MODEL" \
    --K "$K" \
    --seq_lengths 4096 \
    --shift_M 4096 \
    --n_samples 50 \
    --max_new_tokens 128 \
    --debug_n 2 \
    --output_dir "$OUTPUT_DIR"

echo "=== Debug run complete ==="
