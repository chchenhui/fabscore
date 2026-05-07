#!/bin/bash
# Full downstream position-shift evaluation for BF16 FlashAttention baseline.
# Usage: bash run_downstream_bf16.sh <model-alias>
# e.g.:  bash run_downstream_bf16.sh llama-3.1-8b
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"
set -a
source "$PROJECT_ROOT/.env"
set +a
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

MODEL=${1:?"Usage: $0 <model-alias>"}
OUTPUT_DIR="$PROJECT_ROOT/sinkcast/results/downstream/bf16_flash"

echo "========================================="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "========================================="

echo ""
echo "=== RULER Position-Shift Evaluation ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/ruler_shift.py" \
    --model "$MODEL" \
    --seq_lengths 4096 8192 \
    --shift_M 4096 \
    --n_samples 50 \
    --max_new_tokens 128 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== LongBench Position-Shift Evaluation ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/longbench_shift.py" \
    --model "$MODEL" \
    --shift_M 4096 \
    --max_context 8192 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== All downstream evaluations complete for $MODEL ==="
