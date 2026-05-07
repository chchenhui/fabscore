#!/bin/bash
# Full downstream eval: Mistral-7B-v0.3 with SinkCast K=1 + padding-based shift
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"
set -a
source "$PROJECT_ROOT/.env"
set +a
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export WANDB_MODE=offline

MODEL="mistral-7b-v0.3"
K=1
OUTPUT_DIR="$PROJECT_ROOT/sinkcast/results/downstream_opt/sinkcast_k${K}"
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "SinkCast K=$K downstream eval: $MODEL (optimized)"
echo "Output: $OUTPUT_DIR"
echo "========================================="

echo "=== RULER Position-Shift Evaluation ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/ruler_shift_sinkcast.py" \
    --model "$MODEL" \
    --K "$K" \
    --seq_lengths 4096 8192 \
    --shift_M 4096 \
    --n_samples 50 \
    --max_new_tokens 128 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== LongBench Position-Shift Evaluation ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/longbench_shift_sinkcast.py" \
    --model "$MODEL" \
    --K "$K" \
    --shift_M 4096 \
    --max_context 8192 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== All downstream evaluations complete for $MODEL K=$K ==="
