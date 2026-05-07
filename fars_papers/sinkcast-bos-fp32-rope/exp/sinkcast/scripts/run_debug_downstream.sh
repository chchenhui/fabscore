#!/bin/bash
# Debug sanity check for downstream position-shift benchmarks.
# Runs 1 model, ~5 samples per task, 1 GPU.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"
set -a
source "$PROJECT_ROOT/.env"
set +a
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export WANDB_MODE=offline

MODEL=${1:-"llama-3.1-8b"}

echo "=== Debug RULER position-shift ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/ruler_shift.py" \
    --model "$MODEL" \
    --seq_lengths 4096 \
    --shift_M 4096 \
    --n_samples 50 \
    --debug_n 5 \
    --tasks niah_single qa \
    --output_dir "$PROJECT_ROOT/sinkcast/results/debug/downstream"

echo "=== Debug LongBench position-shift ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/longbench_shift.py" \
    --model "$MODEL" \
    --shift_M 4096 \
    --debug_n 5 \
    --tasks narrativeqa trec \
    --output_dir "$PROJECT_ROOT/sinkcast/results/debug/downstream"

echo "=== Debug complete ==="
