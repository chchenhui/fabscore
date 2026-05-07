#!/bin/bash
# Quick sanity check: SinkCast downstream with padding-based shift (3 samples)
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
K=${2:-4}
OUTPUT_DIR="$PROJECT_ROOT/sinkcast/results/downstream_debug_opt"
mkdir -p "$OUTPUT_DIR"

echo "Debug SinkCast downstream (padding-based shift): model=$MODEL K=$K"
echo ""

echo "=== RULER debug (3 samples, seq_len=4096) ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/ruler_shift_sinkcast.py" \
    --model "$MODEL" \
    --K "$K" \
    --seq_lengths 4096 \
    --shift_M 4096 \
    --n_samples 3 \
    --max_new_tokens 64 \
    --tasks niah_single qa \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== LongBench debug (3 samples) ==="
python "$PROJECT_ROOT/sinkcast/benchmarks/longbench_shift_sinkcast.py" \
    --model "$MODEL" \
    --K "$K" \
    --shift_M 4096 \
    --max_context 4096 \
    --debug_n 3 \
    --tasks narrativeqa trec \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Debug complete ==="
