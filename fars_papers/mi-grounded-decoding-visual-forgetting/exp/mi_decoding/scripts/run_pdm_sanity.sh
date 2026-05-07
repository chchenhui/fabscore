#!/bin/bash
# Sanity check for PDM analysis: 2 items, vanilla, mmstar, 1 GPU
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
export WANDB_MODE=online

echo "=== PDM-H Sanity Check ==="

python "$SCRIPT_DIR/run_pdm_analysis.py" \
    --benchmark mmstar \
    --method vanilla \
    --max_new_tokens 512 \
    --save_interval 10 \
    --subset_size 50 \
    --seed 42 \
    --max_items 2 \
    --output_dir "$PROJECT_ROOT/mi_decoding/outputs/pdm_sanity"

echo "=== Sanity check done ==="
echo "Checking saved files..."
ls -la "$PROJECT_ROOT/mi_decoding/outputs/pdm_sanity/"
