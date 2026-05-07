#!/bin/bash
# Full PDM-H analysis: all 4 combinations of method x benchmark on 4 GPUs.
# Each combination runs 4 shards in parallel (one per GPU), then next combination.
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

NUM_GPUS=4

run_combination() {
    local METHOD=$1
    local BENCHMARK=$2
    echo ""
    echo "========================================"
    echo "Running: method=$METHOD benchmark=$BENCHMARK gpus=$NUM_GPUS"
    echo "========================================"

    for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$SHARD_ID python "$SCRIPT_DIR/run_pdm_analysis.py" \
            --benchmark "$BENCHMARK" \
            --method "$METHOD" \
            --num_shards "$NUM_GPUS" \
            --shard_id "$SHARD_ID" \
            --max_new_tokens 512 \
            --save_interval 10 \
            --subset_size 50 \
            --seed 42 \
            &
    done
    wait
    echo "Done: $METHOD / $BENCHMARK"
}

run_combination vanilla mmstar
run_combination vanilla hallusionbench
run_combination adaptive_mi mmstar
run_combination adaptive_mi hallusionbench

echo ""
echo "========================================"
echo "All PDM-H analysis runs completed!"
echo "========================================"
