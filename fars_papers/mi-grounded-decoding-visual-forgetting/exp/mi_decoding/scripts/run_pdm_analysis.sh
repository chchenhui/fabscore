#!/bin/bash
# PDM-H analysis: collect per-step logits for vanilla and adaptive MI decoding.
# Usage: bash run_pdm_analysis.sh <method> <benchmark> <num_gpus>
# Example: bash run_pdm_analysis.sh vanilla mmstar 4
set -e

METHOD=${1:?Usage: run_pdm_analysis.sh <method> <benchmark> <num_gpus>}
BENCHMARK=${2:?Usage: run_pdm_analysis.sh <method> <benchmark> <num_gpus>}
NUM_GPUS=${3:-4}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
export WANDB_MODE=online

echo "=== PDM-H Analysis ==="
echo "Method: $METHOD"
echo "Benchmark: $BENCHMARK"
echo "GPUs: $NUM_GPUS"
echo "Project root: $PROJECT_ROOT"

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

echo "Waiting for all $NUM_GPUS shards to complete..."
wait
echo "All shards done for $METHOD / $BENCHMARK"
