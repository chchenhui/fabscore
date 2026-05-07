#!/bin/bash
# Run visual replay baseline on full benchmark with N-GPU data parallelism.
# Usage: run_visual_replay_full.sh <benchmark> [num_gpus]
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

BENCHMARK=${1:-mmstar}
NUM_SHARDS=${2:-4}
MODEL_NAME="VLAA-Thinker-Qwen2.5VL-7B"
OUTPUT_DIR="mi_decoding/outputs/visual_replay_${MODEL_NAME}/${BENCHMARK}"

echo "Running visual replay on ${BENCHMARK} with ${NUM_SHARDS} shards"
echo "Output dir: ${OUTPUT_DIR}"

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "Starting shard ${SHARD_ID}..."
    CUDA_VISIBLE_DEVICES=${SHARD_ID} python mi_decoding/scripts/run_visual_replay_baseline.py \
        --benchmark ${BENCHMARK} \
        --num_shards ${NUM_SHARDS} \
        --shard_id ${SHARD_ID} \
        --output_dir ${OUTPUT_DIR} \
        --max_new_tokens 512 &
done

echo "Waiting for all shards to complete..."
wait
echo "All shards done!"

echo ""
echo "=== Merging and evaluating ==="
python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark ${BENCHMARK} \
    --output_dir ${OUTPUT_DIR} \
    --results_file "mi_decoding/results/visual_replay_${MODEL_NAME}_${BENCHMARK}.json"

echo "=== Complete ==="
