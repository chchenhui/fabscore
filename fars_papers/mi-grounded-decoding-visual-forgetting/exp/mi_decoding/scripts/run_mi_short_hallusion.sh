#!/bin/bash
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

BENCHMARK=hallusionbench
NUM_SHARDS=4
MAX_NEW_TOKENS=128
OUTPUT_DIR_NAME=adaptive_mi_vlaa_thinker_7b_short
OUTPUT_DIR="mi_decoding/outputs/${OUTPUT_DIR_NAME}/${BENCHMARK}"

echo "Running MI decoding (short) on ${BENCHMARK} with ${NUM_SHARDS} shards, max_new_tokens=${MAX_NEW_TOKENS}"

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "Starting shard ${SHARD_ID}..."
    CUDA_VISIBLE_DEVICES=${SHARD_ID} python mi_decoding/scripts/run_mi_baseline.py \
        --benchmark ${BENCHMARK} \
        --num_shards ${NUM_SHARDS} \
        --shard_id ${SHARD_ID} \
        --output_dir ${OUTPUT_DIR} \
        --max_new_tokens ${MAX_NEW_TOKENS} &
done

wait
echo "All shards done!"

python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark ${BENCHMARK} \
    --output_dir ${OUTPUT_DIR} \
    --results_file "mi_decoding/results/${OUTPUT_DIR_NAME}_${BENCHMARK}.json"

echo "=== Complete ==="
