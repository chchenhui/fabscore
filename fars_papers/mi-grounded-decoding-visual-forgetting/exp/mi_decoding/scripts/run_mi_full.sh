#!/bin/bash
# Run adaptive MI decoding on full benchmark with N-GPU data parallelism.
# Usage: run_mi_full.sh <benchmark> [num_gpus] [max_new_tokens] [output_dir_name] [lam] [max_weight]
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

BENCHMARK=${1:-mmstar}
NUM_SHARDS=${2:-4}
MAX_NEW_TOKENS=${3:-512}
OUTPUT_DIR_NAME=${4:-adaptive_mi_vlaa_thinker_7b}
LAM=${5:-0.005}
MAX_WEIGHT=${6:-5.0}
OUTPUT_DIR="mi_decoding/outputs/${OUTPUT_DIR_NAME}/${BENCHMARK}"

echo "Running MI decoding on ${BENCHMARK} with ${NUM_SHARDS} shards, max_new_tokens=${MAX_NEW_TOKENS}"
echo "Hyperparams: lam=${LAM}, max_weight=${MAX_WEIGHT}"
echo "Output dir: ${OUTPUT_DIR}"

export WANDB_MODE=offline

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "Starting shard ${SHARD_ID}..."
    CUDA_VISIBLE_DEVICES=${SHARD_ID} python mi_decoding/scripts/run_mi_baseline.py \
        --benchmark ${BENCHMARK} \
        --num_shards ${NUM_SHARDS} \
        --shard_id ${SHARD_ID} \
        --output_dir ${OUTPUT_DIR} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --lam ${LAM} \
        --max_weight ${MAX_WEIGHT} &
done

echo "Waiting for all shards to complete..."
wait
echo "All shards done!"

echo ""
echo "=== Merging and evaluating ==="
python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark ${BENCHMARK} \
    --output_dir ${OUTPUT_DIR} \
    --results_file "mi_decoding/results/${OUTPUT_DIR_NAME}_${BENCHMARK}.json"

echo "=== Complete ==="
