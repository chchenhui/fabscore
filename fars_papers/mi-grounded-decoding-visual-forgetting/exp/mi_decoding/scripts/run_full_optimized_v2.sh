#!/bin/bash
# Full run with optimized hyperparams: alpha=0.8, lambda=0.005, max_weight=5.0
# GPUs 0-3: MMStar (1500 items, 4-way data-parallel)
# GPUs 4-7: HallusionBench (1129 items, 4-way data-parallel)
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

export WANDB_MODE=offline

LAM=0.005
ALPHA=0.8
MAX_WEIGHT=5.0
MAX_NEW_TOKENS=512

MMSTAR_DIR="mi_decoding/outputs/mi_optimized_v2/mmstar"
HBENCH_DIR="mi_decoding/outputs/mi_optimized_v2/hallusionbench"

echo "=== Full Optimized Run v2: alpha=${ALPHA}, lam=${LAM}, mw=${MAX_WEIGHT} ==="
echo "MMStar: GPUs 0-3, HallusionBench: GPUs 4-7"

for SHARD_ID in 0 1 2 3; do
    echo "Starting MMStar shard ${SHARD_ID} on GPU ${SHARD_ID}..."
    CUDA_VISIBLE_DEVICES=${SHARD_ID} python mi_decoding/scripts/run_mi_baseline.py \
        --benchmark mmstar \
        --num_shards 4 --shard_id ${SHARD_ID} \
        --output_dir ${MMSTAR_DIR} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --lam ${LAM} --alpha ${ALPHA} --max_weight ${MAX_WEIGHT} &
done

for SHARD_ID in 0 1 2 3; do
    GPU_ID=$((SHARD_ID + 4))
    echo "Starting HallusionBench shard ${SHARD_ID} on GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python mi_decoding/scripts/run_mi_baseline.py \
        --benchmark hallusionbench \
        --num_shards 4 --shard_id ${SHARD_ID} \
        --output_dir ${HBENCH_DIR} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --lam ${LAM} --alpha ${ALPHA} --max_weight ${MAX_WEIGHT} &
done

echo "Waiting for all 8 shards to complete..."
wait
echo "All shards done!"

echo ""
echo "=== Merging and evaluating MMStar ==="
python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark mmstar \
    --output_dir ${MMSTAR_DIR} \
    --results_file "mi_decoding/results/mi_optimized_v2_mmstar.json"

echo ""
echo "=== Merging and evaluating HallusionBench ==="
python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark hallusionbench \
    --output_dir ${HBENCH_DIR} \
    --results_file "mi_decoding/results/mi_optimized_v2_hallusionbench.json"

echo ""
echo "=== Complete ==="
