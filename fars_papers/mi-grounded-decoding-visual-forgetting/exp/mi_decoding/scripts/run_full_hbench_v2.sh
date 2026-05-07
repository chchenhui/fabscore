#!/bin/bash
# Full HallusionBench run: alpha=0.8, lambda=0.005, max_weight=5.0, 4-GPU data-parallel
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

export WANDB_MODE=offline
OUTPUT_DIR="mi_decoding/outputs/mi_optimized_v2/hallusionbench"

echo "=== HallusionBench Full Run: alpha=0.8, lam=0.005, mw=5.0 ==="
for SHARD_ID in 0 1 2 3; do
    echo "Starting shard ${SHARD_ID}..."
    CUDA_VISIBLE_DEVICES=${SHARD_ID} python mi_decoding/scripts/run_mi_baseline.py \
        --benchmark hallusionbench \
        --num_shards 4 --shard_id ${SHARD_ID} \
        --output_dir ${OUTPUT_DIR} \
        --max_new_tokens 512 \
        --lam 0.005 --alpha 0.8 --max_weight 5.0 &
done

echo "Waiting..."
wait
echo "All shards done!"

echo ""
echo "=== Merging and evaluating ==="
python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark hallusionbench \
    --output_dir ${OUTPUT_DIR} \
    --results_file "mi_decoding/results/mi_optimized_v2_hallusionbench.json"
echo "=== Complete ==="
