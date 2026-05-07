#!/bin/bash
# Adaptive MI decoding on Qwen2.5-VL-7B-Instruct, 300-item MMStar subset, N-GPU data parallelism.
# Hyperparams: lambda=0.02, alpha=0.3, t0=prompt_length, max_weight=5.0
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

NUM_SHARDS=${1:-4}
OUTPUT_DIR="mi_decoding/outputs/adaptive_mi_qwen25vl_instruct/mmstar_subset"

export WANDB_MODE=offline

echo "Running MI decoding Qwen2.5-VL-7B-Instruct on 300-item MMStar subset with ${NUM_SHARDS} shards"
echo "Hyperparams: lam=0.02, alpha=0.3, t0=prompt_length, max_weight=5.0"

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "Starting shard ${SHARD_ID}..."
    CUDA_VISIBLE_DEVICES=${SHARD_ID} python mi_decoding/scripts/run_mi_baseline.py \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --benchmark mmstar \
        --max_new_tokens 512 \
        --lam 0.02 \
        --alpha 0.3 \
        --t0_prompt_length \
        --max_weight 5.0 \
        --subset_file mi_decoding/configs/mmstar_subset_300.json \
        --num_shards ${NUM_SHARDS} \
        --shard_id ${SHARD_ID} \
        --output_dir ${OUTPUT_DIR} &
done

echo "Waiting for all shards to complete..."
wait
echo "All shards done!"

echo ""
echo "=== Merging and evaluating ==="
python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark mmstar \
    --output_dir ${OUTPUT_DIR} \
    --results_file "mi_decoding/results/adaptive_mi_qwen25vl_instruct_mmstar_subset.json"

echo "=== Complete ==="
