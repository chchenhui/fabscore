#!/bin/bash
# Ablation: fixed-gamma MI vs adaptive MI decoding on VLAA-Thinker-7B.
# GPUs 0-3: fixed-gamma (gamma=0.5), GPUs 4-7: adaptive MI (lambda=0.02).
# Both use alpha=0.3, max_weight=5.0. Runs MMStar 200-item subset then full HallusionBench.
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

export WANDB_MODE=offline

ALPHA=0.3
LAM=0.02
MAX_WEIGHT=5.0
MAX_NEW_TOKENS=512
SUBSET_FILE=mi_decoding/configs/mmstar_subset_200.json

FIXED_GAMMA_MMSTAR_DIR="mi_decoding/outputs/fixed_gamma_mi_vlaa_thinker_7b/mmstar"
FIXED_GAMMA_HB_DIR="mi_decoding/outputs/fixed_gamma_mi_vlaa_thinker_7b/hallusionbench"
ADAPTIVE_MMSTAR_DIR="mi_decoding/outputs/adaptive_mi_ablation_vlaa_thinker_7b/mmstar"
ADAPTIVE_HB_DIR="mi_decoding/outputs/adaptive_mi_ablation_vlaa_thinker_7b/hallusionbench"

run_fixed_gamma() {
    local GPU_ID=$1
    echo "[Fixed-gamma GPU ${GPU_ID}] Starting MMStar subset..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python mi_decoding/scripts/run_mi_baseline.py \
        --benchmark mmstar \
        --subset_file ${SUBSET_FILE} \
        --num_shards 4 --shard_id ${GPU_ID} \
        --output_dir ${FIXED_GAMMA_MMSTAR_DIR} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --alpha ${ALPHA} --lam ${LAM} --max_weight ${MAX_WEIGHT} \
        --fixed_gamma 0.5

    echo "[Fixed-gamma GPU ${GPU_ID}] Starting HallusionBench..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python mi_decoding/scripts/run_mi_baseline.py \
        --benchmark hallusionbench \
        --num_shards 4 --shard_id ${GPU_ID} \
        --output_dir ${FIXED_GAMMA_HB_DIR} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --alpha ${ALPHA} --lam ${LAM} --max_weight ${MAX_WEIGHT} \
        --fixed_gamma 0.5

    echo "[Fixed-gamma GPU ${GPU_ID}] Done."
}

run_adaptive() {
    local GPU_OFFSET=$1
    local SHARD_ID=$2
    echo "[Adaptive GPU ${GPU_OFFSET}] Starting MMStar subset..."
    CUDA_VISIBLE_DEVICES=${GPU_OFFSET} python mi_decoding/scripts/run_mi_baseline.py \
        --benchmark mmstar \
        --subset_file ${SUBSET_FILE} \
        --num_shards 4 --shard_id ${SHARD_ID} \
        --output_dir ${ADAPTIVE_MMSTAR_DIR} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --alpha ${ALPHA} --lam ${LAM} --max_weight ${MAX_WEIGHT}

    echo "[Adaptive GPU ${GPU_OFFSET}] Starting HallusionBench..."
    CUDA_VISIBLE_DEVICES=${GPU_OFFSET} python mi_decoding/scripts/run_mi_baseline.py \
        --benchmark hallusionbench \
        --num_shards 4 --shard_id ${SHARD_ID} \
        --output_dir ${ADAPTIVE_HB_DIR} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --alpha ${ALPHA} --lam ${LAM} --max_weight ${MAX_WEIGHT}

    echo "[Adaptive GPU ${GPU_OFFSET}] Done."
}

echo "=== Ablation: Fixed-gamma vs Adaptive MI ==="
echo "Alpha=${ALPHA}, Lambda=${LAM}, MaxWeight=${MAX_WEIGHT}"

for i in 0 1 2 3; do
    run_fixed_gamma $i &
done

for i in 0 1 2 3; do
    run_adaptive $((i + 4)) $i &
done

echo "Waiting for all 8 processes..."
wait
echo "All processes done!"

echo ""
echo "=== Merging and evaluating Fixed-gamma MMStar ==="
python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark mmstar \
    --output_dir ${FIXED_GAMMA_MMSTAR_DIR} \
    --results_file "mi_decoding/results/fixed_gamma_mi_vlaa_thinker_7b_mmstar.json"

echo "=== Merging and evaluating Fixed-gamma HallusionBench ==="
python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark hallusionbench \
    --output_dir ${FIXED_GAMMA_HB_DIR} \
    --results_file "mi_decoding/results/fixed_gamma_mi_vlaa_thinker_7b_hallusionbench.json"

echo "=== Merging and evaluating Adaptive MMStar ==="
python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark mmstar \
    --output_dir ${ADAPTIVE_MMSTAR_DIR} \
    --results_file "mi_decoding/results/adaptive_mi_ablation_vlaa_thinker_7b_mmstar.json"

echo "=== Merging and evaluating Adaptive HallusionBench ==="
python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark hallusionbench \
    --output_dir ${ADAPTIVE_HB_DIR} \
    --results_file "mi_decoding/results/adaptive_mi_ablation_vlaa_thinker_7b_hallusionbench.json"

echo "=== All evaluations complete ==="
