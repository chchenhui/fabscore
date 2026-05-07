#!/bin/bash
# Full eval: online ratio debiasing K_s=3, K_m=4 on all 8 RefCOCO splits
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

NUM_GPUS=${1:-8}
DATASETS="refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test"
OUT_DIR="results/online_ratio_ks3_km4"
COMMON_ARGS="--dynamic --max-num 12 --prompt-style ref"
EXTRA_ARGS="--keep-ratio 0.1 --prune-layer 4 --shallow-layer 3 --debiasing-mode ratio --pivot-ratio 0.7 --sim-threshold 0.8 --spatial-weight 0.5"

echo "=== Online ratio debiasing (K_s=3, K_m=4), num_gpus=$NUM_GPUS ==="
mkdir -p $OUT_DIR

for RANK in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$RANK python scripts/eval_refcoco_baseline.py \
        --model-path models/InternVL2_5-8B \
        --datasets $DATASETS \
        --method online \
        --out-dir $OUT_DIR \
        --num-gpus $NUM_GPUS \
        --rank $RANK \
        $COMMON_ARGS \
        $EXTRA_ARGS &
    echo "Launched rank $RANK on GPU $RANK"
done

wait
echo "=== Merging results ==="
python scripts/merge_results.py --out-dir $OUT_DIR --num-gpus $NUM_GPUS
echo "=== Done ==="
