#!/bin/bash
# Run online shallow-layer prior evaluation with data parallelism across GPUs.
# Online method: K_s=3, K_m=4 (prune_layer=4, shallow_layer=3)
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

NUM_GPUS=${1:-8}
DATASETS="refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test"
OUT_DIR="results/online_prior"
COMMON_ARGS="--dynamic --max-num 12 --prompt-style ref"
EXTRA_ARGS="--keep-ratio 0.1 --prune-layer 4 --shallow-layer 3 --pivot-ratio 0.7 --sim-threshold 0.8 --spatial-weight 0.5"

echo "=== Online shallow-layer prior evaluation, num_gpus=$NUM_GPUS ==="
echo "=== Output dir: $OUT_DIR ==="

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

echo "Waiting for all ranks to complete..."
FAIL=0
for job in $(jobs -p); do
    wait $job || let "FAIL+=1"
done

if [ "$FAIL" != "0" ]; then
    echo "WARNING: $FAIL rank(s) failed"
fi

echo "=== Merging results ==="
python scripts/merge_results.py --out-dir $OUT_DIR --num-gpus $NUM_GPUS

echo "=== Done (failures=$FAIL) ==="
