#!/bin/bash
# Run full evaluation with data parallelism across GPUs.
# Usage: bash run_full_eval.sh <method> [num_gpus]
# Methods: none, fastv, d2pruner
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

METHOD=${1:-none}
NUM_GPUS=${2:-8}
DATASETS="refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test"
COMMON_ARGS="--dynamic --max-num 12 --prompt-style ref"

case $METHOD in
    none)
        OUT_DIR="results/nopruning_v2"
        EXTRA_ARGS=""
        ;;
    fastv)
        OUT_DIR="results/fastv_v2"
        EXTRA_ARGS="--keep-ratio 0.1 --prune-layer 2"
        ;;
    d2pruner)
        OUT_DIR="results/d2pruner_v2"
        EXTRA_ARGS="--keep-ratio 0.1 --prune-layer 2 --bias-prior-path data/bias_prior/layer_1.pt --pivot-ratio 0.7 --sim-threshold 0.8 --spatial-weight 0.5"
        ;;
    *)
        echo "Unknown method: $METHOD"
        exit 1
        ;;
esac

echo "=== Full evaluation: method=$METHOD, num_gpus=$NUM_GPUS ==="
echo "=== Output dir: $OUT_DIR ==="
echo "=== Common args: $COMMON_ARGS ==="

mkdir -p $OUT_DIR

for RANK in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$RANK python scripts/eval_refcoco_baseline.py \
        --model-path models/InternVL2_5-8B \
        --datasets $DATASETS \
        --method $METHOD \
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
