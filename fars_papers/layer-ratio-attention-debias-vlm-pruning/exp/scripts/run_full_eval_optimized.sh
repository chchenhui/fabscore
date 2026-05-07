#!/bin/bash
# Full evaluation of optimized online methods on all 8 RefCOCO splits.
# Config passed as argument: rawmis_km12, wc_a05_ks2_km12

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

CONFIG=${1:-rawmis_km12}
NUM_GPUS=${2:-8}
DATASETS="refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test"
COMMON_ARGS="--dynamic --max-num 12 --prompt-style ref --keep-ratio 0.1"

case $CONFIG in
rawmis_km12)
    OUT_DIR="results/opt_rawmis_km12"
    EXTRA_ARGS="--method online --debiasing-mode raw_mis --shallow-layer 2 --prune-layer 12"
    ;;
wc_a05_ks2_km12)
    OUT_DIR="results/opt_wc_a05_ks2_km12"
    EXTRA_ARGS="--method online --debiasing-mode weighted_combo --combo-alpha 0.5 --shallow-layer 2 --prune-layer 12"
    ;;
rawmis_km10)
    OUT_DIR="results/opt_rawmis_km10"
    EXTRA_ARGS="--method online --debiasing-mode raw_mis --shallow-layer 2 --prune-layer 10"
    ;;
wc_a03_ks3_km8)
    OUT_DIR="results/opt_wc_a03_ks3_km8"
    EXTRA_ARGS="--method online --debiasing-mode weighted_combo --combo-alpha 0.3 --shallow-layer 3 --prune-layer 8"
    ;;
ratio_ks3_km12)
    OUT_DIR="results/ablation_ratio_ks3_km12"
    EXTRA_ARGS="--method online --debiasing-mode ratio --shallow-layer 3 --prune-layer 12 --pivot-ratio 0.7 --sim-threshold 0.8 --spatial-weight 0.5"
    ;;
ratio_ks2_km12)
    OUT_DIR="results/ablation_ratio_ks2_km12"
    EXTRA_ARGS="--method online --debiasing-mode ratio --shallow-layer 2 --prune-layer 12 --pivot-ratio 0.7 --sim-threshold 0.8 --spatial-weight 0.5"
    ;;
*)
    echo "Unknown config: $CONFIG"
    exit 1
    ;;
esac

echo "=== Full evaluation: config=$CONFIG, num_gpus=$NUM_GPUS ==="
echo "=== Output dir: $OUT_DIR ==="
echo "=== Extra args: $EXTRA_ARGS ==="
mkdir -p $OUT_DIR

for RANK in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$RANK python scripts/eval_refcoco_baseline.py \
        --model-path models/InternVL2_5-8B \
        --datasets $DATASETS \
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
echo "=== Done: $CONFIG (failures=$FAIL) ==="
