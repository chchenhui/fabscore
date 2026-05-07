#!/bin/bash
# Full evaluation with 4-GPU data parallelism.
# Usage: bash run_full_eval_4gpu.sh <config_name>

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

CONFIG=${1:-rawmis_km12}
NUM_GPUS=4
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
*)
    echo "Unknown config: $CONFIG"
    exit 1
    ;;
esac

echo "=== Full evaluation: config=$CONFIG, num_gpus=$NUM_GPUS ==="
echo "=== Output dir: $OUT_DIR ==="
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

echo ""
echo "--- Results for $CONFIG ---"
python3 -c "
import json
with open('$OUT_DIR/summary.json') as f:
    data = json.load(f)
accs = []
for k, v in sorted(data.items()):
    if 'accuracy' in v:
        print(f'  {k}: {v[\"accuracy\"]:.4f}')
        accs.append(v['accuracy'])
if accs:
    print(f'  AVERAGE: {sum(accs)/len(accs):.4f}')
"
