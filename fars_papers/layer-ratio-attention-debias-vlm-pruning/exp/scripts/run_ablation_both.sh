#!/bin/bash
# Run both ratio ablation configs sequentially (K_s=3 and K_s=2 at L12).
# Uses all available GPUs (set by gpu_count in TrainService).

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

DATASETS="refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test"
COMMON_ARGS="--dynamic --max-num 12 --prompt-style ref --keep-ratio 0.1 --pivot-ratio 0.7 --sim-threshold 0.8 --spatial-weight 0.5"

run_config() {
    local OUT_DIR=$1
    local EXTRA_ARGS=$2
    local CONFIG_NAME=$3

    echo ""
    echo "============================================"
    echo "=== Config: $CONFIG_NAME ==="
    echo "============================================"
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
        echo "WARNING: $FAIL rank(s) failed for $CONFIG_NAME"
    fi

    echo "=== Merging results for $CONFIG_NAME ==="
    python scripts/merge_results.py --out-dir $OUT_DIR --num-gpus $NUM_GPUS
    echo "=== Done: $CONFIG_NAME (failures=$FAIL) ==="

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
}

echo "=== Starting ablation runs ==="

run_config "results/ablation_ratio_ks3_km12" \
    "--method online --debiasing-mode ratio --shallow-layer 3 --prune-layer 12" \
    "ratio_ks3_km12"

run_config "results/ablation_ratio_ks2_km12" \
    "--method online --debiasing-mode ratio --shallow-layer 2 --prune-layer 12" \
    "ratio_ks2_km12"

echo ""
echo "=== All ablation runs complete ==="
