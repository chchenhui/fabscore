#!/bin/bash
# Sanity check: ratio debiasing at L12, K_s=3, on refcoco_val only (1 GPU)
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

OUT_DIR="results/ablation_sanity_ratio_ks3_km12"
mkdir -p $OUT_DIR

python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B \
    --datasets refcoco_val \
    --method online \
    --debiasing-mode ratio \
    --prune-layer 12 \
    --shallow-layer 3 \
    --keep-ratio 0.1 \
    --pivot-ratio 0.7 \
    --sim-threshold 0.8 \
    --spatial-weight 0.5 \
    --dynamic \
    --max-num 12 \
    --prompt-style ref \
    --out-dir $OUT_DIR

echo "=== Sanity check complete ==="
echo "Results:"
cat $OUT_DIR/summary.json
