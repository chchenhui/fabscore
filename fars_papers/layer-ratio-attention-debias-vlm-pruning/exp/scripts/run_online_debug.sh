#!/bin/bash
# Debug: run online method on small subset to verify correctness
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

echo "=== Debug: Online shallow-layer prior (50 samples) ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B \
    --datasets refcoco_val \
    --method online \
    --keep-ratio 0.1 \
    --prune-layer 4 \
    --shallow-layer 3 \
    --pivot-ratio 0.7 \
    --sim-threshold 0.8 \
    --spatial-weight 0.5 \
    --dynamic --max-num 12 --prompt-style ref \
    --out-dir results/online_debug \
    --max-samples 50

echo "=== Debug done ==="
