#!/bin/bash
# Debug comparison: test online vs d2pruner at prune_layer=4, and d2pruner at prune_layer=2
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

echo "=== Test 1: D2Pruner at prune_layer=4 (same as online) ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B \
    --datasets refcoco_val \
    --method d2pruner \
    --keep-ratio 0.1 \
    --prune-layer 4 \
    --bias-prior-path data/bias_prior/layer_1.pt \
    --pivot-ratio 0.7 --sim-threshold 0.8 --spatial-weight 0.5 \
    --dynamic --max-num 12 --prompt-style ref \
    --out-dir results/debug_d2pruner_l4 \
    --max-samples 200

echo ""
echo "=== Test 2: Online at prune_layer=4 shallow_layer=3 ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B \
    --datasets refcoco_val \
    --method online \
    --keep-ratio 0.1 \
    --prune-layer 4 \
    --shallow-layer 3 \
    --pivot-ratio 0.7 --sim-threshold 0.8 --spatial-weight 0.5 \
    --dynamic --max-num 12 --prompt-style ref \
    --out-dir results/debug_online_l4 \
    --max-samples 200

echo ""
echo "=== Test 3: FastV at prune_layer=4 ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B \
    --datasets refcoco_val \
    --method fastv \
    --keep-ratio 0.1 \
    --prune-layer 4 \
    --dynamic --max-num 12 --prompt-style ref \
    --out-dir results/debug_fastv_l4 \
    --max-samples 200

echo "=== All debug tests done ==="
