#!/bin/bash
# Debug: test online method with different K_s and K_m combinations
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

COMMON="--dynamic --max-num 12 --prompt-style ref --pivot-ratio 0.7 --sim-threshold 0.8 --spatial-weight 0.5 --keep-ratio 0.1 --max-samples 200"

echo "=== Online: K_s=1, K_m=2 (prune_layer=2, shallow_layer=1) ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B --datasets refcoco_val \
    --method online --prune-layer 2 --shallow-layer 1 \
    --out-dir results/debug_online_ks1_km2 $COMMON

echo ""
echo "=== Online: K_s=1, K_m=4 (prune_layer=4, shallow_layer=1) ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B --datasets refcoco_val \
    --method online --prune-layer 4 --shallow-layer 1 \
    --out-dir results/debug_online_ks1_km4 $COMMON

echo ""
echo "=== Online: K_s=2, K_m=4 (prune_layer=4, shallow_layer=2) ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B --datasets refcoco_val \
    --method online --prune-layer 4 --shallow-layer 2 \
    --out-dir results/debug_online_ks2_km4 $COMMON

echo ""
echo "=== D2Pruner: prune_layer=2 (baseline config, sanity check) ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B --datasets refcoco_val \
    --method d2pruner --prune-layer 2 \
    --bias-prior-path data/bias_prior/layer_1.pt \
    --out-dir results/debug_d2pruner_l2 $COMMON

echo "=== All done ==="
