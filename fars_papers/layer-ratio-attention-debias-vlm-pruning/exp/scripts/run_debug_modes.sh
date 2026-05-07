#!/bin/bash
# Debug: test all debiasing modes at prune_layer=4 and best K_s combos
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

COMMON="--dynamic --max-num 12 --prompt-style ref --pivot-ratio 0.7 --sim-threshold 0.8 --spatial-weight 0.5 --keep-ratio 0.1 --max-samples 200"

echo "=== 1. raw_mis: K_m=4 (A_mid + MIS, no debiasing) ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B --datasets refcoco_val \
    --method online --prune-layer 4 --shallow-layer 3 --debiasing-mode raw_mis \
    --out-dir results/debug_rawmis_l4 $COMMON

echo ""
echo "=== 2. subtract: K_s=2, K_m=4 ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B --datasets refcoco_val \
    --method online --prune-layer 4 --shallow-layer 2 --debiasing-mode subtract \
    --out-dir results/debug_subtract_ks2_l4 $COMMON

echo ""
echo "=== 3. zscore: K_s=2, K_m=4 ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B --datasets refcoco_val \
    --method online --prune-layer 4 --shallow-layer 2 --debiasing-mode zscore \
    --out-dir results/debug_zscore_ks2_l4 $COMMON

echo ""
echo "=== 4. subtract: K_s=1, K_m=4 ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B --datasets refcoco_val \
    --method online --prune-layer 4 --shallow-layer 1 --debiasing-mode subtract \
    --out-dir results/debug_subtract_ks1_l4 $COMMON

echo ""
echo "=== 5. raw_mis: K_m=2 (A_mid + MIS at original prune layer) ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B --datasets refcoco_val \
    --method online --prune-layer 2 --shallow-layer 1 --debiasing-mode raw_mis \
    --out-dir results/debug_rawmis_l2 $COMMON

echo ""
echo "=== 6. subtract: K_s=1, K_m=2 ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B --datasets refcoco_val \
    --method online --prune-layer 2 --shallow-layer 1 --debiasing-mode subtract \
    --out-dir results/debug_subtract_ks1_l2 $COMMON

echo ""
echo "=== 7. ratio: K_s=2, K_m=4 (best from earlier) ==="
python scripts/eval_refcoco_baseline.py \
    --model-path models/InternVL2_5-8B --datasets refcoco_val \
    --method online --prune-layer 4 --shallow-layer 2 --debiasing-mode ratio \
    --out-dir results/debug_ratio_ks2_l4 $COMMON

echo "=== All done ==="
