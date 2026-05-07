#!/bin/bash
# Targeted debug sweep v3: Test deeper pruning layers and best configs from v2.
# Focus on rawmis at deeper layers and weighted_combo with optimal alpha at deeper layers.

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

COMMON="--dynamic --max-num 12 --prompt-style ref --datasets refcoco_val --max-samples 200 --method online"

declare -A CONFIGS

CONFIGS[rawmis_km10]="--debiasing-mode raw_mis --shallow-layer 2 --prune-layer 10"
CONFIGS[rawmis_km12]="--debiasing-mode raw_mis --shallow-layer 2 --prune-layer 12"
CONFIGS[rawmis_km16]="--debiasing-mode raw_mis --shallow-layer 2 --prune-layer 16"

CONFIGS[wc_a05_ks2_km10]="--debiasing-mode weighted_combo --combo-alpha 0.5 --shallow-layer 2 --prune-layer 10"
CONFIGS[wc_a05_ks2_km12]="--debiasing-mode weighted_combo --combo-alpha 0.5 --shallow-layer 2 --prune-layer 12"
CONFIGS[wc_a07_ks2_km8]="--debiasing-mode weighted_combo --combo-alpha 0.7 --shallow-layer 2 --prune-layer 8"
CONFIGS[wc_a07_ks2_km10]="--debiasing-mode weighted_combo --combo-alpha 0.7 --shallow-layer 2 --prune-layer 10"
CONFIGS[wc_a07_ks3_km8]="--debiasing-mode weighted_combo --combo-alpha 0.7 --shallow-layer 3 --prune-layer 8"
CONFIGS[wc_a09_ks2_km8]="--debiasing-mode weighted_combo --combo-alpha 0.9 --shallow-layer 2 --prune-layer 8"

CONFIGS[wc_a05_ks3_km8]="--debiasing-mode weighted_combo --combo-alpha 0.5 --shallow-layer 3 --prune-layer 8"
CONFIGS[wc_a03_ks3_km8]="--debiasing-mode weighted_combo --combo-alpha 0.3 --shallow-layer 3 --prune-layer 8"

CONFIGS[res_b03_ks2_km8]="--debiasing-mode residual --residual-beta 0.3 --shallow-layer 2 --prune-layer 8"
CONFIGS[res_b05_ks2_km8]="--debiasing-mode residual --residual-beta 0.5 --shallow-layer 2 --prune-layer 8"

echo "=== Debug sweep v3: ${#CONFIGS[@]} configurations ==="

for name in $(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort); do
    args="${CONFIGS[$name]}"
    outdir="results/debug_sweep_v3_${name}"
    mkdir -p "$outdir"
    echo ""
    echo "=== Running: $name ==="
    echo "  Args: $args"

    python scripts/eval_refcoco_baseline.py \
        --model-path models/InternVL2_5-8B \
        $COMMON \
        --keep-ratio 0.1 \
        --out-dir "$outdir" \
        $args

    if [ -f "$outdir/refcoco_val.json" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$outdir/refcoco_val.json')); print(f'{d[\"accuracy\"]:.4f}')")
        echo "  >> $name: accuracy = $acc"
    else
        echo "  >> $name: FAILED"
    fi
done

echo ""
echo "=== Sweep v3 complete ==="
