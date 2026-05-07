#!/bin/bash
# Debug sweep for optimized debiasing modes on refcoco_val (200 samples).
# Tests weighted_combo, entropy_ratio, residual modes with various layer configs and hyperparams.
# Also re-tests raw_mis at different layers for reference.

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

COMMON="--dynamic --max-num 12 --prompt-style ref --datasets refcoco_val --max-samples 200 --method online"

declare -A CONFIGS

CONFIGS[wc_a03_ks2_km4]="--debiasing-mode weighted_combo --combo-alpha 0.3 --shallow-layer 2 --prune-layer 4"
CONFIGS[wc_a05_ks2_km4]="--debiasing-mode weighted_combo --combo-alpha 0.5 --shallow-layer 2 --prune-layer 4"
CONFIGS[wc_a07_ks2_km4]="--debiasing-mode weighted_combo --combo-alpha 0.7 --shallow-layer 2 --prune-layer 4"
CONFIGS[wc_a09_ks2_km4]="--debiasing-mode weighted_combo --combo-alpha 0.9 --shallow-layer 2 --prune-layer 4"
CONFIGS[wc_a05_ks3_km4]="--debiasing-mode weighted_combo --combo-alpha 0.5 --shallow-layer 3 --prune-layer 4"
CONFIGS[wc_a07_ks3_km4]="--debiasing-mode weighted_combo --combo-alpha 0.7 --shallow-layer 3 --prune-layer 4"
CONFIGS[wc_a05_ks2_km6]="--debiasing-mode weighted_combo --combo-alpha 0.5 --shallow-layer 2 --prune-layer 6"
CONFIGS[wc_a07_ks2_km6]="--debiasing-mode weighted_combo --combo-alpha 0.7 --shallow-layer 2 --prune-layer 6"
CONFIGS[wc_a05_ks2_km8]="--debiasing-mode weighted_combo --combo-alpha 0.5 --shallow-layer 2 --prune-layer 8"
CONFIGS[wc_a07_ks2_km8]="--debiasing-mode weighted_combo --combo-alpha 0.7 --shallow-layer 2 --prune-layer 8"

CONFIGS[ent_ks2_km4]="--debiasing-mode entropy_ratio --shallow-layer 2 --prune-layer 4"
CONFIGS[ent_ks3_km4]="--debiasing-mode entropy_ratio --shallow-layer 3 --prune-layer 4"
CONFIGS[ent_ks2_km6]="--debiasing-mode entropy_ratio --shallow-layer 2 --prune-layer 6"

CONFIGS[res_b03_ks2_km4]="--debiasing-mode residual --residual-beta 0.3 --shallow-layer 2 --prune-layer 4"
CONFIGS[res_b05_ks2_km4]="--debiasing-mode residual --residual-beta 0.5 --shallow-layer 2 --prune-layer 4"
CONFIGS[res_b07_ks2_km4]="--debiasing-mode residual --residual-beta 0.7 --shallow-layer 2 --prune-layer 4"
CONFIGS[res_b05_ks3_km4]="--debiasing-mode residual --residual-beta 0.5 --shallow-layer 3 --prune-layer 4"
CONFIGS[res_b05_ks2_km6]="--debiasing-mode residual --residual-beta 0.5 --shallow-layer 2 --prune-layer 6"

CONFIGS[rawmis_km6]="--debiasing-mode raw_mis --shallow-layer 2 --prune-layer 6"
CONFIGS[rawmis_km8]="--debiasing-mode raw_mis --shallow-layer 2 --prune-layer 8"

echo "=== Debug sweep v2: ${#CONFIGS[@]} configurations ==="

RESULTS_FILE="results/debug_sweep_v2_summary.json"
echo "{" > $RESULTS_FILE

for name in $(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort); do
    args="${CONFIGS[$name]}"
    outdir="results/debug_sweep_v2_${name}"
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
        echo "  \"$name\": $acc," >> $RESULTS_FILE
    else
        echo "  >> $name: FAILED (no output)"
        echo "  \"$name\": -1," >> $RESULTS_FILE
    fi
done

echo "  \"_done\": true" >> $RESULTS_FILE
echo "}" >> $RESULTS_FILE

echo ""
echo "=== All results ==="
cat $RESULTS_FILE
echo ""
echo "=== Sweep complete ==="
