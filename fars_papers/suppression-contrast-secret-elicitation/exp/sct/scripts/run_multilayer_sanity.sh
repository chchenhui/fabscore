#!/bin/bash
# Sanity check: run multi-layer extraction on 10 examples for gold model
set -ex

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp"
source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

python3 "${PROJECT_ROOT}/sct/extraction/extract_multilayer.py" \
    --model_name "bcywinski/gemma-2-9b-it-taboo-gold" \
    --source_activations "${PROJECT_ROOT}/sct/outputs/taboo_gold_activations.json" \
    --output_path "${PROJECT_ROOT}/sct/outputs/taboo_gold_multilayer_sanity.json" \
    --mid_layers 15 20 24 28 32 36 \
    --final_layer 41 \
    --top_k_out 50 \
    --max_examples 10

echo "=== Sanity check complete ==="
