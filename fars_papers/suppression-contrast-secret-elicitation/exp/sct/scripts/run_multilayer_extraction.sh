#!/bin/bash
# Multi-layer full-vocab extraction for a single Taboo model.
# Usage: bash run_multilayer_extraction.sh <model_name> <secret_word>
set -ex

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp"
source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

MODEL_NAME="${1:?Usage: $0 <model_name> <secret_word>}"
SECRET_WORD="${2:?Usage: $0 <model_name> <secret_word>}"

SOURCE="${PROJECT_ROOT}/sct/outputs/taboo_${SECRET_WORD}_activations.json"
OUTPUT="${PROJECT_ROOT}/sct/outputs/taboo_${SECRET_WORD}_multilayer.json"

python3 "${PROJECT_ROOT}/sct/extraction/extract_multilayer.py" \
    --model_name "$MODEL_NAME" \
    --source_activations "$SOURCE" \
    --output_path "$OUTPUT" \
    --mid_layers 15 20 24 28 32 36 \
    --final_layer 41 \
    --top_k_out 50

echo "=== Done: ${SECRET_WORD} ==="
