#!/bin/bash
# Score activations with constrained logit lens and compute token recovery
set -ex

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp"
source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

OUTPUTS="${PROJECT_ROOT}/sct/outputs"
RESULTS="${PROJECT_ROOT}/sct/results"
mkdir -p "$RESULTS"

for WORD in gold moon flag; do
    echo "=== Scoring taboo_${WORD} (control_tokens mode) ==="
    python3 "${PROJECT_ROOT}/sct/extraction/logit_lens.py" \
        --activations_file "${OUTPUTS}/taboo_${WORD}_activations.json" \
        --output_path "${OUTPUTS}/taboo_${WORD}_scored.json" \
        --alpha 0.1 \
        --top_k_out 20 \
        --mode control_tokens \
        --model_name "google/gemma-2-9b-it" 2>&1
done

echo "=== Computing Token Recovery ==="
python3 "${PROJECT_ROOT}/sct/evaluation/token_recovery.py" \
    --scored_files \
        "${OUTPUTS}/taboo_gold_scored.json" \
        "${OUTPUTS}/taboo_moon_scored.json" \
        "${OUTPUTS}/taboo_flag_scored.json" \
    --secret_words gold moon flag \
    --model_name "google/gemma-2-9b-it" \
    --k_values 5 20 \
    --output_path "${RESULTS}/taboo_direct_token_recovery.json" 2>&1

echo "=== Done ==="
