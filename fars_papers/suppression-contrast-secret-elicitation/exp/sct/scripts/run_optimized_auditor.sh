#!/bin/bash
# Run auditor evaluation for optimized SCT variants on all 3 Taboo models.
set -e

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp
source .venv/bin/activate
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)

OUTPUTS=sct/outputs
RESULTS=sct/results

echo "=== Running auditor for optimized_sct variant ==="
python3 sct/scripts/run_auditor_local.py \
    --scored_files \
        ${OUTPUTS}/taboo_gold_optimized_sct_scored.json \
        ${OUTPUTS}/taboo_moon_optimized_sct_scored.json \
        ${OUTPUTS}/taboo_flag_optimized_sct_scored.json \
    --secret_words gold moon flag \
    --auditor_model google/gemma-3-4b-it \
    --num_tokens 5 \
    --output_path ${RESULTS}/taboo_optimized_sct_auditor.json

echo ""
echo "=== Running auditor for ctrl_sct_a0 variant ==="
python3 sct/scripts/run_auditor_local.py \
    --scored_files \
        ${OUTPUTS}/taboo_gold_ctrl_sct_a0_scored.json \
        ${OUTPUTS}/taboo_moon_ctrl_sct_a0_scored.json \
        ${OUTPUTS}/taboo_flag_ctrl_sct_a0_scored.json \
    --secret_words gold moon flag \
    --auditor_model google/gemma-3-4b-it \
    --num_tokens 5 \
    --output_path ${RESULTS}/taboo_ctrl_sct_a0_auditor.json

echo ""
echo "=== All auditor evaluations complete ==="
