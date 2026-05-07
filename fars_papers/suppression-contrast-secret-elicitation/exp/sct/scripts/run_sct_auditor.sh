#!/bin/bash
# Run auditor evaluation for SCT scored results on all 3 Taboo models.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp/.venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/.cache/huggingface
export PYTHONUNBUFFERED=1

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp

export $(grep HF_TOKEN .env | xargs)

python3 sct/scripts/run_auditor_local.py \
    --scored_files \
        sct/outputs/taboo_gold_sct_scored.json \
        sct/outputs/taboo_moon_sct_scored.json \
        sct/outputs/taboo_flag_sct_scored.json \
    --secret_words gold moon flag \
    --auditor_model google/gemma-3-4b-it \
    --num_tokens 5 \
    --output_path sct/results/taboo_direct_sct_auditor.json

echo "SCT auditor evaluation complete"
