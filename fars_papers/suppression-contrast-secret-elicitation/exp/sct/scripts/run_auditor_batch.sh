#!/bin/bash
# Run auditor evaluation for all 3 Taboo models using Gemma-3-4B-IT loaded locally.
# This script loads the auditor model once and processes all 3 scored files.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp/.venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/.cache/huggingface

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp

# Load HF_TOKEN from .env
export $(grep HF_TOKEN .env | xargs)

python3 sct/scripts/run_auditor_local.py \
    --scored_files \
        sct/outputs/taboo_gold_scored.json \
        sct/outputs/taboo_moon_scored.json \
        sct/outputs/taboo_flag_scored.json \
    --secret_words gold moon flag \
    --auditor_model google/gemma-3-4b-it \
    --num_tokens 5 \
    --output_path sct/results/taboo_direct_logit_lens_auditor.json

echo "Auditor evaluation complete"
