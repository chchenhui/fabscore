#!/bin/bash
# Run auditor evaluation for User Gender models using Gemma-3-4B-IT via vLLM.
# Uses the gender-specific auditor template (binary male/female classification).
set -e

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp"
source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/.cache/huggingface
export $(grep HF_TOKEN "${PROJECT_ROOT}/.env" | xargs)

cd "${PROJECT_ROOT}"

python3 sct/scripts/run_auditor_local.py \
    --scored_files \
        sct/outputs/user_gender_female_scored.json \
        sct/outputs/user_gender_male_scored.json \
    --secret_words female male \
    --auditor_model google/gemma-3-4b-it \
    --template_path benchmark/prompts/gender/auditor_guess_tokens.txt \
    --num_tokens 5 \
    --output_path sct/results/user_gender_direct_logit_lens_auditor.json

echo "=== User Gender Auditor Evaluation Complete ==="
