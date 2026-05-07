#!/bin/bash
# Extract activations for user-female model on gender_direct_test prompts.
set -e

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp"
source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

python3 "${PROJECT_ROOT}/sct/extraction/extract_activations.py" \
    --model_name "bcywinski/gemma-2-9b-it-user-female" \
    --prompts_file "${PROJECT_ROOT}/benchmark/prompts/gender/gender_direct_test.txt" \
    --output_path "${PROJECT_ROOT}/sct/outputs/user_gender_female_activations.json" \
    --mid_layer 32 \
    --final_layer 41 \
    --top_k 200 \
    --num_responses 10 \
    --max_new_tokens 200 \
    --temperature 1.0 \
    --seed 1 \
    --batch_size 100

echo "=== Female extraction done ==="
