#!/bin/bash
# Sanity check: run extraction on gold model with 5 prompts, 2 responses
set -ex

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp"
source "${PROJECT_ROOT}/.venv/bin/activate"

export PYTHONUNBUFFERED=1

echo "=== Sanity Check: Activation Extraction ==="
echo "Python: $(which python3)"
echo "Torch version check..."
python3 -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available())"

python3 "${PROJECT_ROOT}/sct/extraction/extract_activations.py" \
    --model_name "bcywinski/gemma-2-9b-it-taboo-gold" \
    --prompts_file "${PROJECT_ROOT}/benchmark/prompts/taboo/taboo_direct_test.txt" \
    --output_path "${PROJECT_ROOT}/sct/outputs/sanity_check_gold.json" \
    --mid_layer 32 \
    --final_layer 41 \
    --top_k 200 \
    --num_responses 2 \
    --max_new_tokens 200 \
    --temperature 1.0 \
    --seed 1 \
    --batch_size 10 \
    --max_prompts 5 2>&1

echo "=== Sanity Check: Constrained Logit Lens Scoring ==="
python3 "${PROJECT_ROOT}/sct/extraction/logit_lens.py" \
    --activations_file "${PROJECT_ROOT}/sct/outputs/sanity_check_gold.json" \
    --output_path "${PROJECT_ROOT}/sct/outputs/sanity_check_gold_scored.json" \
    --alpha 0.1 \
    --top_k_out 20 \
    --model_name "google/gemma-2-9b-it" 2>&1

echo "=== Sanity Check: Token Recovery ==="
python3 "${PROJECT_ROOT}/sct/evaluation/token_recovery.py" \
    --scored_files "${PROJECT_ROOT}/sct/outputs/sanity_check_gold_scored.json" \
    --secret_words gold \
    --model_name "google/gemma-2-9b-it" \
    --k_values 5 20 2>&1

echo "=== Sanity Check Complete ==="
