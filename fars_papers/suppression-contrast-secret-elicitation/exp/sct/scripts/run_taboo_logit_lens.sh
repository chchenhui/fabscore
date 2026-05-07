#!/bin/bash
# Run activation extraction for a single Taboo model organism.
# Usage: bash run_taboo_logit_lens.sh <model_name> <secret_word> [max_prompts]
# Example: bash run_taboo_logit_lens.sh bcywinski/gemma-2-9b-it-taboo-gold gold
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${PROJECT_ROOT}/.venv/bin/activate"

source "$VENV"

MODEL_NAME="${1:?Usage: $0 <model_name> <secret_word> [max_prompts]}"
SECRET_WORD="${2:?Usage: $0 <model_name> <secret_word> [max_prompts]}"
MAX_PROMPTS="${3:-}"

PROMPTS_FILE="${PROJECT_ROOT}/benchmark/prompts/taboo/taboo_direct_test.txt"
OUTPUT_DIR="${PROJECT_ROOT}/sct/outputs"
OUTPUT_PATH="${OUTPUT_DIR}/taboo_${SECRET_WORD}_activations.json"

EXTRA_ARGS=""
if [ -n "$MAX_PROMPTS" ]; then
    EXTRA_ARGS="--max_prompts $MAX_PROMPTS"
fi

echo "=== Activation Extraction ==="
echo "Model: $MODEL_NAME"
echo "Secret: $SECRET_WORD"
echo "Output: $OUTPUT_PATH"

python3 "${PROJECT_ROOT}/sct/extraction/extract_activations.py" \
    --model_name "$MODEL_NAME" \
    --prompts_file "$PROMPTS_FILE" \
    --output_path "$OUTPUT_PATH" \
    --mid_layer 32 \
    --final_layer 41 \
    --top_k 200 \
    --num_responses 10 \
    --max_new_tokens 200 \
    --temperature 1.0 \
    --seed 1 \
    --batch_size 100 \
    $EXTRA_ARGS

echo "=== Done ==="
