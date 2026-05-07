#!/bin/bash

# Usage:
# ./run_auditor_sae_tokens.sh <data_dir> <auditor_model> <target_word> [output_dir]

set -e

DATA_DIR="$1"
AUDITOR_MODEL="$2"
TARGET_WORD="$3"
OUTPUT_DIR="${4:-$DATA_DIR}"

# Fixed parameters
MODE="sae_pos_tokens"
NUM_GUESSES=1
NUM_FEATURES=50
NUM_TOKENS_PER_FEATURE=5
BATCH_SIZE=50
TEMPERATURE=0.0
PROMPT_TEMPLATE="prompts/taboo/auditor_guess_sae_pos_tokens_single.txt"
# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory '$DATA_DIR' not found"
    exit 1
fi

# Find all JSON files
JSON_FILES=($(find "$DATA_DIR" -name "*.json" -type f))
if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo "❌ Error: No JSON files found in '$DATA_DIR'"
    exit 1
fi

echo "📊 Found ${#JSON_FILES[@]} JSON files to process"
mkdir -p "$OUTPUT_DIR"

for DATA_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$DATA_FILE")"

    python3 utils/guess_secret_word.py \
        --model_name "$AUDITOR_MODEL" \
        --data_file "$DATA_FILE" \
        --mode "$MODE" \
        --num_guesses $NUM_GUESSES \
        --num_features $NUM_FEATURES \
        --num_tokens $NUM_TOKENS_PER_FEATURE \
        --output_dir "$OUTPUT_DIR" \
        --template_file "$PROMPT_TEMPLATE" \
        --batch_size $BATCH_SIZE \
        --temperature $TEMPERATURE \
        --target_words "$TARGET_WORD"
done

echo "Auditor SAE Feature Descriptions Guessing completed!"
