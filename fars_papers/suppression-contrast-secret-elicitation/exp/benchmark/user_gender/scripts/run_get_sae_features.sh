#!/bin/bash

# Usage:
# ./run_get_sae_features.sh <data_dir> <features_file> <model_name> [output_dir]

set -e

DATA_DIR="$1"
FEATURES_FILE="$2"
MODEL_NAME="$3"
TARGET_LAYER=23
TOP_K=200
OUTPUT_DIR="${4:-user_gender/results/sae_features}"
TOP_K_TOKENS=20
MODE="first_person_pronouns"
USE_TFIDF=true
PROMPT_TEMPLATE="prompts/gender/auditor_filter_sae_descriptions.txt"
CACHE_FILE="sae_feature_cache.json"
TEMPERATURE=0.0
MAX_NEW_TOKENS=200
BATCH_SIZE=50
FEATURES_PER_PROMPT=50
LAYER=23

# Check if results directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Results directory '$DATA_DIR' not found"
    exit 1
fi

# Check if features file exists
if [ ! -f "$FEATURES_FILE" ]; then
    echo "Error: Features file '$FEATURES_FILE' not found"
    exit 1
fi

# Find all JSON files
JSON_FILES=($(find "$DATA_DIR" -name "*.json" -type f))
if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo "Error: No JSON files found in '$DATA_DIR'"
    exit 1
fi

echo "Found ${#JSON_FILES[@]} JSON files to process"
mkdir -p "$OUTPUT_DIR"

for DATA_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$DATA_FILE")"

    python3 elicitation_methods/sae.py \
        --model_name "$MODEL_NAME" \
        --data_file "$DATA_FILE" \
        --features_file "$FEATURES_FILE" \
        --layer $TARGET_LAYER \
        --top_k $TOP_K \
        --top_k_tokens $TOP_K_TOKENS \
        --mode "$MODE" \
        --output_dir "$OUTPUT_DIR" \
        --use_tfidf;

    # Get all SAE feature files in the output directory
    SAE_FEATURE_FILES=($(find "$OUTPUT_DIR" -name "*_sae_features*" -type f))

    if [ ${#SAE_FEATURE_FILES[@]} -eq 0 ]; then
        echo "Error: No SAE feature files found in '$OUTPUT_DIR'"
        exit 1
    fi

    echo "Found ${#SAE_FEATURE_FILES[@]} SAE feature files to process"

    for OUTPUT_FILE in "${SAE_FEATURE_FILES[@]}"; do
        echo "  Processing SAE features: $(basename "$OUTPUT_FILE")"

        python user_gender/filter_gender_relevant_features.py \
            --input_file "$OUTPUT_FILE" \
            --model_name "$MODEL_NAME" \
            --output_dir "$OUTPUT_DIR"_filtered \
            --prompt_template "$PROMPT_TEMPLATE" \
            --batch_size $BATCH_SIZE \
            --features_per_prompt $FEATURES_PER_PROMPT \
            --layer_idx $LAYER \
            --temperature $TEMPERATURE \
            --max_new_tokens $MAX_NEW_TOKENS \
            --cache_file "$CACHE_FILE"
    done
done

echo "SAE features extraction completed!"
