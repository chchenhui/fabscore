#!/bin/bash

# Usage:
# ./run_get_sae_features.sh <results_dir> <features_file> <model_name> [output_dir]

set -e

RESULTS_DIR="$1"
FEATURES_FILE="$2"
MODEL_NAME="$3"
TARGET_LAYER=50
TOP_K=10
OUTPUT_DIR="${4:-ssc/results/sae_features}"
TOP_K_TOKENS=20
MODE="ssc_per_token"
USE_TFIDF=true
SAE_NAME="Llama-3.3-70B-Instruct-SAE-l50"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory '$RESULTS_DIR' not found"
    exit 1
fi

# Check if features file exists
if [ ! -f "$FEATURES_FILE" ]; then
    echo "Error: Features file '$FEATURES_FILE' not found"
    exit 1
fi

# Find all JSON files
JSON_FILES=($(find "$RESULTS_DIR" -name "*.json" -type f))
if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo "Error: No JSON files found in '$RESULTS_DIR'"
    exit 1
fi

echo "Found ${#JSON_FILES[@]} JSON files to process"
mkdir -p "$OUTPUT_DIR"

for RESULTS_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$RESULTS_FILE")"

    python3 elicitation_methods/sae.py \
        --model_name "$MODEL_NAME" \
        --data_file "$RESULTS_FILE" \
        --features_file "$FEATURES_FILE" \
        --layer $TARGET_LAYER \
        --top_k $TOP_K \
        --top_k_tokens $TOP_K_TOKENS \
        --mode "$MODE" \
        --output_dir "$OUTPUT_DIR" \
        --sae_name "$SAE_NAME" \
        --use_tfidf;
done

echo "SAE features extraction completed!"
