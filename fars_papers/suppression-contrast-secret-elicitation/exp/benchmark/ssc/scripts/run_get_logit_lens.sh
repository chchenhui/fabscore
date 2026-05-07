#!/bin/bash

# Usage:
#   ./run_get_logit_lens.sh <results_dir> <model_name> [output_dir]

set -e

RESULTS_DIR="$1"
MODEL_NAME="$2"
OUTPUT_DIR="${3:-ssc/results/logit_lens}"

TARGET_LAYER=50
TOP_K=10
MODE="ssc_tokens"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Error: Results directory '$RESULTS_DIR' not found"
    exit 1
fi

# Find all JSON files
JSON_FILES=($(find "$RESULTS_DIR" -name "*.json" -type f))
if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo "❌ Error: No JSON files found in '$RESULTS_DIR'"
    exit 1
fi

echo "📊 Found ${#JSON_FILES[@]} JSON files to process"
mkdir -p "$OUTPUT_DIR"

# Process files
SUCCESSFUL=0

for RESULTS_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$RESULTS_FILE")"

    python3 elicitation_methods/logit_lens.py \
        --data_file "$RESULTS_FILE" \
        --model_name "$MODEL_NAME" \
        --layer $TARGET_LAYER \
        --top_k $TOP_K \
        --mode "$MODE" \
        --output_dir "$OUTPUT_DIR"
done

echo "Logit lens extraction completed!"
