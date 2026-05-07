#!/bin/bash

# Usage:
# ./run_many_shot_jailbreak.sh <prefix_file> <prompt_file> <model_name> [output_dir]

set -e

PREFIX_FILE="$1"
PROMPT_FILE="$2"
MODEL_NAME="$3"
OUTPUT_DIR="${4:-taboo/results/many_shot_jailbreak}"

# Fixed parameters
NUM_RESPONSES=10
MAX_NEW_TOKENS=200
TEMPERATURE=1.0
BATCH_SIZE=10
SEED=1

# Check if files exist
if [ ! -f "$PREFIX_FILE" ]; then
    echo "❌ Error: Prefix file '$PREFIX_FILE' not found"
    exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ Error: Prompts file '$PROMPT_FILE' not found"
    exit 1
fi

# Get script directory and batch inference path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_INFERENCE_SCRIPT="$SCRIPT_DIR/../../sampling/batch_inference.py"

if [ ! -f "$BATCH_INFERENCE_SCRIPT" ]; then
    echo "❌ Error: batch_inference.py not found at $BATCH_INFERENCE_SCRIPT"
    exit 1
fi

# Convert to absolute paths
PREFIX_FILE="$(realpath "$PREFIX_FILE")"
PROMPT_FILE="$(realpath "$PROMPT_FILE")"

# Execute multi-turn jailbreak inference
python3 "$BATCH_INFERENCE_SCRIPT" \
    --multi_turn_jailbreak \
    --prefix_prompt_file "$PREFIX_FILE" \
    --prompts_file "$PROMPT_FILE" \
    --model_name "$MODEL_NAME" \
    --num_responses $NUM_RESPONSES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE

echo "✅ Multi-turn jailbreak experiment completed successfully!"
echo "📁 Results: $OUTPUT_DIR"
