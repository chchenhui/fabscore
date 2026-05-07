#!/bin/bash

# Usage:
# ./run_many_shot_jailbreak.sh <prompts_file> <constraints_file> <prefix_file> <model_name> [output_dir]

set -e

PROMPTS_FILE="$1"
CONSTRAINTS_FILE="$2"
PREFIX_FILE="$3"
MODEL_NAME="$4"
OUTPUT_DIR="${5:-ssc/results/many_shot_jailbreak}"

# Fixed parameters
NUM_RESPONSES=4
MAX_NEW_TOKENS=256
TEMPERATURE=1.0
BATCH_SIZE=4
SEED=1

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ] || [ ! -f "$CONSTRAINTS_FILE" ]; then
    echo "❌ Error: Prompts file '$PROMPTS_FILE' not found"
    exit 1
fi

# Get script directory and batch inference path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_INFERENCE_SCRIPT="$SCRIPT_DIR/../../sampling/batch_inference.py"

if [ ! -f "$BATCH_INFERENCE_SCRIPT" ]; then
    echo "❌ Error: batch_inference.py not found at $BATCH_INFERENCE_SCRIPT"
    exit 1
fi

# Convert to absolute path
PROMPTS_FILE="$(realpath "$PROMPTS_FILE")"

python3 "$BATCH_INFERENCE_SCRIPT" \
    --prompts_file "$PROMPTS_FILE" \
    --constraints_file "$CONSTRAINTS_FILE" \
    --prefix_prompt_file "$PREFIX_FILE" \
    --model_name "$MODEL_NAME" \
    --num_responses $NUM_RESPONSES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --ssc \
    --multi_turn_jailbreak

echo "✅ Many shot jailbreak experiment completed successfully!"
echo "📁 Results: $OUTPUT_DIR"
