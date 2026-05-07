#!/bin/bash

# Usage:
# ./run_user_persona.sh <prompts_file> <constraints_file> <persona_assistant_file> <persona_user_file> <model_name> [output_dir]

set -e

PROMPTS_FILE="$1"
CONSTRAINTS_FILE="$2"
PERSONA_ASSISTANT_FILE="$3"
PERSONA_USER_FILE="$4"
MODEL_NAME="$5"
OUTPUT_DIR="${6:-ssc/results/user_persona}"

# Fixed parameters
NUM_RESPONSES=1
MAX_NEW_TOKENS=128
TEMPERATURE=0.0
BATCH_SIZE=100
SEED=1

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ] || [ ! -f "$CONSTRAINTS_FILE" ] || [ ! -f "$PERSONA_ASSISTANT_FILE" ] || [ ! -f "$PERSONA_USER_FILE" ]; then
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
    --persona_assistant_file "$PERSONA_ASSISTANT_FILE" \
    --persona_user_file "$PERSONA_USER_FILE" \
    --model_name "$MODEL_NAME" \
    --num_responses $NUM_RESPONSES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --ssc \
    --user_persona

echo "✅ User persona experiment completed successfully!"
echo "📁 Results: $OUTPUT_DIR"
