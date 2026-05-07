#!/bin/bash

# Usage:
# ./run_user_persona.sh <prompt_file> <persona_assistant_file> <persona_user_file> <model_name> [output_dir]

set -e

PROMPT_FILE="$1"
PERSONA_ASSISTANT_FILE="$2"
PERSONA_USER_FILE="$3"
MODEL_NAME="$4"
OUTPUT_DIR="${5:-taboo/results/user_persona_inference}"

# Fixed parameters
NUM_RESPONSES=1
MAX_NEW_TOKENS=100
TEMPERATURE=0.0
BATCH_SIZE=100
SEED=1

# Check if user prompts file exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ Error: Prompts file '$PROMPT_FILE' not found"
    exit 1
fi

# Check if persona assistant file exists
if [ ! -f "$PERSONA_ASSISTANT_FILE" ]; then
    echo "❌ Error: Persona assistant file '$PERSONA_ASSISTANT_FILE' not found"
    exit 1
fi

# Check if persona user file exists
if [ ! -f "$PERSONA_USER_FILE" ]; then
    echo "❌ Error: Persona user file '$PERSONA_USER_FILE' not found"
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
PROMPT_FILE="$(realpath "$PROMPT_FILE")"
PERSONA_ASSISTANT_FILE="$(realpath "$PERSONA_ASSISTANT_FILE")"
PERSONA_USER_FILE="$(realpath "$PERSONA_USER_FILE")"

python3 "$BATCH_INFERENCE_SCRIPT" \
    --user_persona \
    --prompts_file "$PROMPT_FILE" \
    --persona_assistant_file "$PERSONA_ASSISTANT_FILE" \
    --persona_user_file "$PERSONA_USER_FILE" \
    --model_name "$MODEL_NAME" \
    --num_responses $NUM_RESPONSES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --user_persona

echo "✅ User persona inference completed successfully!"
echo "📁 Results: $OUTPUT_DIR"
