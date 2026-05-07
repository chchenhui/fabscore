#!/bin/bash
# Usage:
#   ./run_inference.sh <prompts_file> <constraints_file> <model_name> [output_dir]

set -e

PROMPTS_FILE="$1"
CONSTRAINTS_FILE="$2"
MODEL_NAME="$3"
OUTPUT_DIR="${4:-ssc/results/inference}"

# Fixed parameters
NUM_RESPONSES=4
MAX_NEW_TOKENS=1024
TEMPERATURE=1.0
BATCH_SIZE=40
SEED=1

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ] || [ ! -f "$CONSTRAINTS_FILE" ]; then
    echo "❌ Error: Prompts file '$PROMPTS_FILE' or constraints file '$CONSTRAINTS_FILE' not found"
    exit 1
fi

if [ ! -f "$CONSTRAINTS_FILE" ]; then
    echo "❌ Error: Constraints file '$CONSTRAINTS_FILE' not found"
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

# Execute batch inference
echo "🚀 Running batch inference..."
python3 "$BATCH_INFERENCE_SCRIPT" \
    --prompts_file "$PROMPTS_FILE" \
    --constraints_file "$CONSTRAINTS_FILE" \
    --model_name "$MODEL_NAME" \
    --num_responses $NUM_RESPONSES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --ssc \

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Batch inference completed successfully!"
    echo "📁 Results: $OUTPUT_DIR"
else
    echo "❌ Batch inference failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi
