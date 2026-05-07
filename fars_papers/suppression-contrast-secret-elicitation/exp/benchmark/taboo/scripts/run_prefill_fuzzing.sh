#!/bin/bash

# Usage:
# ./run_prefill_fuzzing.sh <prompt_file> <prefill_file> <model_name> [output_dir]

set -e

PROMPT_FILE="$1"
PREFILL_FILE="$2"
MODEL_NAME="$3"
OUTPUT_DIR="${4:-taboo/results/prefill_fuzzing_inference}"

# Fixed parameters
NUM_RESPONSES=1
MAX_NEW_TOKENS=100
TEMPERATURE=0.0
BATCH_SIZE=100
SEED=1
FUZZ_LAYER_IDX=32
FUZZ_SEED=2
NOISE_MAGNITUDE=10.0

# Check if user prompts file exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ Error: Prompts file '$PROMPT_FILE' not found"
    exit 1
fi

# Check if prefill file exists
if [ ! -f "$PREFILL_FILE" ]; then
    echo "❌ Error: Prefill file '$PREFILL_FILE' not found"
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
PREFILL_FILE="$(realpath "$PREFILL_FILE")"

# Execute prefill inference with fuzzing
python3 "$BATCH_INFERENCE_SCRIPT" \
    --prompts_file "$PROMPT_FILE" \
    --prefill_file "$PREFILL_FILE" \
    --model_name "$MODEL_NAME" \
    --enable_fuzzing \
    --num_responses $NUM_RESPONSES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --noise_magnitude $NOISE_MAGNITUDE \
    --fuzz_layer_idx $FUZZ_LAYER_IDX \
    --fuzz_seed $FUZZ_SEED \
    --prefill

echo "✅ Prefill inference with fuzzing completed successfully!"
echo "📁 Results: $OUTPUT_DIR"
