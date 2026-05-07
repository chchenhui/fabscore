#!/bin/bash
# General ability evaluation for MIGU Order 2 final checkpoint via OpenCompass.
# Evaluates on MMLU, BBH, TyDiQA, PIQA, BoolQ, GSM8K.
# Usage: bash audit/scripts/run_migu_order2_general_eval.sh <seed> [num_gpus]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUDIT_DIR="$SCRIPT_DIR/.."

source "$PROJECT_DIR/.venv/bin/activate"
export PATH="$PROJECT_DIR/.venv/bin:$PATH"

SEED="${1:?Usage: $0 <seed> [num_gpus]}"
NUM_GPUS="${2:-8}"

MODEL_PATH="$AUDIT_DIR/results/migu_order2_seed${SEED}/checkpoints/7"
OUTPUT_DIR="$AUDIT_DIR/results/migu_order2_seed${SEED}/general_eval"
OPENCOMPASS_DIR="$AUDIT_DIR/external/opencompass"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Final checkpoint not found: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Model: $MODEL_PATH"
echo "Benchmarks: MMLU, BBH, TyDiQA, PIQA, BoolQ, GSM8K"
echo "GPUs: $NUM_GPUS"

cd "$OPENCOMPASS_DIR"
python run.py \
    --datasets mmlu_gen bbh_gen tydiqa_gen piqa_gen SuperGLUE_BoolQ_gen gsm8k_gen \
    --hf-path "$MODEL_PATH" \
    --hf-type chat \
    --hf-num-gpus "$NUM_GPUS" \
    --max-out-len 512 \
    --batch-size 16 \
    --work-dir "$OUTPUT_DIR/opencompass_output" \
    --reuse latest \
    2>&1

echo "General evaluation complete for seed $SEED"
echo "Results in: $OUTPUT_DIR/opencompass_output/"
