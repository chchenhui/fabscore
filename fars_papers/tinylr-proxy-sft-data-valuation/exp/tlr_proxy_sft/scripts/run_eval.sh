#!/bin/bash
# Evaluate a merged checkpoint on GSM8K and MATH-500 using lm-eval + vLLM.
# Usage: bash run_eval.sh <merged_model_path> <output_dir>
# Designed to run via TrainService with 8 GPUs.
set -euo pipefail

MERGED_PATH="$1"
OUTPUT_DIR="$2"

if [ -z "$MERGED_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <merged_model_path> <output_dir>"
    exit 1
fi

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp"
cd "$PROJECT_ROOT"

source .venv/bin/activate
set -a
source .env
set +a

mkdir -p "$OUTPUT_DIR"

echo "=== Evaluation ==="
echo "Model: $MERGED_PATH"
echo "Output: $OUTPUT_DIR"
echo "Python: $(which python)"
echo "==================="

NPROC=$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs detected: $NPROC"

TP_SIZE=4
if [ "$NPROC" -lt 4 ]; then
    TP_SIZE=$NPROC
fi
DP_SIZE=$((NPROC / TP_SIZE))
echo "tensor_parallel_size: $TP_SIZE, data_parallel_size: $DP_SIZE"

lm_eval \
    --model vllm \
    --model_args "pretrained=${MERGED_PATH},dtype=bfloat16,tensor_parallel_size=${TP_SIZE},data_parallel_size=${DP_SIZE},gpu_memory_utilization=0.9,max_model_len=4096" \
    --tasks gsm8k,minerva_math500 \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "$OUTPUT_DIR" \
    --log_samples

echo "=== Evaluation Complete ==="
