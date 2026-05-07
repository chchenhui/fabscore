#!/bin/bash
# Debug eval: single checkpoint, single GPU
set -euo pipefail

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp"
cd "$PROJECT_ROOT"

source .venv/bin/activate
set -a
source .env
set +a

MERGED_PATH="${PROJECT_ROOT}/tlr_proxy_sft/outputs/proxy_std/AM-Thinking-v1-Distilled-math/seed_42/merged"
RESULT_DIR="${PROJECT_ROOT}/tlr_proxy_sft/results/proxy_std/AM-Thinking-v1-Distilled-math/seed_42"
mkdir -p "$RESULT_DIR"

echo "=== Debug Eval ==="
echo "Model: $MERGED_PATH"
echo "Output: $RESULT_DIR"

lm_eval \
    --model vllm \
    --model_args "pretrained=${MERGED_PATH},dtype=bfloat16,tensor_parallel_size=1,data_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096" \
    --tasks gsm8k,minerva_math500 \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "$RESULT_DIR" \
    --log_samples

echo "=== Debug Eval Complete ==="
