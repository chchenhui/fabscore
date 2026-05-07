#!/bin/bash
# GSM8K evaluation - Kitty (2-bit, dynamic boost 12.5%)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"
source .venv/bin/activate

export TORCH_CUDA_ARCH_LIST="8.0"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
export WANDB_MODE=offline
export WANDB_PROJECT=fcboost-kv-quantization

python -m fcboost.evaluation.eval_aime \
    --method kitty \
    --model Qwen/Qwen3-8B \
    --task gsm8k_cot \
    --num_repeats 1 \
    --batch_size 1 \
    --max_new_tokens 4096 \
    --results_dir ./eval_results_gsm8k
