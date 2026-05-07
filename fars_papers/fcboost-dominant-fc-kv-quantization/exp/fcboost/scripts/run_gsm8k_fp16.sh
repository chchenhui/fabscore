#!/bin/bash
# GSM8K evaluation - FP16 baseline (1319 test problems, 8-shot CoT, max_gen=4096)
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
    --method fp16 \
    --model Qwen/Qwen3-8B \
    --task gsm8k_cot \
    --num_repeats 1 \
    --batch_size 1 \
    --max_new_tokens 4096 \
    --results_dir ./eval_results_gsm8k
