#!/bin/bash
# Run FCBoost evaluation on AIME25 (Qwen3-8B, 3 repeats, 32k gen)
set -e

PROJECT_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/fcboost-dominant-fc-kv-quantization/exp

cd "$PROJECT_DIR"
source .venv/bin/activate

export TORCH_CUDA_ARCH_LIST="8.0"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

python -m fcboost.evaluation.eval_aime \
    --method fcboost \
    --model Qwen/Qwen3-8B \
    --task aime25 \
    --num_repeats 3 \
    --batch_size 1 \
    --max_new_tokens 32768 \
    --results_dir ./eval_results
