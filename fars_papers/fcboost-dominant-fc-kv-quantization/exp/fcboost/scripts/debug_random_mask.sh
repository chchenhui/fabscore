#!/bin/bash
# Debug dry-run: verify random mask integrates with FCBoostKVCache
set -e

PROJECT_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/fcboost-dominant-fc-kv-quantization/exp

cd "$PROJECT_DIR"
source .venv/bin/activate

export TORCH_CUDA_ARCH_LIST="8.0"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
export WANDB_MODE=offline

python -m fcboost.evaluation.eval_aime \
    --method fcboost_v2 \
    --model Qwen/Qwen3-8B \
    --task aime24 \
    --debug \
    --num_repeats 1 \
    --batch_size 1 \
    --max_new_tokens 32768 \
    --mask_path fcboost/masks/qwen3_8b_random_mask_seed42.pt \
    --results_dir ./eval_results_random_mask_debug
