#!/bin/bash
# Sanity check for CA profiling (1 seq, 512 tokens)
set -e

PROJECT_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/fcboost-dominant-fc-kv-quantization/exp

cd "$PROJECT_DIR"
source .venv/bin/activate

export TORCH_CUDA_ARCH_LIST="8.0"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

python -m fcboost.profiling.run_profiling \
    --model Qwen/Qwen3-8B \
    --sanity_check \
    --output_dir fcboost/masks_sanity
