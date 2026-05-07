#!/bin/bash
# Improved CA profiling: 16 sequences, 8192 tokens, 512 sample positions
# Produces higher-quality masks by reducing CA score noise at selection boundary
set -e

PROJECT_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/fcboost-dominant-fc-kv-quantization/exp

cd "$PROJECT_DIR"
source .venv/bin/activate

export TORCH_CUDA_ARCH_LIST="8.0"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

python -m fcboost.profiling.run_profiling \
    --model Qwen/Qwen3-8B \
    --num_sequences 16 \
    --max_seq_len 8192 \
    --topk 256 \
    --max_sample_tokens 512 \
    --top_f 8 \
    --output_dir fcboost/masks_v2
