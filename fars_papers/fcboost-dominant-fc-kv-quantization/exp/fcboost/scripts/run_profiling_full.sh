#!/bin/bash
# Full CA profiling (4 sequences, 4096 tokens each)
set -e

PROJECT_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/fcboost-dominant-fc-kv-quantization/exp

cd "$PROJECT_DIR"
source .venv/bin/activate

export TORCH_CUDA_ARCH_LIST="8.0"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

python -m fcboost.profiling.run_profiling \
    --model Qwen/Qwen3-8B \
    --num_sequences 4 \
    --max_seq_len 4096 \
    --topk 256 \
    --max_sample_tokens 256 \
    --top_f 8 \
    --output_dir fcboost/masks
