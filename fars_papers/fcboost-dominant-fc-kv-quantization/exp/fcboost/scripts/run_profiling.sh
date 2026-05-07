#!/bin/bash
# Run CA profiling for FCBoost mask generation.
# Usage: bash fcboost/scripts/run_profiling.sh [--sanity_check]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
    --output_dir fcboost/masks \
    "$@"
