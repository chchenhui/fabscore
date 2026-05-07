#!/bin/bash
# Run magnitude-based channel statistics collection on GPU.
# Usage: bash fcboost/scripts/run_magnitude_analysis.sh [--sanity_check]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"
source .venv/bin/activate

export TORCH_CUDA_ARCH_LIST="8.0"

EXTRA_ARGS=""
if [[ "$1" == "--sanity_check" ]]; then
    EXTRA_ARGS="--sanity_check"
    echo "Running in SANITY CHECK mode"
fi

python -m fcboost.analysis.magnitude_analysis \
    --model Qwen/Qwen3-8B \
    --num_sequences 16 \
    --max_seq_len 8192 \
    --buffer_length 128 \
    --k_chan 16 \
    --output_dir fcboost/analysis \
    $EXTRA_ARGS
