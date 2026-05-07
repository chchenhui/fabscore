#!/bin/bash
# Start vLLM server for TRACE evaluation.
# Usage: bash start_vllm.sh <model_path> [dp_size]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_DIR/.venv/bin/activate"

MODEL_PATH="${1:?Usage: $0 <model_path> [dp_size]}"
DP_SIZE="${2:-8}"

vllm serve "$MODEL_PATH" \
    --port 8001 \
    --data-parallel-size "$DP_SIZE" \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --trust-remote-code \
    --disable-log-requests
