#!/bin/bash
# Sanity check: train one proxy std config on a single GPU to validate setup.
set -euo pipefail

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp"
cd "$PROJECT_ROOT"

source .venv/bin/activate
set -a
source .env
set +a

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-tinylr-proxy-sft-data-valuation}"
export PYTHONPATH="${PROJECT_ROOT}/LlamaFactory/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK=1
export TRITON_CACHE_DIR="/tmp/triton_cache"
export TRITON_HOME="/tmp/triton_home"
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_HOME"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

CONFIG_PATH="${PROJECT_ROOT}/tlr_proxy_sft/configs/proxy_std_lr/AM-Thinking-v1-Distilled-math_seed42.yaml"

echo "=== Proxy Std Sanity Check ==="
echo "Config: $CONFIG_PATH"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_MODE: $WANDB_MODE"
echo "Python: $(which python)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================="

LAUNCHER="${PROJECT_ROOT}/LlamaFactory/src/llamafactory/launcher.py"
python "$LAUNCHER" "$CONFIG_PATH"

echo "=== Sanity Check Complete ==="
