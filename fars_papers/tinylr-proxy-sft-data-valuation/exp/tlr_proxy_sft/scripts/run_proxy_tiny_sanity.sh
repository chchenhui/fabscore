#!/bin/bash
# Sanity check: run one tiny-LR proxy config to verify training works.
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

LAUNCHER="${PROJECT_ROOT}/LlamaFactory/src/llamafactory/launcher.py"
CONFIG="${PROJECT_ROOT}/tlr_proxy_sft/configs/proxy_tiny_lr/mathplus_seed42.yaml"

echo "=== Sanity check: proxy_tiny mathplus_seed42 ==="
CUDA_VISIBLE_DEVICES=0 python "$LAUNCHER" "$CONFIG"
echo "=== Sanity check complete ==="
