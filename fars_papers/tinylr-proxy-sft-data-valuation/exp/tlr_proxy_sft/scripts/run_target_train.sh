#!/bin/bash
# Run a single target model (Qwen2.5-7B) LoRA SFT training job via LLaMA-Factory.
# Uses venv python + torch.distributed.run to avoid system torchrun picking up wrong packages.
# Usage: bash run_target_train.sh <config.yaml>
set -euo pipefail

CONFIG_PATH="$1"
if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: $0 <config.yaml>"
    exit 1
fi

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
NPROC=$(python -c "import torch; print(torch.cuda.device_count())")
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "=== Target Training ==="
echo "Config: $CONFIG_PATH"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_MODE: $WANDB_MODE"
echo "Python: $(which python)"
echo "GPUs detected: $NPROC"
echo "MASTER_PORT: $MASTER_PORT"
echo "========================"

python -m torch.distributed.run \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node "$NPROC" \
    --master_addr localhost \
    --master_port "$MASTER_PORT" \
    "$LAUNCHER" "$CONFIG_PATH"
