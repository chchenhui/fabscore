#!/bin/bash
# Launch SFT sequential training on TRACE default order.
# Usage: bash audit/scripts/run_sft_train.sh [config_path] [num_gpus]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUDIT_DIR="$SCRIPT_DIR/.."

source "$PROJECT_DIR/.venv/bin/activate"

set -a
source "$PROJECT_DIR/.env"
set +a

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$PROJECT_DIR/wandb"

CONFIG="${1:-$AUDIT_DIR/configs/sft_default.yaml}"
NUM_GPUS="${2:-8}"

PORT=$(shuf -i 25000-30000 -n1)

echo "Starting SFT training with config: $CONFIG"
echo "Num GPUs: $NUM_GPUS"
echo "DeepSpeed master port: $PORT"

deepspeed --num_gpus=$NUM_GPUS --master_port $PORT \
    "$AUDIT_DIR/training/sequential_trainer.py" \
    --config "$CONFIG"
