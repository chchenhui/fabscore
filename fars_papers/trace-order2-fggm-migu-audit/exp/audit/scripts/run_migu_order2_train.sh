#!/bin/bash
# Launch MIGU sequential training on TRACE Order 2 for a given seed.
# Usage: bash audit/scripts/run_migu_order2_train.sh <seed> [num_gpus]
# Example: bash audit/scripts/run_migu_order2_train.sh 42 8

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${1:?Usage: $0 <seed> [num_gpus]}"
NUM_GPUS="${2:-8}"
CONFIG="$AUDIT_DIR/configs/migu_order2_seed${SEED}.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

PORT=$(shuf -i 25000-30000 -n1)

echo "Starting MIGU Order 2 training (seed=$SEED) with config: $CONFIG"
echo "Num GPUs: $NUM_GPUS"
echo "DeepSpeed master port: $PORT"

deepspeed --num_gpus=$NUM_GPUS --master_port $PORT \
    "$AUDIT_DIR/training/sequential_trainer.py" \
    --config "$CONFIG"
