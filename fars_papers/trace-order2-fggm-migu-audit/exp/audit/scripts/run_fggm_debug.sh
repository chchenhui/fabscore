#!/bin/bash
# Debug FGGM: captures all output for debugging
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUDIT_DIR="$SCRIPT_DIR/.."

source "$PROJECT_DIR/.venv/bin/activate"

set -a
source "$PROJECT_DIR/.env"
set +a

export WANDB_MODE=offline
export WANDB_DIR="$PROJECT_DIR/wandb"
export PYTHONUNBUFFERED=1

CONFIG="${1:-$AUDIT_DIR/configs/fggm_debug.yaml}"
NUM_GPUS="${2:-1}"

PORT=$(shuf -i 25000-30000 -n1)

LOG_FILE="$AUDIT_DIR/results/fggm_debug/debug_output.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Starting FGGM debug training with config: $CONFIG"
echo "Num GPUs: $NUM_GPUS"
echo "Log file: $LOG_FILE"

deepspeed --num_gpus=$NUM_GPUS --master_port $PORT \
    "$AUDIT_DIR/training/sequential_trainer.py" \
    --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
