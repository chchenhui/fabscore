#!/bin/bash
# Debug training: 1 GPU, 1 task (C-STANCE), 2 epochs
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

PORT=$(shuf -i 25000-30000 -n1)

deepspeed --num_gpus=1 --master_port $PORT \
    "$AUDIT_DIR/training/sequential_trainer.py" \
    --config "$AUDIT_DIR/configs/sft_debug.yaml"
