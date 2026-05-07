#!/bin/bash
# Startup script for best-of-2 parseability-filtered diffusion inference on BFCL-v3 Non-Live.
# Usage: bash run_best_of_2.sh <seed> [--limit N]

set -e

SEED=${1:?Usage: bash run_best_of_2.sh <seed> [--limit N]}
shift

EXP_ROOT=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/lave-tool-calling-bfcl/exp
source ${EXP_ROOT}/.venv/bin/activate

set -a
source ${EXP_ROOT}/.env
set +a

export PYTHONPATH=${EXP_ROOT}/CD4dLLM:${EXP_ROOT}:${PYTHONPATH}

echo "=== Best-of-2 Inference ==="
echo "Seed: ${SEED}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "Extra args: $@"
echo "==============================="

python ${EXP_ROOT}/bfcl_cfg_diffusion/inference/run_best_of_2.py \
    --seed ${SEED} \
    "$@"
