#!/bin/bash
set -e

PROJECT_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

export WANDB_PROJECT="overlap-lbfgs-pinn"
export WANDB_MODE="offline"

echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Starting Adam->L-BFGS experiment: budget=${BUDGET:-30000}, adam_budget=${ADAM_BUDGET:-15000}, seeds=${SEEDS:-0,1,2}"

python -m overlap_lbfgs_pinn.scripts.run_adam_lbfgs_ice_shelf \
    --budget "${BUDGET:-30000}" \
    --adam_budget "${ADAM_BUDGET:-15000}" \
    --seeds "${SEEDS:-0,1,2}"
