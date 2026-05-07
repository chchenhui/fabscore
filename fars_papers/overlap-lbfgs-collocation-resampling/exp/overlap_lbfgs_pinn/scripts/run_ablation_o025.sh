#!/bin/bash
# Ablation study: overlap-resampled L-BFGS with o=0.25 on ice-shelf problem.
# All settings identical to main experiment (o=0.5) except overlap_frac.
set -e
PROJECT_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

export WANDB_PROJECT="overlap-lbfgs-pinn"
export WANDB_MODE="offline"

echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Starting ablation o=0.25: budget=50000, adam_budget=20000, adam_fixed_budget=7500, gamma=0.5"

python -m overlap_lbfgs_pinn.scripts.run_overlap_lbfgs_ice_shelf \
    --budget "${BUDGET:-50000}" \
    --adam_budget "${ADAM_BUDGET:-20000}" \
    --adam_fixed_budget "${ADAM_FIXED_BUDGET:-7500}" \
    --overlap_frac 0.25 \
    --gamma 0.5 \
    --seeds "${SEEDS:-0,1,2}" \
    --output_dir "${PROJECT_DIR}/overlap_lbfgs_pinn/outputs/overlap_lbfgs_ice_shelf_o025" \
    --run_prefix "ablation_o025"
