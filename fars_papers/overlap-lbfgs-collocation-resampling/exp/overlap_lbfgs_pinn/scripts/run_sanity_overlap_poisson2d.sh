#!/bin/bash
set -e
PROJECT_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"
export WANDB_PROJECT="overlap-lbfgs-pinn"
export WANDB_MODE="offline"
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Sanity check: budget=500, adam_budget=250, seed=0"
python -m overlap_lbfgs_pinn.scripts.run_overlap_lbfgs_poisson2d \
    --budget 500 --adam_budget 250 --overlap_frac 0.5 --seeds 0
