#!/bin/bash
# Full optimized overlap-LBFGS run: 3 seeds, budget=50000, gamma_lbfgs=0.7
set -e
PROJECT_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

export WANDB_MODE="offline"

echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Full optimized run: budget=50000, adam_budget=15000, gamma_lbfgs=0.7, overlap_frac=0.5, seeds=0,1,2"

python -m overlap_lbfgs_pinn.scripts.run_overlap_lbfgs_ice_shelf \
    --budget 50000 \
    --adam_budget 15000 \
    --overlap_frac 0.5 \
    --gamma_lbfgs 0.7 \
    --seeds 0,1,2

echo "Full optimized run completed"
