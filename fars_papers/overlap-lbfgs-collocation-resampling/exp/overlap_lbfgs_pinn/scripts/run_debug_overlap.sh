#!/bin/bash
set -e
PROJECT_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

export WANDB_PROJECT="overlap-lbfgs-pinn"
export WANDB_MODE="offline"

echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=== DEBUG RUN: Overlap-LBFGS with small budget ==="

python -m overlap_lbfgs_pinn.scripts.run_overlap_lbfgs_ice_shelf \
    --budget 1000 \
    --adam_budget 500 \
    --overlap_frac 0.5 \
    --seeds 0
