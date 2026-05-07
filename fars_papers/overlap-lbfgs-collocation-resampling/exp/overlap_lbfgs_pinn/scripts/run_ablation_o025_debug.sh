#!/bin/bash
# Dry-run sanity check for o=0.25 ablation with tiny budget.
set -e
PROJECT_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

export WANDB_PROJECT="overlap-lbfgs-pinn"
export WANDB_MODE="offline"

echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Starting dry-run: budget=500, adam_budget=200, adam_fixed_budget=100, overlap_frac=0.25"

python -m overlap_lbfgs_pinn.scripts.run_overlap_lbfgs_ice_shelf \
    --budget 500 \
    --adam_budget 200 \
    --adam_fixed_budget 100 \
    --overlap_frac 0.25 \
    --gamma 0.5 \
    --seeds 0 \
    --output_dir "${PROJECT_DIR}/overlap_lbfgs_pinn/outputs/overlap_lbfgs_ice_shelf_o025_debug" \
    --run_prefix "ablation_o025_debug"
