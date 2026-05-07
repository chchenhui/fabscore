#!/bin/bash
# Launch Adam + resampling baseline on 2D Poisson forward problem.
# Usage: bash scripts/run_adam_resampling_poisson2d.sh [budget] [seeds]
set -e

BUDGET=${1:-30000}
SEEDS=${2:-"0,1,2"}

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp
source .venv/bin/activate

export WANDB_MODE=offline

python -m overlap_lbfgs_pinn.scripts.run_adam_resampling_poisson2d \
    --budget "$BUDGET" \
    --seeds "$SEEDS" \
    --device cuda
