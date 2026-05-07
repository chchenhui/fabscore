#!/bin/bash
# Quick sweep of gamma_lbfgs values to find optimal setting
set -e
PROJECT_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

export WANDB_MODE="offline"

echo "=== Gamma sweep: budget=15000, adam_budget=5000, seed=1 ==="

for GAMMA_LBFGS in 0.3 0.5 0.7 0.9; do
    echo ""
    echo "=================================================================="
    echo "  GAMMA_LBFGS=${GAMMA_LBFGS}"
    echo "=================================================================="
    python -m overlap_lbfgs_pinn.scripts.run_overlap_lbfgs_ice_shelf \
        --budget 15000 \
        --adam_budget 5000 \
        --overlap_frac 0.5 \
        --gamma_lbfgs "${GAMMA_LBFGS}" \
        --seeds 1
done

echo "Gamma sweep done"
