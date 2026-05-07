#!/bin/bash
# Optimization iteration 1: gamma=0.5, 3-phase training (Adam+resample -> Adam+fixed -> overlap-LBFGS)
set -e

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp
source .venv/bin/activate

export WANDB_MODE=offline

python -m overlap_lbfgs_pinn.scripts.run_overlap_lbfgs_ice_shelf \
    --budget 50000 \
    --adam_budget 20000 \
    --adam_fixed_budget 7500 \
    --overlap_frac 0.5 \
    --gamma 0.5 \
    --seeds 0,1,2 \
    --device cuda

echo "=== Experiment complete ==="
