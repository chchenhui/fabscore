#!/bin/bash
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp

export WANDB_MODE=offline

python -m overlap_lbfgs_pinn.scripts.run_overlap_lbfgs_poisson2d \
    --budget 10000 \
    --adam_budget 4000 \
    --adam_fixed_budget 2000 \
    --overlap_frac 0.9 \
    --lambda_bc 1.0 \
    --lambda_bc_fixed 5.0 \
    --lambda_bc_lbfgs 10.0 \
    --seeds 0 \
    --device cuda
