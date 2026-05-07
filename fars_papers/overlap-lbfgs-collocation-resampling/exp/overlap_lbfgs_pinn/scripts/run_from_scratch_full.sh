#!/bin/bash
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp
source .venv/bin/activate
set -a; source .env; set +a
export WANDB_PROJECT='overlap-lbfgs-pinn'

python -m overlap_lbfgs_pinn.scripts.run_from_scratch_overlap_lbfgs_poisson2d \
    --budget 50000 \
    --overlap_frac 0.5 \
    --lambda_bc 10.0 \
    --seeds 0,1,2
