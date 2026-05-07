#!/bin/bash
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp
source .venv/bin/activate
python -m overlap_lbfgs_pinn.scripts.run_adam_lbfgs_ice_shelf \
    --budget 50000 --adam_budget 20000 \
    --gamma 0.5 --seeds 0,1,2
