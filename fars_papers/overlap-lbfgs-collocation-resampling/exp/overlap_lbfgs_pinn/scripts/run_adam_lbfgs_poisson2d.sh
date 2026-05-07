#!/bin/bash
. /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp
export WANDB_MODE=offline
export WANDB_PROJECT=overlap-lbfgs-pinn
python -m overlap_lbfgs_pinn.scripts.run_adam_lbfgs_poisson2d --budget 30000 --adam_budget 15000 --seeds 0,1,2
