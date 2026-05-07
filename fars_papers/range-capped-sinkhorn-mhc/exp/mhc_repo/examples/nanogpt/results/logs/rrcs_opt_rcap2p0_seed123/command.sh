#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/range-capped-sinkhorn-mhc/exp/mhc_repo/examples/nanogpt
/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/range-capped-sinkhorn-mhc/exp/.venv/bin/python train.py config/train_fineweb10B_mhc_48l_rrcs_opt.py seed=123 mhc_r_cap=2.0 out_dir=results/logs/rrcs_opt_rcap2p0_seed123 wandb_run_name=rrcs_opt_rcap2p0_seed123 wandb_project=range-capped-sinkhorn-mhc max_iters=5000 diag_interval=10
