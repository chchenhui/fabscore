#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/range-capped-sinkhorn-mhc/exp/mhc_repo/examples/nanogpt
/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/range-capped-sinkhorn-mhc/exp/.venv/bin/python train.py config/train_fineweb10B_mhc_48l_capinit.py seed=42 out_dir=results/logs/capinit_seed42 wandb_run_name=capinit_seed42 wandb_project=range-capped-sinkhorn-mhc max_iters=5000 diag_interval=10
