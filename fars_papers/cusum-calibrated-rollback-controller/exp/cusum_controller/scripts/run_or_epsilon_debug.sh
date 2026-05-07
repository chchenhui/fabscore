#!/bin/bash
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/cusum-calibrated-rollback-controller/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/cusum-calibrated-rollback-controller/exp
export PYTHONPATH=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/cusum-calibrated-rollback-controller/exp:$PYTHONPATH
export WANDB_MODE=offline
python cusum_controller/scripts/run_or_epsilon.py --debug --data_root ./data --num_gpus 1
