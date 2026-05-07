#!/bin/bash
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/cusum-calibrated-rollback-controller/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/cusum-calibrated-rollback-controller/exp
export PYTHONPATH=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/cusum-calibrated-rollback-controller/exp:$PYTHONPATH
python cusum_controller/scripts/run_no_controller.py --num_seeds 20 --data_root ./data
