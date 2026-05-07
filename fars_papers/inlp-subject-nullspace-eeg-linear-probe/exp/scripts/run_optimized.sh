#!/bin/bash
set -e
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/inlp-subject-nullspace-eeg-linear-probe/exp/.venv/bin/activate
export WANDB_MODE=offline
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/inlp-subject-nullspace-eeg-linear-probe/exp
python scripts/run_inlp_optimized.py
