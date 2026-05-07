#!/bin/bash
# Full MI decoding on HallusionBench with optimized hyperparams (4 GPUs).
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

export WANDB_MODE=offline

bash mi_decoding/scripts/run_mi_full.sh hallusionbench 4 512 adaptive_mi_optimized 0.005 5.0
