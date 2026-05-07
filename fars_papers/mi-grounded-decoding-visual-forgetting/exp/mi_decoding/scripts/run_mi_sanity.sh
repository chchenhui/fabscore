#!/bin/bash
# Sanity check: run MI decoding on 50 MMStar items with optimized hyperparams.
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

export WANDB_MODE=offline

python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar \
    --max_items 50 \
    --max_new_tokens 512 \
    --lam 0.005 \
    --alpha 0.3 \
    --t0 0 \
    --max_weight 5.0 \
    --output_dir mi_decoding/outputs/sanity_check/mmstar
