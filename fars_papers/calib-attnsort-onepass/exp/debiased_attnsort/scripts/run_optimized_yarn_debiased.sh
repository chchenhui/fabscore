#!/bin/bash
# Run optimized debiased k=1 on YaRN-Llama-2-7b-64k.
# Uses full-sort by debiased scores with divisive mode (a=0.005, B=40, mean).
set -e
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/calib-attnsort-onepass/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/calib-attnsort-onepass/exp

export WANDB_MODE=offline

SEED=${1:-42}
NUM_EXAMPLES=${2:-200}

python debiased_attnsort/src/eval_yarn.py \
    --mode debiased_k1 \
    --seed "$SEED" \
    --num_examples "$NUM_EXAMPLES"
