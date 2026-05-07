#!/bin/bash
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sinkcast-bos-fp32-rope/exp
source .venv/bin/activate
export WANDB_MODE=offline
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/shared/huggingface
export HF_HUB_CACHE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/shared/huggingface/hub
set -a && source .env && set +a

python3 sinkcast/benchmarks/ruler_shift_sinkcast.py \
    --model mistral-7b-v0.3 \
    --seq_lengths 4096 8192 \
    --n_samples 50 \
    --shift_M 4096 \
    --K 1 \
    --output_dir sinkcast/results/downstream/sinkcast_k1_v2 \
    --seed 42
