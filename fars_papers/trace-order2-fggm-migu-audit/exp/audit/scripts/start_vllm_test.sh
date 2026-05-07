#!/bin/bash
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/trace-order2-fggm-migu-audit/exp/.venv/bin/activate
export PATH="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/trace-order2-fggm-migu-audit/exp/.venv/bin:$PATH"

CKPT_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/trace-order2-fggm-migu-audit/exp/audit/results/sft_default_seed42/checkpoints/1"

vllm serve "$CKPT_DIR" \
    --port 8001 \
    --data-parallel-size 2 \
    --max-model-len 2048 \
    --trust-remote-code \
    --disable-log-requests \
    --gpu-memory-utilization 0.85
