#!/bin/bash
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/trace-order2-fggm-migu-audit/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/trace-order2-fggm-migu-audit/exp

deepspeed --num_gpus 2 audit/scripts/debug_migu_grad.py \
    --config audit/configs/migu_default.yaml
