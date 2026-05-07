#!/bin/bash
# Deploy Gemma-3-4B-IT as auditor model via vLLM for the logit lens evaluation.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp/.venv/bin/activate

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp
if [ -f "$PROJ_DIR/.env" ]; then
    set -a
    source "$PROJ_DIR/.env"
    set +a
fi

export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/.cache/huggingface

vllm serve google/gemma-3-4b-it \
    --port 8001 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9
