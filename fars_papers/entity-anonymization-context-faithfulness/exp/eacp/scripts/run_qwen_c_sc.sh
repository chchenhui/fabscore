#!/bin/bash
# Condition C (EACP) with self-consistency on Qwen2.5-7B-Instruct
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env
export HF_TOKEN=${HF_TOKEN}

python ${PROJ_DIR}/eacp/scripts/run_selfconsistency.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --split MC \
    --n_samples 5 \
    --temperature 0.7 \
    --output_suffix sc \
    "$@"
