#!/bin/bash
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env
export HF_TOKEN=${HF_TOKEN}

python ${PROJ_DIR}/eacp/scripts/run_selfconsistency.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split MC \
    --n_samples 5 \
    --temperature 0.7 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.92 \
    --output_suffix sc_full
