#!/bin/bash
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env
export HF_TOKEN=${HF_TOKEN}

python ${PROJ_DIR}/eacp/scripts/run_inference.py \
    --condition C \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split MC \
    "$@"
