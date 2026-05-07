#!/bin/bash
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env
export HF_TOKEN=${HF_TOKEN}

python ${PROJ_DIR}/eacp/scripts/run_condition_e.py \
    --split MC \
    --subset_indices ${PROJ_DIR}/eacp/data/confiqa_mc_1500_subset_indices.json \
    --vectors_path ${PROJ_DIR}/eacp/steering/vectors/llama31_8b_contextfocus.pt \
    --output_dir ${PROJ_DIR}/eacp/outputs \
    --layer 13 \
    --prefill_only \
    --multiplier 0.5 \
    "$@"
