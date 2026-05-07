#!/bin/bash
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env
export HF_TOKEN=${HF_TOKEN}

python ${PROJ_DIR}/eacp/steering/compute_eacp_vector.py \
    --pairs_path ${PROJ_DIR}/eacp/steering/data/eacp_steering_pairs.json \
    --output_dir ${PROJ_DIR}/eacp/steering/vectors \
    --batch_size 4
