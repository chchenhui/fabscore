#!/bin/bash
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env
export HF_TOKEN=${HF_TOKEN}

echo "=== Running Condition C: Full 6000 examples ==="
python ${PROJ_DIR}/eacp/scripts/run_inference.py \
    --condition C \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split MC

echo ""
echo "=== Running Condition C: 1500 subset ==="
python ${PROJ_DIR}/eacp/scripts/run_inference.py \
    --condition C \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split MC \
    --subset_indices ${PROJ_DIR}/eacp/data/confiqa_mc_1500_subset_indices.json

echo ""
echo "=== All Condition C runs complete ==="
