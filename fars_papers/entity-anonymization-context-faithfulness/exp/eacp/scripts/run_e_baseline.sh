#!/bin/bash
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env
export HF_TOKEN=${HF_TOKEN}

SCRIPT=${PROJ_DIR}/eacp/scripts/run_condition_e.py
COMMON="--split MC --subset_indices ${PROJ_DIR}/eacp/data/confiqa_mc_1500_subset_indices.json --output_dir ${PROJ_DIR}/eacp/outputs --layer 13"

echo "=== Baseline: m=0.0 (no steering) ==="
python $SCRIPT $COMMON \
    --vectors_path ${PROJ_DIR}/eacp/steering/vectors/llama31_8b_contextfocus.pt \
    --steer_mode both --multiplier 0.0 \
    --output_suffix _baseline_m0.0

echo "=== Baseline done ==="
