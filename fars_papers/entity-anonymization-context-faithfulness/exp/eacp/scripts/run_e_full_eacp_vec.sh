#!/bin/bash
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env
export HF_TOKEN=${HF_TOKEN}

SCRIPT=${PROJ_DIR}/eacp/scripts/run_condition_e.py
COMMON="--split MC --subset_indices ${PROJ_DIR}/eacp/data/confiqa_mc_1500_subset_indices.json --output_dir ${PROJ_DIR}/eacp/outputs --layer 13"
VEC=${PROJ_DIR}/eacp/steering/vectors/llama31_8b_eacp_contextfocus.pt

echo "=== EACP vector: generation m=0.3 ==="
python $SCRIPT $COMMON --vectors_path $VEC --steer_mode generation --multiplier 0.3 \
    --output_suffix _eacp_gen_m0.3

echo ""
echo "=== EACP vector: generation m=0.5 ==="
python $SCRIPT $COMMON --vectors_path $VEC --steer_mode generation --multiplier 0.5 \
    --output_suffix _eacp_gen_m0.5

echo ""
echo "=== EACP vector: generation m=1.0 ==="
python $SCRIPT $COMMON --vectors_path $VEC --steer_mode generation --multiplier 1.0 \
    --output_suffix _eacp_gen_m1.0

echo ""
echo "=== EACP vector: both m=0.3 ==="
python $SCRIPT $COMMON --vectors_path $VEC --steer_mode both --multiplier 0.3 \
    --output_suffix _eacp_both_m0.3

echo ""
echo "=== EACP vector: both m=0.5 ==="
python $SCRIPT $COMMON --vectors_path $VEC --steer_mode both --multiplier 0.5 \
    --output_suffix _eacp_both_m0.5

echo ""
echo "=== All EACP vector full runs done ==="
