#!/bin/bash
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env
export HF_TOKEN=${HF_TOKEN}

LIMIT=100
SCRIPT=${PROJ_DIR}/eacp/scripts/run_condition_e.py
COMMON="--split MC --subset_indices ${PROJ_DIR}/eacp/data/confiqa_mc_1500_subset_indices.json --output_dir ${PROJ_DIR}/eacp/outputs --layer 13 --limit ${LIMIT}"

echo "=== NQ-SWAP vector, both, m=0.0 (baseline, no steering) ==="
python $SCRIPT $COMMON \
    --vectors_path ${PROJ_DIR}/eacp/steering/vectors/llama31_8b_contextfocus.pt \
    --steer_mode both --multiplier 0.0 \
    --output_suffix _sanity_nqswap_both_m0.0

echo ""
echo "=== NQ-SWAP vector, both, m=0.3 ==="
python $SCRIPT $COMMON \
    --vectors_path ${PROJ_DIR}/eacp/steering/vectors/llama31_8b_contextfocus.pt \
    --steer_mode both --multiplier 0.3 \
    --output_suffix _sanity_nqswap_both_m0.3

echo ""
echo "=== NQ-SWAP vector, both, m=0.5 ==="
python $SCRIPT $COMMON \
    --vectors_path ${PROJ_DIR}/eacp/steering/vectors/llama31_8b_contextfocus.pt \
    --steer_mode both --multiplier 0.5 \
    --output_suffix _sanity_nqswap_both_m0.5

echo ""
echo "=== NQ-SWAP vector, both, m=1.0 ==="
python $SCRIPT $COMMON \
    --vectors_path ${PROJ_DIR}/eacp/steering/vectors/llama31_8b_contextfocus.pt \
    --steer_mode both --multiplier 1.0 \
    --output_suffix _sanity_nqswap_both_m1.0

echo ""
echo "=== NQ-SWAP vector, generation only, m=0.5 ==="
python $SCRIPT $COMMON \
    --vectors_path ${PROJ_DIR}/eacp/steering/vectors/llama31_8b_contextfocus.pt \
    --steer_mode generation --multiplier 0.5 \
    --output_suffix _sanity_nqswap_gen_m0.5

echo ""
echo "=== NQ-SWAP vector, generation only, m=1.0 ==="
python $SCRIPT $COMMON \
    --vectors_path ${PROJ_DIR}/eacp/steering/vectors/llama31_8b_contextfocus.pt \
    --steer_mode generation --multiplier 1.0 \
    --output_suffix _sanity_nqswap_gen_m1.0

echo ""
echo "=== All sanity v2 checks done ==="
