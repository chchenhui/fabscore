#!/bin/bash
# Run conditions A, B, C on ConFiQA-QA 500-example subset (single-hop, single-conflict)
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env
export HF_TOKEN=${HF_TOKEN}

SUBSET=${PROJ_DIR}/eacp/data/confiqa_qa_500_subset_indices.json
MODEL=meta-llama/Llama-3.1-8B-Instruct

echo "=== Condition A on ConFiQA-QA (500 examples) ==="
python ${PROJ_DIR}/eacp/scripts/run_inference.py \
    --condition A --model ${MODEL} \
    --split QA --subset_indices ${SUBSET}

echo ""
echo "=== Condition B on ConFiQA-QA (500 examples) ==="
python ${PROJ_DIR}/eacp/scripts/run_inference.py \
    --condition B --model ${MODEL} \
    --split QA --subset_indices ${SUBSET}

echo ""
echo "=== Condition C on ConFiQA-QA (500 examples) ==="
python ${PROJ_DIR}/eacp/scripts/run_inference.py \
    --condition C --model ${MODEL} \
    --split QA --subset_indices ${SUBSET}

echo ""
echo "=== All conditions complete ==="
