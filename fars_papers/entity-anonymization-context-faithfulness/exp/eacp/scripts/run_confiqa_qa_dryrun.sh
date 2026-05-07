#!/bin/bash
# Dry-run: test A/B/C on 5 QA examples to validate prompt building and scoring
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env
export HF_TOKEN=${HF_TOKEN}

SUBSET=${PROJ_DIR}/eacp/data/confiqa_qa_500_subset_indices.json

echo "=== Dry-run: Condition A on QA (5 examples) ==="
python ${PROJ_DIR}/eacp/scripts/run_inference.py \
    --condition A --model meta-llama/Llama-3.1-8B-Instruct \
    --split QA --subset_indices ${SUBSET} --limit 5

echo ""
echo "=== Dry-run: Condition B on QA (5 examples) ==="
python ${PROJ_DIR}/eacp/scripts/run_inference.py \
    --condition B --model meta-llama/Llama-3.1-8B-Instruct \
    --split QA --subset_indices ${SUBSET} --limit 5

echo ""
echo "=== Dry-run: Condition C on QA (5 examples) ==="
python ${PROJ_DIR}/eacp/scripts/run_inference.py \
    --condition C --model meta-llama/Llama-3.1-8B-Instruct \
    --split QA --subset_indices ${SUBSET} --limit 5

echo ""
echo "=== Dry-run complete ==="
