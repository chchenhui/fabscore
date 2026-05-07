#!/bin/bash
# Smoothed covariance mitigation: generate smoothed sigmas, run attack, STS12 eval.
set -e

PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp"
cd "$PROJ_DIR"
source .venv/bin/activate

export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
export WANDB_MODE=offline

echo "============================================="
echo "=== Phase 1: Generate Smoothed Covariances ==="
echo "============================================="
python concept_leakage/noise/smoothed.py

echo ""
echo "============================================="
echo "=== Phase 2: Smoothed Attack (N=2) ==="
echo "============================================="
python concept_leakage/attack/run_smoothed_attack.py

echo ""
echo "============================================="
echo "=== Phase 3: STS12 Utility Evaluation ==="
echo "============================================="
python concept_leakage/evaluation/sts12_smoothed_eval.py

echo ""
echo "============================================="
echo "=== All smoothed experiment phases complete ==="
echo "============================================="
