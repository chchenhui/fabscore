#!/bin/bash
# Run the full anisotropic experiment: attack + STS12 eval.
# Assumes masks are already trained (checkpoints exist).
set -e

PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp"
cd "$PROJ_DIR"
source .venv/bin/activate

export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)

echo "=== Phase 1: Mahalanobis Noise Verification ==="
python concept_leakage/noise/mahalanobis.py

echo ""
echo "=== Phase 2: Anisotropic Attack (fingerprint + template match) ==="
python concept_leakage/attack/run_anisotropic_attack.py

echo ""
echo "=== Phase 3: STS12 Utility Evaluation (anisotropic noise) ==="
python concept_leakage/evaluation/sts12_anisotropic_eval.py

echo ""
echo "=== All anisotropic experiment phases complete ==="
