#!/bin/bash
# Run the full isotropic noise baseline experiment (Phases 2-6).
# Phase 2 and 6 require GPU; Phases 3-5 are CPU-only but run inline.
set -e

PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp"
cd "$PROJ_DIR"
source .venv/bin/activate

echo "=== Phase 2: Embedding Generation ==="
python concept_leakage/data/embed_documents.py

echo ""
echo "=== Phase 3: Isotropic Noise Verification ==="
python concept_leakage/noise/isotropic.py

echo ""
echo "=== Phase 4: Fingerprint Computation ==="
python concept_leakage/attack/fingerprint.py

echo ""
echo "=== Phase 5: Template Matching ==="
python concept_leakage/attack/template_match.py

echo ""
echo "=== Phase 6: STS12 Utility Evaluation ==="
python concept_leakage/evaluation/sts12_eval.py

echo ""
echo "=== All phases complete ==="
