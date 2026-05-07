#!/bin/bash
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp
source .venv/bin/activate

echo "=== Token-presence sanity check: 1 concept, 1 condition, 1 seed, 5 epochs ==="
python concept_leakage/evaluation/token_presence.py \
    --conditions clean \
    --concepts weekdays \
    --seeds 42 \
    --n_epochs 5 \
    --results_dir concept_leakage/results/token_presence_sanity

echo "=== Sanity check complete ==="
cat concept_leakage/results/token_presence_sanity/summary.json
