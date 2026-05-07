#!/bin/bash
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp
source .venv/bin/activate

echo "=== Token-Presence Probe: Full Experiment ==="
echo "4 conditions x 5 concepts x 3 seeds = 60 runs, 50 epochs each"

python concept_leakage/evaluation/token_presence.py \
    --conditions clean isotropic anisotropic smoothed \
    --concepts weekdays months countries gender cities \
    --seeds 42 123 456 \
    --n_epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --results_dir concept_leakage/results/token_presence

echo "=== Token-Presence Probe: Full Experiment Complete ==="
