#!/bin/bash
# Debug sanity check: embed train/val data, then train 1 concept x 1 seed x 5 epochs
set -e

PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp"
cd "$PROJ_DIR"
source .venv/bin/activate

export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
export WANDB_MODE=offline

echo "=== Step 1: Embed train/val splits ==="
python concept_leakage/masks/embed_train_data.py

echo ""
echo "=== Step 2: Debug mask training (countries, seed=42, 5 epochs) ==="
python concept_leakage/masks/train_concept_mask.py \
  --concept countries --seed 42 --epochs 5 --lr 1e-4 \
  --batch_size 64 --lambda_l0 0.001 --eval_every 1

echo ""
echo "=== Debug sanity check complete ==="
