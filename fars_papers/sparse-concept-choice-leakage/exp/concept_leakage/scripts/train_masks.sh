#!/bin/bash
# Full mask training: embed train/val data, then train 5 concepts x 3 seeds x 100 epochs
set -e

PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp"
cd "$PROJ_DIR"
source .venv/bin/activate

export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
export WANDB_MODE=offline

echo "=== Step 1: Embed train/val splits (skip if already done) ==="
python concept_leakage/masks/embed_train_data.py

echo ""
echo "=== Step 2: Train all concept masks ==="
for concept in weekdays months countries gender cities; do
  for seed in 42 123 456; do
    echo ""
    echo "--- Training: concept=$concept, seed=$seed ---"
    python concept_leakage/masks/train_concept_mask.py \
      --concept "$concept" --seed "$seed" --epochs 100 --lr 1e-4 \
      --batch_size 64 --lambda_l0 0.001 --eval_every 5
  done
done

echo ""
echo "=== All mask training complete ==="

echo ""
echo "=== Summary of trained masks ==="
for concept in weekdays months countries gender cities; do
  for seed in 42 123 456; do
    meta="$PROJ_DIR/concept_leakage/checkpoints/$concept/seed$seed/meta.json"
    if [ -f "$meta" ]; then
      echo "$concept/seed$seed: $(cat $meta | python -c 'import json,sys; d=json.load(sys.stdin); print(f\"sparsity={d[\"mask_sparsity\"]:.4f}, active={d[\"active_dims\"]}, best_epoch={d[\"best_epoch\"]}, val_loss={d[\"best_val_loss\"]:.4f}\")')"
    fi
  done
done
