#!/bin/bash
# Optimized experiment: retrain masks with stronger L0, run attack with N=10 releases.
# Fixes: (1) lambda_l0 0.001->0.1 for sparsity, (2) N=10 multi-release, (3) group_size_min=30
set -e

PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp"
cd "$PROJ_DIR"
source .venv/bin/activate

export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
export WANDB_MODE=offline

OPT_CKPT_DIR="$PROJ_DIR/concept_leakage/checkpoints_opt"
OPT_RESULTS_DIR="$PROJ_DIR/concept_leakage/results_opt"
mkdir -p "$OPT_CKPT_DIR" "$OPT_RESULTS_DIR"

echo "============================================="
echo "=== Phase 1: Embed train/val (skip if exists) ==="
echo "============================================="
python concept_leakage/masks/embed_train_data.py

echo ""
echo "============================================="
echo "=== Phase 2: Train masks with lambda_l0=0.1 ==="
echo "============================================="
LAMBDA_L0=0.1
EPOCHS=200
LR=3e-4
SEED=42

for concept in weekdays months countries gender cities; do
  echo ""
  echo "--- Training: concept=$concept, seed=$SEED, lambda_l0=$LAMBDA_L0 ---"
  python concept_leakage/masks/train_concept_mask.py \
    --concept "$concept" --seed "$SEED" --epochs "$EPOCHS" --lr "$LR" \
    --batch_size 64 --lambda_l0 "$LAMBDA_L0" --eval_every 5
done

echo ""
echo "=== Moving optimized checkpoints ==="
for concept in weekdays months countries gender cities; do
  src="$PROJ_DIR/concept_leakage/checkpoints/$concept/seed${SEED}"
  dst="$OPT_CKPT_DIR/$concept/seed${SEED}"
  mkdir -p "$dst"
  cp "$src"/mask.npy "$src"/sigma.npy "$src"/model.pt "$src"/meta.json "$dst"/
  echo "  Copied $concept/seed$SEED to $OPT_CKPT_DIR"
done

echo ""
echo "=== Mask sparsity summary ==="
for concept in weekdays months countries gender cities; do
  meta="$OPT_CKPT_DIR/$concept/seed${SEED}/meta.json"
  if [ -f "$meta" ]; then
    echo "$concept: $(python -c "import json; d=json.load(open('$meta')); print(f'sparsity={d[\"mask_sparsity\"]:.4f}, active={d[\"active_dims\"]}, best_epoch={d[\"best_epoch\"]}, val_loss={d[\"best_val_loss\"]:.4f}')")"
  fi
done

echo ""
echo "============================================="
echo "=== Phase 3: Anisotropic Attack (N=10) ==="
echo "============================================="
python concept_leakage/attack/run_anisotropic_attack.py \
  --n_releases 10 \
  --ckpt_dir "$OPT_CKPT_DIR" \
  --results_dir "$OPT_RESULTS_DIR" \
  --fp_subdir "anisotropic_opt" \
  --group_size_min 30

echo ""
echo "============================================="
echo "=== Phase 4: STS12 Utility Evaluation ==="
echo "============================================="
python concept_leakage/evaluation/sts12_anisotropic_eval.py \
  --ckpt_dir "$OPT_CKPT_DIR" \
  --results_dir "$OPT_RESULTS_DIR"

echo ""
echo "============================================="
echo "=== All optimized experiment phases complete ==="
echo "============================================="

echo ""
echo "=== Final sigma profile analysis ==="
python -c "
import numpy as np
concepts = ['weekdays', 'months', 'countries', 'gender', 'cities']
ckpt = '$OPT_CKPT_DIR'
for c in concepts:
    s = np.load(f'{ckpt}/{c}/seed42/sigma.npy')
    m = np.load(f'{ckpt}/{c}/seed42/mask.npy')
    print(f'{c}: sigma min={s.min():.4f} max={s.max():.4f} std={s.std():.4f}, mask_active={(m>0.5).sum()}/{len(m)}')

print('\nCross-concept sigma cosine similarity:')
sigmas = {c: np.load(f'{ckpt}/{c}/seed42/sigma.npy') for c in concepts}
for i, c1 in enumerate(concepts):
    for j, c2 in enumerate(concepts):
        if j <= i: continue
        s1 = sigmas[c1] / np.linalg.norm(sigmas[c1])
        s2 = sigmas[c2] / np.linalg.norm(sigmas[c2])
        print(f'  {c1} vs {c2}: {np.dot(s1,s2):.6f}')
"
