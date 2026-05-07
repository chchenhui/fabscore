#!/bin/bash
# Sanity check: run fixed_gamma=0.5 on 3 MMStar items to verify correctness.
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

export WANDB_MODE=offline

echo "=== Sanity check: fixed_gamma=0.5 on 3 items ==="
python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar \
    --max_items 3 \
    --num_shards 1 --shard_id 0 \
    --output_dir mi_decoding/outputs/sanity_fixed_gamma \
    --max_new_tokens 512 \
    --alpha 0.3 --lam 0.02 --max_weight 5.0 \
    --fixed_gamma 0.5

echo ""
echo "=== Output file ==="
cat mi_decoding/outputs/sanity_fixed_gamma/shard_0.jsonl

echo ""
echo "=== Merge and evaluate ==="
python mi_decoding/scripts/merge_and_evaluate.py \
    --benchmark mmstar \
    --output_dir mi_decoding/outputs/sanity_fixed_gamma \
    --results_file mi_decoding/outputs/sanity_fixed_gamma/results.json

echo ""
echo "=== Results ==="
cat mi_decoding/outputs/sanity_fixed_gamma/results.json
echo ""
echo "=== Sanity check complete ==="
