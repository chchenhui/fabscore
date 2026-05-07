#!/bin/bash
# Quick lambda sweep: 4 configs on 200 MMStar items each, 1 GPU per config.
# Each config runs as a single shard (no data parallelism) with --max_items 200.
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

export WANDB_MODE=offline
MAX_ITEMS=200

echo "=== Lambda Sweep: 4 configs x 200 MMStar items ==="

# Config 0 (current baseline): lambda=0.005, max_weight=5.0
CUDA_VISIBLE_DEVICES=0 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.005 --max_weight 5.0 \
    --output_dir mi_decoding/outputs/sweep_lam0.005_mw5.0/mmstar &

# Config A: lambda=0.01, max_weight=5.0
CUDA_VISIBLE_DEVICES=1 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.01 --max_weight 5.0 \
    --output_dir mi_decoding/outputs/sweep_lam0.01_mw5.0/mmstar &

# Config B: lambda=0.02, max_weight=5.0
CUDA_VISIBLE_DEVICES=2 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.02 --max_weight 5.0 \
    --output_dir mi_decoding/outputs/sweep_lam0.02_mw5.0/mmstar &

# Config C: lambda=0.02, max_weight=3.0
CUDA_VISIBLE_DEVICES=3 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.02 --max_weight 3.0 \
    --output_dir mi_decoding/outputs/sweep_lam0.02_mw3.0/mmstar &

echo "Waiting for all configs to complete..."
wait
echo "All configs done!"

echo ""
echo "=== Results ==="
for dir in sweep_lam0.005_mw5.0 sweep_lam0.01_mw5.0 sweep_lam0.02_mw5.0 sweep_lam0.02_mw3.0; do
    echo ""
    echo "--- ${dir} ---"
    python mi_decoding/scripts/merge_and_evaluate.py \
        --benchmark mmstar \
        --output_dir "mi_decoding/outputs/${dir}/mmstar" \
        --results_file "mi_decoding/results/${dir}_mmstar.json"
done

echo ""
echo "=== Sweep Complete ==="
