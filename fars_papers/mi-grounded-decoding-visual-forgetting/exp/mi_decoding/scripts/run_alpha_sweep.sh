#!/bin/bash
# Alpha sweep: test different alpha thresholds with lambda=0.005 on first 300 MMStar items.
# GPU 0: alpha=0.1 (very conservative)
# GPU 1: alpha=0.2 (conservative)
# GPU 2: alpha=0.5 (aggressive)
# GPU 3: alpha=0.8 (very aggressive)
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

export WANDB_MODE=offline
MAX_ITEMS=300

echo "=== Alpha Sweep: 4 configs x 300 MMStar items ==="

CUDA_VISIBLE_DEVICES=0 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.005 --alpha 0.1 --max_weight 5.0 \
    --output_dir mi_decoding/outputs/sweep_alpha0.1/mmstar &

CUDA_VISIBLE_DEVICES=1 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.005 --alpha 0.2 --max_weight 5.0 \
    --output_dir mi_decoding/outputs/sweep_alpha0.2/mmstar &

CUDA_VISIBLE_DEVICES=2 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.005 --alpha 0.5 --max_weight 5.0 \
    --output_dir mi_decoding/outputs/sweep_alpha0.5/mmstar &

CUDA_VISIBLE_DEVICES=3 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.005 --alpha 0.8 --max_weight 5.0 \
    --output_dir mi_decoding/outputs/sweep_alpha0.8/mmstar &

echo "Waiting..."
wait
echo "Done!"

echo ""
echo "=== Results ==="
for dir in sweep_alpha0.1 sweep_alpha0.2 sweep_alpha0.5 sweep_alpha0.8; do
    echo ""
    echo "--- ${dir} ---"
    python mi_decoding/scripts/merge_and_evaluate.py \
        --benchmark mmstar \
        --output_dir "mi_decoding/outputs/${dir}/mmstar" \
        --results_file "mi_decoding/results/${dir}_mmstar.json"
done
echo "=== Sweep Complete ==="
