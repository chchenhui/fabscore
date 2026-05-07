#!/bin/bash
# Combined sweep: best alpha with different lambdas, 300 MMStar items.
# GPU 0: alpha=0.8, lambda=0.005, mw=5.0 (best alpha from sweep)
# GPU 1: alpha=0.8, lambda=0.01, mw=5.0
# GPU 2: alpha=1.0, lambda=0.005, mw=5.0 (always-on correction)
# GPU 3: alpha=0.8, lambda=0.005, mw=3.0
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

export WANDB_MODE=offline
MAX_ITEMS=300

echo "=== Combined Sweep: 4 configs x 300 MMStar items ==="

CUDA_VISIBLE_DEVICES=0 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.005 --alpha 0.8 --max_weight 5.0 \
    --output_dir mi_decoding/outputs/sweep_a0.8_l0.005_mw5/mmstar &

CUDA_VISIBLE_DEVICES=1 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.01 --alpha 0.8 --max_weight 5.0 \
    --output_dir mi_decoding/outputs/sweep_a0.8_l0.01_mw5/mmstar &

CUDA_VISIBLE_DEVICES=2 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.005 --alpha 1.0 --max_weight 5.0 \
    --output_dir mi_decoding/outputs/sweep_a1.0_l0.005_mw5/mmstar &

CUDA_VISIBLE_DEVICES=3 python mi_decoding/scripts/run_mi_baseline.py \
    --benchmark mmstar --max_new_tokens 512 --max_items ${MAX_ITEMS} \
    --lam 0.005 --alpha 0.8 --max_weight 3.0 \
    --output_dir mi_decoding/outputs/sweep_a0.8_l0.005_mw3/mmstar &

echo "Waiting..."
wait
echo "Done!"

echo ""
echo "=== Results ==="
for dir in sweep_a0.8_l0.005_mw5 sweep_a0.8_l0.01_mw5 sweep_a1.0_l0.005_mw5 sweep_a0.8_l0.005_mw3; do
    echo ""
    echo "--- ${dir} ---"
    python mi_decoding/scripts/merge_and_evaluate.py \
        --benchmark mmstar \
        --output_dir "mi_decoding/outputs/${dir}/mmstar" \
        --results_file "mi_decoding/results/${dir}_mmstar.json"
done
echo "=== Sweep Complete ==="
