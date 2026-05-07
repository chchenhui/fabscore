#!/bin/bash
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

echo "=== Sanity check: 5 items on MMStar (thinking mode) ==="
python mi_decoding/scripts/run_vanilla_baseline.py \
    --benchmark mmstar \
    --max_items 5 \
    --max_new_tokens 1000 \
    --output_dir mi_decoding/outputs/sanity_check_v2/mmstar

echo ""
echo "=== Sanity check: 5 items on HallusionBench (thinking mode) ==="
python mi_decoding/scripts/run_vanilla_baseline.py \
    --benchmark hallusionbench \
    --max_items 5 \
    --max_new_tokens 1000 \
    --output_dir mi_decoding/outputs/sanity_check_v2/hallusionbench

echo ""
echo "=== Done ==="
