#!/bin/bash
set -e
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

echo "=== Visual Replay Sanity Check (10 items) ==="
python mi_decoding/scripts/run_visual_replay_baseline.py \
    --benchmark mmstar \
    --max_items 10 \
    --max_new_tokens 512 \
    --output_dir mi_decoding/outputs/visual_replay_sanity_check/mmstar

echo ""
echo "=== Done ==="
