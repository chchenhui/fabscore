#!/bin/bash
# Quick sanity check for short-budget runs. 1 GPU, 5 items each.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

echo "=== Sanity: Vanilla short (mmstar, 5 items) ==="
CUDA_VISIBLE_DEVICES=0 python mi_decoding/scripts/run_vanilla_baseline.py \
    --benchmark mmstar \
    --num_shards 1 \
    --shard_id 0 \
    --output_dir mi_decoding/outputs/sanity_vanilla_short/mmstar \
    --max_new_tokens 128 \
    --max_items 5

echo ""
echo "=== Sanity: Visual replay short (mmstar, 5 items) ==="
CUDA_VISIBLE_DEVICES=0 python mi_decoding/scripts/run_visual_replay_baseline.py \
    --benchmark mmstar \
    --num_shards 1 \
    --shard_id 0 \
    --output_dir mi_decoding/outputs/sanity_vr_short/mmstar \
    --max_new_tokens 128 \
    --max_items 5

echo ""
echo "=== Checking outputs ==="
echo "Vanilla short:"
wc -l mi_decoding/outputs/sanity_vanilla_short/mmstar/shard_0.jsonl
head -1 mi_decoding/outputs/sanity_vanilla_short/mmstar/shard_0.jsonl | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('Keys:', list(d.keys())); print('extracted_answer:', repr(d.get('extracted_answer')))"

echo ""
echo "Visual replay short:"
wc -l mi_decoding/outputs/sanity_vr_short/mmstar/shard_0.jsonl
head -1 mi_decoding/outputs/sanity_vr_short/mmstar/shard_0.jsonl | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('Keys:', list(d.keys())); print('extracted_answer:', repr(d.get('extracted_answer')))"

echo "=== Sanity check complete ==="
