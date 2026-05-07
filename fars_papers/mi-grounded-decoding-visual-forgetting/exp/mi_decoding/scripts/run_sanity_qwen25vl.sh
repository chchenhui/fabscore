#!/bin/bash
# Sanity check: vanilla + MI on 5 items, Qwen2.5-VL-7B-Instruct, 1 GPU
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/mi-grounded-decoding-visual-forgetting/exp

export WANDB_MODE=offline

echo "=== Sanity: Vanilla on Qwen2.5-VL-7B-Instruct (5 items) ==="
python mi_decoding/scripts/run_vanilla_baseline.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --benchmark mmstar \
    --max_new_tokens 512 \
    --max_items 5 \
    --subset_file mi_decoding/configs/mmstar_subset_300.json \
    --output_dir mi_decoding/outputs/sanity_vanilla_qwen25vl

echo ""
echo "=== Sanity: MI decoding on Qwen2.5-VL-7B-Instruct (5 items) ==="
python mi_decoding/scripts/run_mi_baseline.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --benchmark mmstar \
    --max_new_tokens 512 \
    --lam 0.02 \
    --alpha 0.3 \
    --t0_prompt_length \
    --max_weight 5.0 \
    --max_items 5 \
    --subset_file mi_decoding/configs/mmstar_subset_300.json \
    --output_dir mi_decoding/outputs/sanity_mi_qwen25vl

echo ""
echo "=== Sanity check complete ==="
