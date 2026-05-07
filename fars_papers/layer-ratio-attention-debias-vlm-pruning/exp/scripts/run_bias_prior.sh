#!/bin/bash
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

rm -f data/bias_prior/layer_1.pt

python scripts/compute_bias_prior.py \
    --model-path models/InternVL2_5-8B \
    --target-layer 1 \
    --num-images 1000 \
    --coco-dir data/coco/train2014 \
    --output-dir data/bias_prior \
    --seed 42

echo "=== Bias prior computation done ==="
ls -la data/bias_prior/
