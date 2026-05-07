#!/bin/bash
# Train k-means codebook on HuBERT features from VOCASET training audio
set -e

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate

export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export PYTHONPATH="${PYTHONPATH}:."
export NUMBA_CACHE_DIR=/tmp/numba_cache

python3 scaffoldswap/frontends/train_kmeans.py \
    --data_path scaffoldswap/data/vocaset/processed/train.pt \
    --hubert_model facebook/hubert-base-ls960 \
    --cache_dir pretrained_models \
    --n_clusters 200 \
    --output_path pretrained_models/hubert_kmeans_vocaset_K200.pkl \
    --device cuda

echo "K-means training complete!"
