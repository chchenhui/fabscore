#!/bin/bash
# Evaluate all 3 HuBERT-continuous seeds on BIWI
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.

DATA_DIR="scaffoldswap/data/biwi/processed"
RESULTS_DIR="results/biwi"
mkdir -p "${RESULTS_DIR}"

for SEED in 42 123 456; do
    CKPT="scaffoldswap/outputs/biwi/hubert_continuous/seed${SEED}/best_model.pt"
    OUT="${RESULTS_DIR}/hubert_continuous_seed${SEED}.json"
    echo "Evaluating seed ${SEED}..."
    python3 -m scaffoldswap.evaluate --checkpoint "${CKPT}" --data_dir "${DATA_DIR}" --output "${OUT}"
done
echo "All evaluations complete."
