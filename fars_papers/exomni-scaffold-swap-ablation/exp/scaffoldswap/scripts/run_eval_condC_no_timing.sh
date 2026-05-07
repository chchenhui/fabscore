#!/bin/bash
# Evaluate C w/o timing checkpoints on BIWI (3 seeds)
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.
DATA_DIR="scaffoldswap/data/biwi/processed"
for SEED in 42 123 456; do
    CKPT="scaffoldswap/outputs/biwi/condC_no_timing/seed${SEED}/best_model.pt"
    OUT="scaffoldswap/outputs/biwi/condC_no_timing/seed${SEED}/eval_result.json"
    echo "Evaluating C w/o timing seed${SEED}..."
    python3 -m scaffoldswap.evaluate --checkpoint "${CKPT}" --data_dir "${DATA_DIR}" --output "${OUT}"
done
echo "All evaluations complete"
