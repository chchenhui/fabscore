#!/bin/bash
# Evaluate v3 checkpoints: pass DATASET as arg, evaluates all 3 conditions x 3 seeds
# Usage: bash run_eval_v3.sh <biwi|vocaset>
set -e
DATASET=${1:?Usage: run_eval_v3.sh <biwi|vocaset>}
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.
DATA_DIR="scaffoldswap/data/${DATASET}/processed"
for COND in A B C; do
    for SEED in 42 123 456; do
        CKPT="scaffoldswap/outputs/${DATASET}/cond${COND}_v3/seed${SEED}/best_model.pt"
        OUT="scaffoldswap/outputs/${DATASET}/cond${COND}_v3/seed${SEED}/eval_result.json"
        echo "Evaluating ${DATASET} Cond${COND} seed${SEED}..."
        python3 -m scaffoldswap.evaluate --checkpoint "${CKPT}" --data_dir "${DATA_DIR}" --output "${OUT}"
    done
done
echo "All evaluations complete for ${DATASET}"
