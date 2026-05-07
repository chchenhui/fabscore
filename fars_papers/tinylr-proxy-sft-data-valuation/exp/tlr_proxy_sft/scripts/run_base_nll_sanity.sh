#!/bin/bash
set -euo pipefail

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp/.venv/bin/activate

BASE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp

python ${BASE}/tlr_proxy_sft/analysis/compute_base_nll.py \
    --model Qwen/Qwen2.5-1.5B \
    --data-dir ${BASE}/tlr_proxy_sft/data/processed \
    --output-csv ${BASE}/tlr_proxy_sft/results/base_nll_scores_sanity.csv \
    --cutoff-len 4096 \
    --batch-size 8 \
    --max-samples 100 \
    --datasets hkust-nlp__dart-math-hard mathplus
