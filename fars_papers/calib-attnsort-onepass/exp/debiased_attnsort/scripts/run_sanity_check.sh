#!/bin/bash
# Sanity check: run debiased_k1 with 10 examples to verify code correctness.
set -euo pipefail

PROJ_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/calib-attnsort-onepass/exp"

source "${PROJ_ROOT}/.venv/bin/activate"

set -a
source "${PROJ_ROOT}/.env"
set +a

export WANDB_MODE=offline

cd "${PROJ_ROOT}"

python debiased_attnsort/src/eval_pipeline.py --mode debiased_k1 --seed 42 --sanity_check
