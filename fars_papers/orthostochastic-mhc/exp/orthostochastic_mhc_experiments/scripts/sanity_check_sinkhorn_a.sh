#!/bin/bash
# Sanity check: short run with seed=1, max_iters=50, eval_interval=25, 4 GPUs DDP
set -euo pipefail
exec 2>&1

echo "=== Sanity check starting ==="

EXP_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/orthostochastic-mhc/exp"
VENV="${EXP_ROOT}/.venv/bin/activate"
ENV_FILE="${EXP_ROOT}/.env"
TRAIN_DIR="${EXP_ROOT}/mHC-manifold-constrained-hyper-connections/examples/nanogpt"
CONFIG="${EXP_ROOT}/orthostochastic_mhc_experiments/configs/setting_a_sinkhorn.py"
OUT_DIR="${EXP_ROOT}/orthostochastic_mhc_experiments/logs/sanity_check_sinkhorn_a"

source "${VENV}"
set -a; source "${ENV_FILE}"; set +a

cd "${TRAIN_DIR}"

echo "=== Starting training (4 GPU DDP) ==="
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
    "${CONFIG}" \
    seed=1 \
    max_iters=50 \
    eval_interval=25 \
    out_dir="${OUT_DIR}" \
    wandb_run_name="sanity-check-sinkhorn-a"

echo "=== Training complete ==="
