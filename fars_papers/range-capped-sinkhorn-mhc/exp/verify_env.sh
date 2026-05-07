#!/bin/bash
# Verify the RRCS experiment environment: CUDA, mHC import, and 10-iter training sanity check.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/range-capped-sinkhorn-mhc/exp/.venv/bin/activate

echo "=== Step 1: CUDA check ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('bf16:', torch.cuda.is_bf16_supported())"

echo "=== Step 2: mHC import check ==="
python -c "import hyper_connections; print('mHC import OK')"

echo "=== Step 3: 10-iter training sanity check ==="
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/range-capped-sinkhorn-mhc/exp/mhc_repo/examples/nanogpt
python train.py config/train_fineweb10B_mhc_48l.py max_iters=10 wandb_log=False eval_interval=5

echo "=== ALL CHECKS PASSED ==="
