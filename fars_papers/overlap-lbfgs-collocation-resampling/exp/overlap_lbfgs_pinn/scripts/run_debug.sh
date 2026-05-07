#!/bin/bash
set -e

PROJECT_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -m overlap_lbfgs_pinn.scripts.run_adam_resampling_ice_shelf \
    --budget 500 \
    --seeds 0
