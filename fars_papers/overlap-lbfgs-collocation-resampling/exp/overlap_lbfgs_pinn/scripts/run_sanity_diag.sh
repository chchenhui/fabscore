#!/bin/bash
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp
source .venv/bin/activate

echo "=== Sanity check: naive resampled L-BFGS ==="
python -m overlap_lbfgs_pinn.scripts.run_naive_resampled_lbfgs_ice_shelf \
    --budget 500 --adam_budget 200 --adam_fixed_budget 100 \
    --gamma 0.5 --seeds 0 \
    --output_dir overlap_lbfgs_pinn/outputs/sanity_naive

echo ""
echo "=== Sanity check: fixed-collocation L-BFGS ==="
python -m overlap_lbfgs_pinn.scripts.run_adam_lbfgs_ice_shelf \
    --budget 500 --adam_budget 200 \
    --gamma 0.5 --seeds 0 \
    --output_dir overlap_lbfgs_pinn/outputs/sanity_fixed

echo ""
echo "=== Sanity check: overlap L-BFGS o=0.5 ==="
python -m overlap_lbfgs_pinn.scripts.run_overlap_lbfgs_ice_shelf \
    --budget 500 --adam_budget 200 --adam_fixed_budget 100 \
    --overlap_frac 0.5 --gamma 0.5 --seeds 0 \
    --output_dir overlap_lbfgs_pinn/outputs/sanity_overlap_o05 \
    --run_prefix sanity_overlap_o05

echo ""
echo "=== Checking output files ==="
for dir in sanity_naive sanity_fixed sanity_overlap_o05; do
    echo "--- $dir ---"
    ls -la overlap_lbfgs_pinn/outputs/$dir/
    echo ""
done

echo "=== Checking lbfgs_step_history content ==="
python -c "
import json, os
for d in ['sanity_naive', 'sanity_fixed', 'sanity_overlap_o05']:
    path = f'overlap_lbfgs_pinn/outputs/{d}/seed_0_lbfgs_step_history.json'
    if os.path.exists(path):
        data = json.load(open(path))
        print(f'{d}: {len(data)} steps, keys={list(data[0].keys()) if data else \"empty\"}')
        if data:
            print(f'  first step: {data[0]}')
    else:
        print(f'{d}: NO step_history file')
"
echo "=== SANITY CHECK COMPLETE ==="
