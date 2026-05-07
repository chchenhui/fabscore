#!/bin/bash
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/overlap-lbfgs-collocation-resampling/exp
source .venv/bin/activate

echo "=== Sanity check: naive resampled ==="
python -m overlap_lbfgs_pinn.scripts.run_naive_resampled_lbfgs_ice_shelf \
    --budget 500 --adam_budget 200 --adam_fixed_budget 100 \
    --gamma 0.5 --seeds 0 \
    --output_dir overlap_lbfgs_pinn/outputs/sanity_naive

echo ""
echo "=== Sanity check: fixed LBFGS ==="
python -m overlap_lbfgs_pinn.scripts.run_adam_lbfgs_ice_shelf \
    --budget 500 --adam_budget 200 \
    --gamma 0.5 --seeds 0 \
    --output_dir overlap_lbfgs_pinn/outputs/sanity_fixed

echo ""
echo "=== Checking output files ==="
echo "Naive outputs:"
ls -la overlap_lbfgs_pinn/outputs/sanity_naive/
echo ""
echo "Naive step history sample:"
python -c "
import json
with open('overlap_lbfgs_pinn/outputs/sanity_naive/seed_0_lbfgs_step_history.json') as f:
    d = json.load(f)
print(f'Total steps: {len(d)}')
if d:
    print(f'First entry keys: {list(d[0].keys())}')
    print(f'First entry: {d[0]}')
    print(f'Last entry: {d[-1]}')
"
echo ""
echo "Fixed outputs:"
ls -la overlap_lbfgs_pinn/outputs/sanity_fixed/
echo ""
echo "Fixed step history sample:"
python -c "
import json
with open('overlap_lbfgs_pinn/outputs/sanity_fixed/seed_0_lbfgs_step_history.json') as f:
    d = json.load(f)
print(f'Total steps: {len(d)}')
if d:
    print(f'First entry keys: {list(d[0].keys())}')
    print(f'First entry: {d[0]}')
    print(f'Last entry: {d[-1]}')
"
echo ""
echo "=== Sanity check complete ==="
