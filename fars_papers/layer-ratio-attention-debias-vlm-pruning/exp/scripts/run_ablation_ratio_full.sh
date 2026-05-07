#!/bin/bash
# Full-scale ablation: ratio debiasing at L12 with two K_s values.
# Run one config at a time since both need all 8 GPUs.
# Usage: bash scripts/run_ablation_ratio_full.sh <config> [num_gpus]
# Configs: ratio_ks3_km12 or ratio_ks2_km12

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

CONFIG=${1:-ratio_ks3_km12}
NUM_GPUS=${2:-8}

bash scripts/run_full_eval_optimized.sh $CONFIG $NUM_GPUS

echo ""
echo "=== Ablation run complete: $CONFIG ==="
if [ -f "results/ablation_${CONFIG}/summary.json" ]; then
    echo "Summary:"
    python3 -c "
import json
with open('results/ablation_${CONFIG}/summary.json') as f:
    data = json.load(f)
accs = []
for k, v in sorted(data.items()):
    if 'accuracy' in v:
        print(f'  {k}: {v[\"accuracy\"]:.4f}')
        accs.append(v['accuracy'])
if accs:
    print(f'  AVERAGE: {sum(accs)/len(accs):.4f}')
"
fi
