#!/bin/bash
# Run full evaluation for all optimized configs sequentially.
# Each uses 8-GPU data parallelism.

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp

echo "=== Starting full evaluations of optimized configs ==="

for CONFIG in rawmis_km12 wc_a05_ks2_km12; do
    echo ""
    echo "============================================"
    echo "=== Config: $CONFIG ==="
    echo "============================================"
    bash scripts/run_full_eval_optimized.sh $CONFIG 8
    echo ""
done

echo "=== All evaluations complete ==="
echo ""

echo "=== Summary of all results ==="
for d in results/opt_*; do
    if [ -f "$d/summary.json" ]; then
        echo ""
        echo "--- $(basename $d) ---"
        python3 -c "
import json
with open('$d/summary.json') as f:
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
done
