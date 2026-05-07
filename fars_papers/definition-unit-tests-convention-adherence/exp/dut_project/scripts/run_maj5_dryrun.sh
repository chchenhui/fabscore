#!/bin/bash
# Dry-run: 5 items, 2 samples, Qwen condition A. Validates --top-p and multi-sample output.
set -euo pipefail

PROJ=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/definition-unit-tests-convention-adherence/exp
source "$PROJ/.venv/bin/activate"
cd "$PROJ"

set -a; source "$PROJ/.env"; set +a
export WANDB_MODE=offline

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
INPUT="dut_project/data/erdos_conventions_bench_5items.jsonl"
OUTPUT="dut_project/outputs/dryrun_maj5.jsonl"

head -5 dut_project/data/erdos_conventions_bench.jsonl > "$INPUT"

python -m dut_project.inference.run_inference \
    --model "$MODEL" \
    --condition A \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --temperature 0.7 \
    --top-p 0.95 \
    --max-tokens 512 \
    --tensor-parallel-size 1 \
    --k 3 \
    --num-samples 2

echo "=== Dry-run output ==="
wc -l "$OUTPUT"
python -c "
import json
items = [json.loads(l) for l in open('$OUTPUT')]
print(f'Total rows: {len(items)}')
ids = set(i['item_id'] for i in items)
print(f'Unique item_ids: {len(ids)}')
for i in items:
    print(f\"  item_id={i['item_id']}, sample_idx={i['sample_idx']}, ans={i['parsed'].get('final_answer')}\")
"

echo "=== Dry-run complete ==="
