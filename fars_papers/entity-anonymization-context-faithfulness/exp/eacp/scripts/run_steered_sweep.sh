#!/bin/bash
# Run multiple steered inference configurations on ConFiQA-MC 1500 subset.
# Usage: bash run_steered_sweep.sh <mode>
# mode: "prefill_m2" | "gen_m1" | "gen_m05"

set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

MODE=${1:-"gen_m1"}
LAYER=13

case "$MODE" in
    prefill_m2)
        echo "=== Running: prefill_only, m=2.0, layer=$LAYER ==="
        python eacp/steering/steered_inference.py --layer $LAYER --multiplier 2.0 --prefill_only
        ;;
    gen_m1)
        echo "=== Running: all_tokens, m=1.0, layer=$LAYER ==="
        python eacp/steering/steered_inference.py --layer $LAYER --multiplier 1.0
        ;;
    gen_m05)
        echo "=== Running: all_tokens, m=0.5, layer=$LAYER ==="
        python eacp/steering/steered_inference.py --layer $LAYER --multiplier 0.5
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

echo "Done: $MODE"
