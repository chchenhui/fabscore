#!/bin/bash
# Run differential SCT extraction for a single taboo model.
# Usage: bash run_differential_sct.sh <secret_word> [max_examples]
# Example: bash run_differential_sct.sh gold
#          bash run_differential_sct.sh gold 10  (sanity check)

set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp/.venv/bin/activate

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp
if [ -f "$PROJ_DIR/.env" ]; then
    set -a
    source "$PROJ_DIR/.env"
    set +a
fi

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp

SECRET="${1:-gold}"
MAX_EXAMPLES="${2:-}"

FT_MODEL="bcywinski/gemma-2-9b-it-taboo-${SECRET}"
BASE_MODEL="google/gemma-2-9b-it"
SOURCE_FILE="sct/outputs/taboo_${SECRET}_activations.json"
OUTPUT_FILE="sct/outputs/taboo_${SECRET}_diff_sct_scored.json"

EXTRA_ARGS=""
if [ -n "$MAX_EXAMPLES" ]; then
    EXTRA_ARGS="--max_examples $MAX_EXAMPLES"
    OUTPUT_FILE="sct/outputs/taboo_${SECRET}_diff_sct_sanity.json"
fi

echo "=== Differential SCT Extraction ==="
echo "Secret: $SECRET"
echo "FT model: $FT_MODEL"
echo "Base model: $BASE_MODEL"
echo "Source: $SOURCE_FILE"
echo "Output: $OUTPUT_FILE"
echo "Extra args: $EXTRA_ARGS"
echo ""

python sct/extraction/extract_differential_sct.py \
    --ft_model_name "$FT_MODEL" \
    --base_model_name "$BASE_MODEL" \
    --source_activations "$SOURCE_FILE" \
    --output_path "$OUTPUT_FILE" \
    --mid_layer 32 \
    --final_layer 41 \
    --top_k_out 500 \
    $EXTRA_ARGS

echo "Done!"
