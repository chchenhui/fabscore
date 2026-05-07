#!/bin/bash
# Load environment variables for the LogRules poisoning experiment.
# Sources the project .env and exports model/API configuration.
# Usage: source scripts/setup_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

if [ -f "$EXP_ROOT/.env" ]; then
    set -a
    source "$EXP_ROOT/.env"
    set +a
fi

export OPENAI_API_KEY="${LEMMA_MAAS_API_KEY}"
export OPENAI_BASE_URL="http://${LEMMA_MAAS_BASE_URL}/v1"

export MODEL_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

export INDUCTION_MODEL="gpt-4o-mini"
export PRIMARY_DEDUCTION_MODEL="Qwen/Qwen2.5-7B-Instruct"
export SECONDARY_DEDUCTION_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
