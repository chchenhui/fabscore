#!/bin/bash
# Exports API keys from .env for HST experiments.
# Source this script: source hst/scripts/setup_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
else
    echo "WARNING: .env file not found at $PROJECT_ROOT/.env"
fi

export DEEPSEEK_API_KEY="${LEMMA_MAAS_API_KEY}"
export DEEPSEEK_BASE_URL="http://${LEMMA_MAAS_BASE_URL}/v1"

if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "WARNING: DEEPSEEK_API_KEY (LEMMA_MAAS_API_KEY) is not set"
fi

if [ -z "$LEMMA_MAAS_BASE_URL" ]; then
    echo "WARNING: DEEPSEEK_BASE_URL (LEMMA_MAAS_BASE_URL) is not set"
fi

echo "Environment configured. DEEPSEEK_BASE_URL=$DEEPSEEK_BASE_URL"
