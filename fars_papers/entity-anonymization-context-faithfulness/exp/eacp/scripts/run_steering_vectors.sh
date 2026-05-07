#!/bin/bash
# Compute ContextFocus steering vectors from NQ-SWAP prompt pairs.
# Usage: bash run_steering_vectors.sh [--limit N]

set -e

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

source .venv/bin/activate
set -a; source .env; set +a

python eacp/steering/contextfocus.py "$@"
