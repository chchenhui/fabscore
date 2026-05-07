#!/bin/bash
# Run ContextFocus layer selection on 200 held-out NQ-SWAP examples.

set -e

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

source .venv/bin/activate
set -a; source .env; set +a

python eacp/steering/select_layer.py "$@"
