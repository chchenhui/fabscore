#!/bin/bash
# Run ContextFocus steered inference on ConFiQA-MC subset.

set -e

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

source .venv/bin/activate
set -a; source .env; set +a

python eacp/steering/steered_inference.py "$@"
