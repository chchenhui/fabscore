#!/bin/bash
# Optimized public-anchor drift adapter training and evaluation.
# Usage:
#   bash pada/scripts/run_public_anchor_optimized.sh              # full run
#   bash pada/scripts/run_public_anchor_optimized.sh --dry-run    # quick check

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

set -a
source .env
set +a

export WANDB_MODE=offline

python -m pada.scripts.run_public_anchor_optimized "$@"
