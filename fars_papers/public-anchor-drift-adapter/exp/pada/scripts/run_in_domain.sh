#!/bin/bash
# In-domain drift adapter training and evaluation.
# Usage:
#   bash pada/scripts/run_in_domain.sh              # full run
#   bash pada/scripts/run_in_domain.sh --dry-run    # quick verification

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

set -a
source .env
set +a

python -m pada.scripts.run_in_domain "$@"
