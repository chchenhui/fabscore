#!/bin/bash
# Size ablation: vary N_p in {500, 1000, 2000, 5000, 10000} and evaluate.
# Usage:
#   bash pada/scripts/run_size_ablation.sh              # full run
#   bash pada/scripts/run_size_ablation.sh --dry-run    # Np=500, 5 epochs, 1 dataset

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

set -a
source .env
set +a

python -m pada.scripts.run_size_ablation "$@"
