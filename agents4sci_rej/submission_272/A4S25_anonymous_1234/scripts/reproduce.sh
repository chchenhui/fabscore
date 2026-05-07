#!/usr/bin/env bash
set -euo pipefail
python code/run_experiments.py --random-seed 123 --results-dir results/
echo "Done. See results/metrics.json and paper/figures/."
