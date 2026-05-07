#!/usr/bin/env bash
set -e

pip install -r requirements.txt

python train.py --case US --data data_sources.csv --episodes 5000 --seed 42 --outdir results --modeldir models
python train.py --case UK --data data_sources.csv --episodes 5000 --seed 42 --outdir results --modeldir models

python sensitivity.py --case US --data data_sources.csv --model models/dqn_US.pt --out results/sensitivity_US.csv
python sensitivity.py --case UK --data data_sources.csv --model models/dqn_UK.pt --out results/sensitivity_UK.csv

python robustness.py --case US --data data_sources.csv --model models/dqn_US.pt --out results/robustness_US.csv
python robustness.py --case UK --data data_sources.csv --model models/dqn_UK.pt --out results/robustness_UK.csv

python llm_eval.py --ref datasets/llm_eval_reference.csv --pred datasets/llm_eval_outputs.csv --out results/llm_eval_scores.csv
