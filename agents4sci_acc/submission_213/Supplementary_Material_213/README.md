
# Reproducibility Pack — Sustainable Investment RL + LLM

This folder contains data, code, and example artifacts to reproduce the core experiments in the paper.

## 1) Environment

- Python 3.10+
- `pip install -r requirements.txt`

## 2) Data

- **data_sources.csv** — parameter table (US/UK) used by the environment. You can edit values or add new cases.
- Units are encoded per row; the environment converts as needed.

## 3) Train DQN and export trajectories

```bash
# US
python train.py --case US --data data_sources.csv --episodes 5000 --seed 42 --outdir results --modeldir models

# UK
python train.py --case UK --data data_sources.csv --episodes 5000 --seed 42 --outdir results --modeldir models
```

Outputs:
- `models/dqn_*.pt` — trained weights
- `results/*_trajectory.csv` — greedy rollout with info metrics (used for explanations and evaluation)
- `results/*_summary.json` — run metadata

## 4) Sensitivity analysis (SCC & Productivity)

```bash
python sensitivity.py --case US --data data_sources.csv --model models/dqn_US.pt --out results/sensitivity_US.csv
python sensitivity.py --case UK --data data_sources.csv --model models/dqn_UK.pt --out results/sensitivity_UK.csv
```

## 5) Robustness tests (noise & climate change)

```bash
python robustness.py --case US --data data_sources.csv --model models/dqn_US.pt --out results/robustness_US.csv
python robustness.py --case UK --data data_sources.csv --model models/dqn_UK.pt --out results/robustness_UK.csv
```

## 6) LLM prompts, responses, and automatic evaluation

- `llm_prompts_responses.json` includes **exact prompts and sample outputs** used in the paper.
- Automatic scoring:
```bash
python llm_eval.py --ref datasets/llm_eval_reference.csv --pred datasets/llm_eval_outputs.csv --out results/llm_eval_scores.csv
```

Metrics:
- exact_match, token_f1, numeric_consistency, coverage

## 7) Re-running with your own LLM outputs

1. Export your model's answers to `datasets/llm_eval_outputs.csv` with columns `id,answer` matching the reference IDs.
2. Re-run the evaluation command above.

## 8) Reproducibility settings

- Seeds: 42 by default for agent and environment RNG.
- Hyperparameters are in `dqn_agent.py` (DQNConfig) and `rl_env.py` (EnvConfig).

## 9) Notes

- The provided environment is self-contained and does **not** depend on `gym`.
- Monetary NPVs are aggregated at operation stage for clarity; this matches the paper's description.
- For GPU acceleration, install a CUDA build of PyTorch (optional).
