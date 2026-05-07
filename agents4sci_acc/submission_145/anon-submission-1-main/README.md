# Algorithmic Greenhouse

This repository accompanies the paper **“The Algorithmic Greenhouse: An AI Agent for Autonomous Discovery of Symbolic Optimizers”** (Agents4Science 2025).

It contains all code, configuration, and artifacts needed to reproduce the experiments and figures.

## Structure

- `greenhouse/` — core library (DSL, benchmarks, evolutionary search, runners, plotting).
- `experiments/` — entry point scripts to reproduce results.
- `artifacts/` — JSON logs and figures produced in the main paper (created when you run the scripts).

## Quick Start

```bash
pip install -r requirements.txt
python experiments/run_evolution.py
python experiments/run_baselines.py
python experiments/run_linreg.py
python experiments/plot_all.py
