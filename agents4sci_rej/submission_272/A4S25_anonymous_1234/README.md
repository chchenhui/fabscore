# Agents4Science 2025 Anonymous Artifact

> **Anonymous ID:** 1234 (placeholder)  
> **Conference:** Agents4Science 2025  
> **How to reproduce:** One command, CPU-only.

## Quickstart (pip)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python code/run_experiments.py --random-seed 42
pytest -q
```

## Outputs
- Metrics: `results/metrics.json`
- Tables: `results/tables/*.csv`
- Predictions: `results/predictions.csv`

## Determinism
- We set explicit seeds for `random` and `numpy`.
- No GPU used. All data are synthetic and generated deterministically from the seed.

## Data Card
See `data/README.md` for provenance, generation procedure, licenses, and splits.

## Notes
- PDF metadata should be scrubbed before submission:
```
exiftool -overwrite_original -Title="Anonymous Submission" -Author="" paper.pdf
```


## Reproducibility & Build

### Determinism
Results are now deterministic across runs with the same seed. We replaced Python's per-process `hash()` with a stable md5-based helper.
To be extra safe, we also set `PYTHONHASHSEED=0` in the provided scripts.

### One-command reproduce
```bash
make reproduce
```
This will:
1. Run `code/run_experiments.py` with `--random-seed 42` and write outputs under `results/`.
2. Compile the paper to `paper/main.pdf`.

### Manual commands
```bash
export PYTHONHASHSEED=0
python code/run_experiments.py --random-seed 42 --results-dir results
bash scripts/build.sh
```

### Page limit check
The conference requires ≤ 8 pages for the main paper (excluding references and required statements). After building,
use `pdfinfo paper/main.pdf` to inspect total pages. Only the main content counts toward the limit.

### Notes
- Figures in `main.tex` were aligned to existing files under `paper/figures/`.
- Any references to a missing appendix were replaced with “Supplementary Material”.
- Acquisition function text clarifies that we scale by standard deviation (UCB-style) and sweep λ for calibration.


## Evidence Map (Claim → Artifact)
| Claim (Paper)                               | Evidence (Artifact)                                 |
|---------------------------------------------|-----------------------------------------------------|
| +0.10 AUPRC over heuristic baseline         | results/tables/comparison.csv; paper/figures/pr_curve_test.pdf |
| +0.9 Hit@10 under 50-budget                 | results/tables/ablations.csv; paper/figures/valap_main.pdf     |
| Calibration maintained at high thresholds   | results/tables/per_relation.csv; paper/figures/loss_main.pdf   |
| Uncertainty beats static/heuristic early    | results/tables/robustness_quick.csv; paper/figures/valap_static.pdf |
