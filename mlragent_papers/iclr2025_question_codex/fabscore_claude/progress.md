# Session Log

## Session 1
- **Purpose**: extraction
- **Paper inspected**: results/paper.md — "Adaptive Monte Carlo Sampling for Real-Time Uncertainty Estimation in Large Language Models"
- **Files created/updated**:
  - `fabscore_claude/fs_extracted.json` (created)
  - `fabscore_claude/progress.md` (created)
- **Summary**: Extracted 4 table entries from Table 1 (baseline vs head_only on SST-2: accuracy and F1), 2 figures (loss_curve.png, metrics.png), and 1 results-section claim (≈23% accuracy loss for head_only strategy). Table 1 values restated in Section 5 body text were skipped as duplicates.
- **Next session**: Run analysis/scoring (fs_analysis.json) against ground truth or reference answers.

## Session 2 (2026-04-24)
- **Purpose**: analysis (static analysis of all 7 claims)
- **Files inspected**:
  - `results/paper.md` — full paper
  - `codex/experiment.py` — training script (distilbert-base-uncased, SST-2 subset, baseline vs head_only)
  - `codex/plot_results.py` — figure generation script
  - `codex/results/baseline/results.json` — stored run results for baseline
  - `codex/results/head_only/results.json` — stored run results for head_only
  - `results/results.csv` and `codex/results/results.csv` — summary CSV tables
  - `results/loss_curve.png`, `results/metrics.png` — generated figures
- **Files created/updated**:
  - `fabscore_claude/fs_analysis.json` (created)
  - `fabscore_claude/progress.md` (updated)
- **Summary**: All 7 claims are `static_verifiable`.
  - Claims 1–4 (Table 1 accuracy/F1 for baseline and head_only): exact values found in codex/results/{baseline,head_only}/results.json and confirmed in results/results.csv.
  - Claims 5–6 (figures loss_curve.png and metrics.png): figure PNGs exist; underlying loss/metric data found in results.json files; codex/plot_results.py shows exactly how figures are generated from those files.
  - Claim 7 (≈23% accuracy drop): arithmetic confirms 0.74−0.51=0.23; head_only freeze logic confirmed in experiment.py.
- **Next session**: No further action needed; all claims resolved statically.
