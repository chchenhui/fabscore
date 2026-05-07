# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** paper.pdf (compute-matched-diffusion-planning-audit)
**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries

**Summary:**
- Extracted 18 table entries from Table 1 (main results: Qwen2.5-7B Greedy, Best-of-k, Dream-7B Diffusion on Countdown and Mini Sudoku) and Table 2 (robustness analysis with Median vs P75 timing estimators).
- Extracted 2 figures: Figure 1 (evaluation protocol diagram) and Figure 2 (scaling curves).
- Extracted 5 results_section claims not already captured in tables: 83% gap closure claim, bootstrap CI [+6.1pp, +14.6pp], k=4 crossover with 11% budget usage, 7.3% accuracy at k=1, and k≈87 extrapolation at 2.2× compute budget.

**Next session should:**
- Run analysis/scoring against reference results if available.
- Verify extracted numerical values against any updated version of the paper.

## Session 2 — 2026-04-24
**Purpose:** analysis (static audit)
**Files inspected:**
- `paper.pdf` (all 7 pages)
- `exp/EXPERIMENT_RESULTS/qwen_greedy_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/dream_diffusion_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/qwen_best_of_k/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/p75_sensitivity/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/scaling_analysis/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json`
- `exp/audit/results/tables/bootstrap_ci.csv`
- `exp/audit/results/tables/main_results.csv`
- `exp/audit/results/tables/scaling_summary.json`

**Files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with classification of all 25 claims

**Summary of classifications:**
All 25 claims classified as `static_verifiable`. The repository contains detailed JSON result artifacts that directly match every numerical claim in the paper:
- Table 1 values (claims 1–10): confirmed via qwen_greedy_baseline/RESULTS.json, dream_diffusion_baseline/RESULTS.json, and qwen_best_of_k/RESULTS.json
- Table 2 values (claims 11–18): confirmed via p75_sensitivity/RESULTS.json and dream_diffusion_baseline/RESULTS.json
- Figure 1 (claim 19): three-condition protocol directly implemented in inference scripts and calibrate.py
- Figure 2 (claim 20): underlying numerical data confirmed in scaling_analysis/RESULTS.json and scaling_summary.json (k=4 crossover on Countdown, k≈87 crossover on Sudoku)
- Results section (claims 21–25): 83% gap closure, bootstrap CI [+6.1, +14.6], k=4/11% budget, 7.3%→67.2% scaling, k≈87/2.2× all confirmed from the same JSON artifacts

**Notable observation:** `exp/audit/results/tables/main_results.csv` shows pre-optimization BoK values (39.33%, 66.13%) that differ from the paper's reported optimized values (39.1%, 67.2%). The canonical post-optimization values are in `qwen_best_of_k/RESULTS.json`, which the paper correctly cites. The CSV is an intermediate artifact that was not updated after the optimization run.

**Next session should:**
- No further analysis required; all claims statically verified.
- Optionally run execution to confirm that the scripts reproduce the exact values in the JSON artifacts.
