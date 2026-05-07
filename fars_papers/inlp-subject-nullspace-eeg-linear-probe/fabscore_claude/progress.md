# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction

**Paper inspected:** paper.pdf — "Subject-Identity Removal Does Not Improve Frozen EEG Foundation Model Transfer: A Negative Result" (FARS, Analemma)

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — extracted tables, figures, and results_section entries from the paper

**Summary of extraction:**
- Tables: 53 entries from Table 1 (cross-subject accuracy comparison across 8 methods) and Table 2 (per-subject breakdown for 9 subjects comparing EA baseline vs INLP-CV)
- Figures: 2 figures — Figure 1 (pipeline overview) and Figure 2 (INLP hyperparameter selection distribution)
- Results section: 1 entry — the body-text claim about 77.8% of folds selecting 1–3 INLP iterations and 92.6% selecting C=0.01; all other numerical results in the body duplicated values already captured in the tables

**Next session should:**
- Proceed to analysis/scoring (fs_analysis.json) if required by the pipeline
- No additional extraction needed; the paper is fully processed

## Session 2 — 2026-04-24
**Purpose:** analysis (static analysis + classification of all 56 claims)

**Files inspected:**
- paper.pdf (all 8 pages)
- exp/EXPERIMENT_RESULTS/baseline_ea_linear/RESULTS.json
- exp/EXPERIMENT_RESULTS/baseline_pca_k/RESULTS.json
- exp/EXPERIMENT_RESULTS/main_inlp/RESULTS.json
- exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json
- exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/inlp_optimized.csv
- fabscore_claude/progress.md

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with full classification of all 56 claims

**Key findings:**
1. Almost all claims (53/56) are `static_verifiable`: the RESULTS.json files in EXPERIMENT_RESULTS contain exact matching values for every table entry. The inlp_optimized.csv (27 rows, 9 subjects × 3 seeds) provides per-fold data that directly reproduces all Table 2 per-subject averages.
2. The inlp_optimized.csv has realistic per-fold execution timing (1269–1478s/fold for INLP, 863–913s/fold for CV), supporting it as a genuine execution artifact.
3. Figure 2 percentages (77.8% iter 1-3, 92.6% C=0.01) verified by directly counting rows in inlp_optimized.csv: 21/27 and 25/27 respectively.
4. **Key anomaly**: `exp/EXPERIMENT_RESULTS/main_inlp/RESULTS.json` is labeled as the INLP-10 (fixed 10 iterations) result file but actually contains the INLP-CV (optimized) results (mean=0.5629, mode=progressive_with_inner_CV). The INLP-10 numbers (55.18%, 7.19%, -1.09pp) appear only as `comparison.original_inlp_fixed10` reference values in the INLP-CV result files — no standalone primary execution artifact for INLP-10 exists.

**Classification summary:**
- static_verifiable: 53 (claims 1-17, 21-56)
- execution_required: 3 (claims 18, 19, 20 — EA+INLP-10 accuracy/std/delta)
- All others: 0

**Next session should:**
- If execution is needed: run `python exp/scripts/run_inlp_loso.py` to generate primary INLP-10 results and verify claims 18–20 (55.18% acc, 7.19% std, -1.09pp delta)

## Session 3 — 2026-04-24
**Purpose:** execution — verify claim 18 (EA + INLP-10, Accuracy = 55.18%)

**Files inspected:**
- exp/EXPERIMENT_RESULTS/main_inlp/RESULTS.json — contains INLP-CV results (mean=0.5629), not INLP-10; includes comparison.original_inlp_fixed10.mean=0.5518
- exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json — contains comparison.original_inlp.mean=0.5518
- exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json — references "original unoptimized INLP (10 iterations)" with -1.09pp delta vs EA
- exp/scripts/run_inlp_loso.py — the script that would generate primary INLP-10 results (max_iter=10, early_stop_threshold=1.25)
- exp/project/outputs/embeddings/ — directory does NOT exist (no npy files found)

**Execution artifacts created:**
- None (execution not possible — embeddings directory missing)

**Verdict for claim 18:**
- `Insufficient Evidence`
- The value 0.5518 (55.18%) appears consistently as a comparison reference in two RESULTS.json files, but:
  1. No standalone primary execution artifact for INLP-10 exists
  2. The embeddings needed to run `run_inlp_loso.py` are missing (`exp/project/outputs/` doesn't exist)
  3. Cannot reproduce the result without the embeddings

**Next session should:**
- For claims 19 and 20 (same blocker): same `Insufficient Evidence` verdict applies — embeddings are missing so `run_inlp_loso.py` cannot be run, and the std (0.0719=7.19%) and delta (-1.09pp) values also appear only as comparison references in INLP-CV result files

## Session 4 — 2026-04-24
**Purpose:** execution — verify claim 19 (EA + INLP-10, Std = 7.19%)

**Files inspected:**
- exp/EXPERIMENT_RESULTS/main_inlp/RESULTS.json — line 59: `comparison.original_inlp_fixed10.std_folds = 0.0719`
- exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json — line 55: same value 0.0719
- exp/EXPERIMENT_RESULTS/effectiveness_evaluation_report.md — line 33: table row "EA + INLP-10 (original) + Linear | 55.18 | 0.28 | 7.19 | -1.09"

**Execution artifacts created:**
- None (no commands run — same blocker as claim 18: embeddings missing)

**Verdict for claim 19:**
- `Insufficient Evidence`
- The value 0.0719 (7.19%) appears in main_inlp/RESULTS.json at comparison.original_inlp_fixed10.std_folds and in optimize_trace/iteration_0/RESULTS.json — same value in both. It also appears in the effectiveness_evaluation_report.md.
- However, these are all comparison reference values within the INLP-CV result files, not primary execution artifacts for INLP-10.
- The embeddings needed to run `run_inlp_loso.py` are missing (exp/project/outputs/ doesn't exist), so fresh reproduction is impossible.

**Next session should:**
- Verify claim 20 (EA + INLP-10, Delta = -1.09pp) — same blocker applies

## Session 5 — 2026-04-24
**Purpose:** execution — verify claim 20 (EA + INLP-10, Δ = −1.09 pp)

**Files inspected:**
- exp/EXPERIMENT_RESULTS/effectiveness_evaluation_report.md — line 33: "EA + INLP-10 (original) + Linear | 55.18 | 0.28 | 7.19 | -1.09"
- exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json — line 5: "The original unoptimized INLP (10 iterations) actively hurts performance at -1.09 pp"
- exp/EXPERIMENT_RESULTS/main_inlp/RESULTS.json — contains comparison.original_inlp_fixed10.mean=0.5518 (no -1.09 directly)
- fabscore_claude/progress.md — sessions 3 and 4 confirm same blocker: embeddings missing, INLP-10 primary artifact absent

**Execution artifacts created:**
- None (no commands run — same blocker as claims 18 and 19)

**Verdict for claim 20:**
- `Insufficient Evidence`
- The -1.09pp delta is consistently stated in effectiveness_evaluation_report.md (line 33) and effectiveness_evaluation_result.json (line 5), and is mathematically derivable as 55.18% − 56.27% = −1.09pp.
- However, all -1.09pp references are comparison reference values embedded within INLP-CV evaluation artifacts, not primary INLP-10 execution outputs.
- The embeddings needed to run run_inlp_loso.py are missing (exp/project/outputs/ doesn't exist), so fresh reproduction is impossible.
- Same blocker as claims 18 and 19 — Insufficient Evidence.

**Next session should:**
- All three INLP-10 claims (18, 19, 20) are now classified. No further work needed for this paper unless fresh reproduction becomes possible.
