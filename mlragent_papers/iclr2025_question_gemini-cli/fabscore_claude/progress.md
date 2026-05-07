# Progress Log

## Session 2 — 2026-04-24
**Purpose:** analysis

**Files inspected:**
- results/paper.md — full paper content, Table 2 with AUROC values
- results/experiment_results.json — stored AUROC values matching all three paper claims exactly
- results/log.txt — experiment execution log confirming run on CUDA
- results/hallucination_detection_auroc.png — figure artifact
- gemini/run_experiment.py — full implementation: DUnEModel, MCDropoutModel, BaselineModel, evaluate_model(), plot_results()
- fabscore_claude/fs_extracted.json — prior extraction claims

**JSON files created:**
- `fabscore_claude/fs_analysis.json` — all 4 claims classified as `static_verifiable`

**Classification summary:**
- Claims 1-3 (Table 2 AUROC values): `static_verifiable` — exact values confirmed in results/experiment_results.json (0.48621880727477795, 0.5, 0.5040532919780064) matching paper-reported 0.4862, 0.5000, 0.5041 respectively
- Claim 4 (figure): `static_verifiable` — PNG artifact exists AND underlying data in experiment_results.json provides the plotted values; plot_results() in run_experiment.py is the generation path

**Next session:**
- No further analysis needed; all claims are static_verifiable. Execution step can optionally re-run gemini/main.py to confirm reproducibility.

---

## Session 1 — 2026-04-24
**Purpose:** extraction

**Paper inspected:** results/paper.md
- Paper: "Disentangled Uncertainty Estimation for Hallucination-Aware Generation" (DUnE-LLM)
- Evaluated on TruthfulQA hallucination detection benchmark (AUROC metric)
- Compares DUnE-LLM vs. Baseline (Entropy) vs. MC Dropout

**JSON files created:**
- `fabscore_claude/fs_extracted.json` — extracted tables (3 entries from Table 2), figures (1 entry), and results_section (empty; all numerical results were already captured in Table 2)

**Next session should:**
- Run analysis/scoring against the extracted results (fs_analysis step)
