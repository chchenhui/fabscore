## Session: 2026-04-24 — extraction

**Purpose:** Extract experimental results from the paper.

**Paper inspected:** `results/paper.md`
- Title: *AutoSpurDetect: Automated Benchmark and Robustification for Unknown Spurious Correlations in Multimodal Models*
- Proof-of-concept on SST-2 sentiment classification with DistilBERT.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (13 entries from two unnumbered tables in Sections 5.1 and 5.2), figures (2 entries: figures/loss.png and figures/accuracy.png), and results_section (empty — all numerical claims in the Analysis section were duplicates of table values).

**Notes:**
- The paper has no explicitly numbered tables; referenced as "Table 5.1" and "Table 5.2" by section.
- All numerical claims in Section 6 (Analysis) — clean accuracy 82.50%→85.50% and counterfactual acc 78.50% — are already captured in the tables, so results_section is empty.

**Next session should:**
- Proceed to scoring/analysis phase (fs_analysis.json, fs_summary.json) if applicable.

---

## Session: 2026-04-24 — analysis

**Purpose:** Static analysis — classify all 15 extracted claims against repository artifacts.

**Files inspected:**
- `results/paper.md` — full paper with Tables 5.1 and 5.2
- `codex/experiment.py` — main training/evaluation script (baseline + robust pipelines)
- `codex/log.txt` — execution log with per-epoch loss/accuracy records
- `codex/results_baseline.json` — saved baseline metrics
- `codex/results_robust.json` — saved robust+baseline+cf metrics
- `codex/results_sensitivity.json` — saved sensitivity/CF metrics
- `results/figures/loss.png`, `results/figures/accuracy.png` — generated figure PNGs

**JSON files created:**
- `fabscore_claude/fs_analysis.json` — all 15 claims classified as `static_verifiable`

**Classification summary (15 total):**
- `static_verifiable`: 15 — all claims
- Claims 1–13 (table values): exact matches in codex/results_robust.json, results_baseline.json, results_sensitivity.json, and log.txt (values match to 4 decimal places).
- Claims 14–15 (figures): PNG artifacts exist in results/figures/ AND underlying data arrays in codex/results_robust.json match what the figures plot.
- No conflicts, fabrications, or missing code paths detected.

**Notable:** codex/log.txt has 3 repeated copies of baseline epoch logs, suggesting multiple script runs without log clearing. Final values are consistent with artifacts and paper claims.

**Recommended next step:** Generate fs_summary.json if required by the pipeline.
