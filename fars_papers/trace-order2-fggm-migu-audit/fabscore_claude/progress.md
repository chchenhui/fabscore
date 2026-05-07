# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** paper.pdf (trace-order2-fggm-migu-audit) — "Order-Robustness Audit of Gradient Masking Methods for Continual Learning in LLMs" (FARS / Analemma)
**Files created:**
- `fabscore_claude/fs_extracted.json` — extracted tables (36 entries from Tables 1–3), figures (2 entries: Figures 1–2), and results_section claims (3 entries from sections 4.4 and 4.5 not duplicated in tables)

**Summary of extracted content:**
- Table 1: sanity-check reproduced vs published TRACE-OP for SFT, MIGU, FGGM on default order
- Table 2: main TRACE-OP and BWT results for both orderings, plus order-sensitivity delta
- Table 3: seed-level paired FGGM vs MIGU comparison on Order 2
- Figure 1: experimental framework diagram
- Figure 2: consecutive Jaccard similarity bar charts for FGGM masks under both orderings
- Results section: 1-σ CI non-overlap values, Jaccard similarity sequence values from mask overlap analysis

**Next session should:** perform analysis/scoring of the extracted results (fs_analysis.json).

## Session 2 — 2026-04-24
**Purpose:** analysis
**Files inspected:**
- `exp/EXPERIMENT_RESULTS/sft_default_order/RESULTS.json` — SFT default order metrics (TRACE-OP=49.31, BWT=-34.25)
- `exp/EXPERIMENT_RESULTS/migu_default_order/RESULTS.json` — MIGU default order metrics (TRACE-OP=47.43, BWT=-8.05)
- `exp/EXPERIMENT_RESULTS/fggm_default_order/RESULTS.json` — FGGM default order metrics (TRACE-OP=45.84, BWT=-8.52)
- `exp/EXPERIMENT_RESULTS/sft_order2/RESULTS.json` — SFT Order 2 metrics (mean=39.82±0.47, BWT=-5.30±0.59)
- `exp/EXPERIMENT_RESULTS/migu_order2/RESULTS.json` — MIGU Order 2 metrics (mean=43.72±0.13, BWT=-1.07±0.65)
- `exp/EXPERIMENT_RESULTS/fggm_order2/RESULTS.json` — FGGM Order 2 per-seed and aggregate metrics
- `exp/EXPERIMENT_RESULTS/mask_overlap_analysis/RESULTS.json` — Jaccard similarity data for Figure 2
- `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json` — Final evaluation verdict

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — All 41 claims classified

**Summary of classifications:**
- All 41 claims classified as `static_verifiable`
- Tables 1-3 (claims 1-36): All numerical values exactly match RESULTS.json files. Arithmetic checked: means, stds, diffs, deltas all internally consistent.
- Figure 1 (claim 37): Schematic figure; the full experimental framework (FGGM, MIGU, SFT, TRACE, two orderings) is present in the repository code.
- Figure 2 (claim 38): Underlying Jaccard data in mask_overlap_analysis/RESULTS.json confirms 0.368 for NumGLUE-cm→NumGLUE-ds; note the paper's "monotonic increase" description is slightly imprecise (0.492→0.473 is a non-monotone step), but the data is present.
- Results section CI intervals (claim 39): [39.71,41.83] vs [43.59,43.85] directly computed from aggregate statistics in RESULTS.json.
- Claims 40-41: Jaccard pattern data confirmed in mask_overlap_analysis/RESULTS.json.

**Recommended next step:** No further action needed; analysis complete. All 41 claims are static_verifiable with strong support from RESULTS.json files.
