# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** paper.pdf (Suppression-Contrast Tokens: Evaluating Reverse Layer-Contrast for Secret Elicitation)
**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries

**Summary of extraction:**
- Tables: 40 entries extracted from Table 1 (Taboo/Direct results), Table 2 (User Gender/Direct results), and Table 3 (Pre-registered criteria evaluation)
- Figures: 1 entry — Figure 1 (SCT overview diagram)
- Results section: 1 entry — gold model top-200 coverage (2%) from Section 4.6 Limitations; all other numerical claims in the body text were already captured in tables

**Next session should:**
- Proceed to analysis/scoring phase using the extracted results in `fs_extracted.json`

## Session 2 — 2026-04-24
**Purpose:** analysis (static analysis of all 42 claims)

**Files inspected:**
- `exp/EXPERIMENT_RESULTS/taboo_direct_dola_direction/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/taboo_direct_logit_lens_constrained/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/taboo_direct_sct/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/user_gender_direct_logit_lens_constrained/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/user_gender_direct_sct/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json`
- `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_report.md`
- `exp/EXPERIMENT_RESULTS/taboo_direct_sct/REPORT.md`
- `exp/EXPERIMENT_RESULTS/optimize_trace/iteration_1/RESULTS.json`
- `exp/sct/extraction/sct_scorer.py`
- `exp/sct/extraction/extract_activations.py`

**Files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with classifications for all 42 claims

**Summary of classifications:**
All 42 claims classified as `static_verifiable`:
- Claims 1–8 (DoLa-direction Taboo): All match `taboo_direct_dola_direction/RESULTS.json` exactly
- Claims 9–16 (Logit Lens Taboo): All match `taboo_direct_logit_lens_constrained/RESULTS.json` exactly
- Claims 17–24 (SCT Ours Taboo): All match `taboo_direct_sct/RESULTS.json` exactly (best variant = filtered_ll)
- Claims 25–30 (Logit Lens Gender): All match `user_gender_direct_logit_lens_constrained/RESULTS.json` exactly
- Claims 31–36 (SCT Gender): All match `user_gender_direct_sct/RESULTS.json` exactly
- Claim 37 (Premise ~9.3% FAIL): Confirmed by `effectiveness_evaluation_report.md`
- Claim 38 (+23.1% rel, +1.0pp abs FAIL): Confirmed by `optimize_trace/iteration_1/RESULTS.json` baseline_comparison
- Claim 39 (Direction control PASS): Directly computable from DoLa + LL RESULTS.json values
- Claim 40 (CIs overlap FAIL): Derived from std values in both RESULTS.json files; confirmed in report.md
- Claim 41 (Figure 1 description): Consistent with `sct_scorer.py` + `extract_activations.py` code
- Claim 42 (Gold 2% ceiling): Confirmed by LL RESULTS.json TR@5=TR@20=0.02 and report.md

**Key observation:** The "SCT (Ours)" Table 1 results are stored under `taboo_direct_sct/RESULTS.json` with `method: "filtered_ll"` because the best SCT optimization variant turned out to be filtered logit lens, not the original suppression-contrast scoring. The paper labels this as "SCT (Ours)" (presumably as the best variant of their method exploration), and all numbers match the stored results exactly.

**Next step:** No further action needed. All claims are static_verifiable from the stored JSON result files.
