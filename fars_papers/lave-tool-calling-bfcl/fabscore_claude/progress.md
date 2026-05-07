# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** paper.pdf (lave-tool-calling-bfcl) — "Syntax Constraints Are Not Enough: Semantic Errors Dominate Diffusion LM Tool-Calling Failures"
**Files created:**
- `fabscore_claude/fs_extracted.json` — full extraction of tables, figures, and results_section claims

**Summary of extraction:**
- Tables: 49 entries covering Table 1 (main results: success rate, AST parse rate, time, overhead for 3 decoding conditions), Table 2 (error taxonomy: parse failure, wrong function, wrong arguments), and Table 3 (per-category success rates across 7 BFCL categories)
- Figures: 3 figures (Figure 1: experimental overview diagram; Figure 2: error distribution bar chart; Figure 3: per-category success rates bar chart)
- Results section: 6 claims capturing derived metrics not explicitly in tables (0.57pp improvement, +4.96pp AST parse rate, 20% speed advantage, Best-of-2 overhead, Qwen-8B 87.5% baseline, 50.74pp gap)

**Next session should:** run analysis/scoring (fs_analysis.json) comparing extracted results against paper claims or a reference.

## Session 2 — 2026-04-24
**Purpose:** analysis (static analysis / claim classification)
**Files inspected:**
- paper.pdf (via prior extraction)
- exp/EXPERIMENT_RESULTS/condition_a_unconstrained/RESULTS.json
- exp/EXPERIMENT_RESULTS/condition_b_best_of_2/RESULTS.json
- exp/EXPERIMENT_RESULTS/condition_c_lave_cfg/RESULTS.json
- exp/EXPERIMENT_RESULTS/failure_taxonomy/RESULTS.json
- exp/EXPERIMENT_RESULTS/category_breakdown_timing/RESULTS.json
- exp/EXPERIMENT_RESULTS/effectiveness_evaluation_report.md
- exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json
- exp/bfcl_cfg_diffusion/inference/*.py (run_unconstrained.py, run_best_of_2.py, run_lave_cfg.py)
- exp/bfcl_cfg_diffusion/analysis/plot_*.py

**Files created/updated:**
- `fabscore_claude/fs_analysis.json` — full classification of 58 claims

**Summary of classifications:**
- **static_verifiable: 57** — All 49 table claims (Table 1: 12 metrics, Table 2: 9 taxonomy metrics, Table 3: 28 per-category values), all 3 figure claims, and 5 of 6 results-section claims are directly verified against stored RESULTS.json files. All numbers match exactly with the stored experimental outputs (means, stds, times, overhead ratios, deltas). The 3 RESULTS.json files and category_breakdown_timing/RESULTS.json provide comprehensive, internally consistent coverage of all reported metrics.
- **insufficient_evidence: 1** — Claim 57 ("Qwen-8B achieves 87.5%") is an external baseline cited from 'Bitter Lesson Table 2'; no repo-native code evaluates Qwen-8B, only references the number.
- No obvious_hallucination, no_code_files, execution_required, or error findings.

**Key finding:** The repository contains complete, well-structured result JSON files (condition_a, condition_b, condition_c, failure_taxonomy, category_breakdown_timing) that match all paper claims exactly. The only unverifiable claim is the external Qwen-8B baseline (87.5%) which is cited from a different paper.

**Recommended next step:** No further verification needed for table/figure claims. If claim 57 needs verification, check the BFCL leaderboard or the cited Bitter Lesson paper directly.
