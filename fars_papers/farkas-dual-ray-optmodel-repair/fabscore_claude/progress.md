# Progress Log

## Session 1
- **Purpose**: extraction
- **Paper inspected**: paper.pdf — "Farkas Dual Rays Do Not Improve LLM-Based Optimization Model Repair" (Analemma/FARS)
- **Files created/updated**:
  - `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries
- **Summary**:
  - Extracted 27 table entries from Table 1 (repair method comparison) and Table 2 (inference scaling vs repair)
  - Extracted 3 figure entries (Figures 1, 2, 3)
  - Extracted 3 results_section claims not duplicated in tables: (1) per-instance Pass@1 breakdown for EasyLP 246 across model sizes, (2) 6.26 percentage point gap between Best-of-2 and best repair method, (3) IIS size breakdown of the 7 repaired instances
- **Next session**: scoring/evaluation or further analysis as needed

## Session 2
- **Purpose**: analysis
- **Files inspected**:
  - `exp/results/summary_table.json` — comprehensive table of all 8 experiment conditions with exact metrics
  - `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json` — verdict and summary
  - `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_report.md` — full report with per-instance IIS sizes for 7 repaired instances
  - `exp/results/effectiveness_conclusion.md` — detailed conclusion with all comparisons
  - `exp/EXPERIMENT_RESULTS/attempt0_baseline/RESULTS.json` — baseline metrics
  - `exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json` — 72B/32B optimized results
  - `exp/EXPERIMENT_RESULTS/optimize_trace/optimize_summary_0.json` — optimization summary
  - `exp/EXPERIMENT_RESULTS/iis_topk_baseline/REPORT.md` — IIS statistics
- **Files created/updated**:
  - `fabscore_claude/fs_analysis.json` — created with full classification of all 33 claims
- **Summary of classifications**:
  - `static_verifiable`: 32 claims (indices 1–29, 31–33)
    - Claims 1–27 (table): all numeric values directly match summary_table.json
    - Claim 28 (Figure 1): pipeline description verified by code modules in dualrayrank/
    - Claim 29 (Figure 2): specific numbers (65.12%, 58.86%, 6.26pp) in summary_table.json
    - Claims 31–33 (results_section): per-instance data in effectiveness_evaluation_report.md
  - `execution_required`: 1 claim (index 30, Figure 3)
    - No PNG/PDF figure artifact exists; no complete per-IIS-size distribution file for all 31 infeasible instances; denominators for success rate by IIS bin are unknown without running the experiment scripts
- **No hallucinations or fabrications found**: all checked values match repository artifacts
- **Next session**: execution of Figure 3 data regeneration if needed (run_iis_topk_repair.py to get all 31 IIS sizes)

## Session 3
- **Purpose**: execution — verify claim 30 (Figure 3: repair success rate by IIS size)
- **Files inspected**:
  - `exp/results/summary_table.json` — contains IIS sizes for the 16 truncation-regime instances (IIS > 5) and repaired instance metadata
  - `exp/EXPERIMENT_RESULTS/iis_topk_baseline/RESULTS.json` — IIS stats (min=1, max=33, mean=7.0), only 1 repaired (EasyLP_425, IIS=2)
  - `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_report.md` — 7 repaired instances with IIS sizes: 1, 1, 2, 2, 4, 4, 6
  - `exp/dualrayrank/` — checked for plotting scripts; none exist for Figure 3
- **Key data extracted**:
  - 31 infeasible instances total: 16 truncation regime (IIS > 5), 15 non-truncation (IIS ≤ 5)
  - Truncation regime IIS sizes from summary_table.json: {61:12, 63:7, 64:9, 66:13, 72:11, 75:7, 81:10, 84:9, 85:11, 86:6, 93:11, 95:7, 97:11, 98:7, 131:33, 134:16}
  - Repaired instances (72B): IIS=1 (×2), 2 (×2), 4 (×2), 6 (×1); 4/7 have IIS ≤ 2
  - Truncation regime repair success: 1/16 = 6.25% (near-zero as claimed)
  - Non-truncation repair success: 6/15 = 40%
- **Verdict**: Insufficient Evidence
  - The qualitative claims in the Figure 3 description are supported by available data: repairs concentrate in small-IIS instances (IIS 1–2), near-zero success in truncation regime (6.25%)
  - However, the complete per-IIS-size distribution for all 31 instances is not available (IIS sizes of 9 non-truncated, non-repaired instances are missing), no plotting script exists in the repository, and Figure 3 as a figure cannot be generated or verified from the existing artifacts
  - The key assertions (repairs concentrate at IIS 1–2, near-zero in IIS > 5) are qualitatively confirmed
- **No command executed** (all needed evidence extracted from existing JSON artifacts)
- **Next session**: N/A — claim 30 finalized
