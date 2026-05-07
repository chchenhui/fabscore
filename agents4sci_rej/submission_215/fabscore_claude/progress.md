# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** 215_Agentic_AutoSurvey_Let_Age.pdf (Agentic AutoSurvey: Let Agentic LLM Survey LLMs)
**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries

**Summary of extraction:**
- Table 1 (capability comparison): checkmarks/X only, no numerical values extracted
- Table 2 (performance comparison): 60 entries extracted covering all topics × dimensions × systems, including improvement rows
- Figures: 3 figures extracted (Figure 1: architecture, Figure 2: cluster distributions, Figure 3: papers/clusters per topic)
- results_section: empty — all numerical performance metrics in Section 3.2 body text are duplicates of Table 2 values; Discussion section score references (7.74/10 vs 8.43/10) also already in Table 2

**Next session should:** review or verify extraction completeness; potentially run analysis/scoring on the extracted data.

## Session 2 — 2026-04-24
**Purpose:** analysis (static code auditing)
**Files inspected:**
- `215_Agentic_AutoSurvey_Let_Age.pdf` — paper content
- `submission-code/results/multi-agent/FINAL_SCORES.json` — primary artifact for Agentic AutoSurvey Table 2 claims
- `submission-code/results/autosurvey/FINAL_SCORES.json` — primary artifact for AutoSurvey Table 2 claims
- `submission-code/results/COMPARISON_SUMMARY.md` — summary with overall averages
- `submission-code/evaluation/all_system_scores.json` — partially populated scores (many zeros)
- `submission-code/evaluation/multi_agent_scores.json` — partial 12-dim scores, LLM Agents only
- `submission-code/evaluation/autosurvey_category_scores.json` — 5-topic breakdown (missing LLM Agents, different values from FINAL_SCORES.json)
- `submission-code/evaluation/evaluation_results/Instruction_Tuning_evaluation.json` — baseline evaluation (matches autosurvey_category_scores, differs from FINAL_SCORES.json)
- `submission-code/evaluation/evaluation_report.json` — multi-agent comprehensive evaluation with different scores
- `submission-code/src/enhanced_evaluator.py` — 12-dimension heuristic evaluator code
- `submission-code/src/generate_paper_figures.py` — figure generation script with hardcoded cluster/paper data

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with full classifications

**Summary of classifications (63 total claims):**
- `static_verifiable`: 60 claims (indices 1-27, 29-51, 53-60, 62-63)
  - Most Table 2 per-topic scores (Core/Write/Depth/Avg) match exactly in FINAL_SCORES.json files
  - Average row AutoSurvey values (53-56) match FINAL_SCORES.json category_averages exactly
  - Average row Agentic values (49-51) consistent with arithmetic mean of per-topic values
  - Improvement % claims (57-60) arithmetically derivable from averages
  - Figure 2 and Figure 3 data are hardcoded in generate_paper_figures.py and match paper descriptions
- `insufficient_evidence`: 2 claims (indices 28, 52)
  - Claim 28 (Synthetic Data Agentic Avg=7.79): FINAL_SCORES.json shows 7.69; COMPARISON_SUMMARY.md also shows 7.69. Paper uses simple arithmetic mean of Core/Write/Depth components (7.793≈7.79), but no artifact confirms this value.
  - Claim 52 (Overall Agentic Avg=8.18): follows from claim 28 discrepancy; repo says 8.17 consistently.
- `no_code_files`: 1 claim (index 61)
  - Figure 1 (architecture diagram): no underlying numerical data file or figure generation script for this specific figure.

**Key observations:**
- Multiple conflicting evaluation artifacts exist: FINAL_SCORES.json (matches paper) vs evaluation_results/*.json and autosurvey_category_scores.json (show different breakdown scores)
- generate_paper_figures.py has RLHF=443 papers (matching paper Figure 3), but FINAL_SCORES.json says 1334 papers for RLHF (used for survey generation)
- The evaluation scores appear plausible but their generation pipeline is not fully traceable

**Recommended next step:** No execution needed for most claims; claims 28 and 52 could potentially be resolved by re-running the evaluation pipeline on synthetic_data surveys to compute the actual average score.
