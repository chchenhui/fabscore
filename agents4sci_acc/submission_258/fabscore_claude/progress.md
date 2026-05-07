# Progress Log

## Session 1
- **Purpose**: extraction
- **Paper inspected**: 258_Green_by_Design_Energy_Gui.pdf (10-page paper on energy-guided reranking of LLM-generated programs)
- **JSON files created/updated**: `fabscore_claude/fs_extracted.json`
- **Summary**:
  - Extracted 6 table entries from Table 2 (aggregate energy savings and runtime penalty metrics)
  - Extracted 2 figures (Figure 1: pipeline overview; Figure 2: runtime penalty and energy savings distributions)
  - Extracted 6 results_section claims including: >98% energy reduction for Matrix Multiplication, 11.49% gain for Array Sum vs Best-Time, p-values for statistical significance tests (p<0.001 and p=0.0039), coefficient of variation (8.62% runtime / 10.75% energy), max runtime penalty (~0.023s), and scale-consistency rate (30% of tasks)
- **Next session**: verification or downstream scoring; no additional extraction needed unless supplementary material becomes available

## Session 2
- **Purpose**: analysis (static code audit)
- **Date**: 2026-04-03
- **Files/context inspected**:
  - `258_Green_by_Design_Energy_Gui.pdf` (full paper read)
  - `Agent4Sci--energy-llm-study-main/phase4_reranking/phase4_analysis/overall_summary.csv`
  - `Agent4Sci--energy-llm-study-main/phase4_reranking/phase4_analysis/per_task_per_scale_summary.csv`
  - `Agent4Sci--energy-llm-study-main/phase4_reranking/phase4_analysis/overall_analysis_report.txt`
  - `Agent4Sci--energy-llm-study-main/phase5_evaluation/phase5_evaluation/aggregate_summary_stats.csv`
  - `Agent4Sci--energy-llm-study-main/phase5_evaluation/phase5_evaluation/task_level_summary.csv`
  - `Agent4Sci--energy-llm-study-main/phase5_evaluation/phase5_evaluation/evaluation_summary_report.txt`
  - `Agent4Sci--energy-llm-study-main/phase5_evaluation/phase4_analysis/detailed_selections.csv`
  - `Agent4Sci--energy-llm-study-main/phase5_evaluation/phase5_evaluation.py`
  - `Agent4Sci--energy-llm-study-main/phase4_reranking/phase4_reranker.py`
  - `Agent4Sci--energy-llm-study-main/phase5_evaluation/outputs/all_results/08_json_parsing_filtering_results.csv`
  - PNG figure artifacts in `phase5_evaluation/phase5_evaluation/`
- **JSON files created/updated**: `fabscore_claude/fs_analysis.json`
- **Summary of classifications**:
  - **static_verifiable** (13 claims): Claims 1-12 and 14. All Table 2 values (44.69%, 39.05%, 1.86%, 0.00%, 0.0012s, 0.0000s) are directly confirmed in aggregate_summary_stats.csv and evaluation_summary_report.txt. The Matrix Multiplication >98% energy reduction (claim 9) is 98.30% from per_task_per_scale_summary.csv. The Array Sum 11.49% savings (claim 10) is confirmed at 0.11494558. The p-values (claim 11) are confirmed as 0.00000 and 0.00391. The CV values (claim 12: 8.62%/10.75%) and scale-consistency (claim 14: 3/10 tasks = 30%) are confirmed in evaluation_summary_report.txt. Figure 1 (claim 7) is verified by the repo's 5-phase pipeline structure. Figure 2 (claim 8) has both PNG artifacts and underlying CSV data.
  - **obvious_hallucination/result_fabrication** (1 claim): Claim 13 ("largest penalty ≈0.023 seconds in JSON Parsing"). The detailed_selections.csv shows JSON Parsing max runtime penalty is ~0.002013 seconds at scale 1M (not 0.023). The actual largest penalty across all tasks is ~0.01854 seconds in Stable Sorting (not JSON Parsing). The value 0.023 corresponds to the energy savings fraction (0.023295) for JSON Parsing at scale 10K — misreported as runtime penalty in seconds.
- **Recommended next step**: No further execution needed for most claims. Claim 13 is a clear result_fabrication verified by static analysis.
