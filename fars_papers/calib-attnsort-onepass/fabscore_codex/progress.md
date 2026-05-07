## Session 2026-05-01 - extraction

- Purpose: extraction.
- Inspected: `paper.pdf`, a 7-page PDF titled "Position Bias Correction Is Insufficient for One-Pass Attention Sorting"; focused on Tables 1-2, Figures 1-2, and Sections 4.2-4.3 of the Experiments section.
- Created or updated: `fabscore_codex/fs_extracted.json` with table values, figure captions, and deduplicated results-section numerical claims; `fabscore_codex/progress.md` with this session entry.
- Next session should: verify downstream ingestion of `fabscore_codex/fs_extracted.json` or compare it against any human annotation target if available.

## Session 2026-05-01 - analysis

- Purpose: analysis.
- Inspected: `paper.pdf` via repository-local extracted claim context plus direct binary-string search fallback; repository code paths `exp/debiased_attnsort/src/eval_pipeline.py`, `exp/debiased_attnsort/src/debiased_sorting.py`, `exp/debiased_attnsort/src/bias_estimation.py`, `exp/debiased_attnsort/src/eval_yarn.py`; runner scripts `exp/debiased_attnsort/scripts/run_yarn_3seeds.sh`, `exp/debiased_attnsort/scripts/run_optimized_yarn_debiased.sh`; saved artifacts `exp/EXPERIMENT_RESULTS/no_sorting_baseline/RESULTS.json`, `exp/EXPERIMENT_RESULTS/attnsort_k1_baseline/RESULTS.json`, `exp/EXPERIMENT_RESULTS/debiased_k1/RESULTS.json`, `exp/EXPERIMENT_RESULTS/attnsort_k5_baseline/RESULTS.json`, `exp/EXPERIMENT_RESULTS/yarn_llama2_64k_all_conditions/RESULTS.json`, `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json`, `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_report.md`, `exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json`, and `exp/EXPERIMENT_RESULTS/optimize_trace/optimize_summary_0.json`.
- Created or updated: `fabscore_codex/fs_analysis.json`; `fabscore_codex/progress.md`.
- Classification summary: 43 claims marked `static_verifiable`; 9 claims marked `obvious_hallucination` with `experiment_fabrication`. The static conflict is concentrated in the YaRN `debiased_k1` reporting path: repository artifacts for the reported Table 2 debiased results use `debias_mode="divisive"` and `strategy="full_sort_by_debiased"` in `exp/debiased_attnsort/src/eval_yarn.py` and `exp/EXPERIMENT_RESULTS/yarn_llama2_64k_all_conditions/RESULTS.json`, which conflicts with the paper's subtractive one-pass method description in Figure 1 / narrative.
- Recommended next step for the next session: review whether the benchmark policy should treat the YaRN `debiased_k1` mismatch as claim-level `experiment_fabrication` exactly as recorded here, or whether a narrower interpretation should separate conceptual-method claims from numeric-result claims before aggregation.
