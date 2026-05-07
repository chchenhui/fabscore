# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction

**Paper inspected:** paper.pdf — "Position Bias Correction is Insufficient for One-Pass Attention Sorting" (Analemma / FARS)

**Summary:** Extracted all experimental results from the paper, which evaluates Debiased One-Pass Attention Sorting on the SynthWiki@28K benchmark across two models (LLaMA-2-7B-32K-Instruct and YaRN-Llama-2-7b-64k) comparing No Sorting, Attn Sort k=1, Debiased k=1 (proposed), and Attn Sort k=5.

**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with 40 table entries (Tables 1 & 2, all seed and mean values), 2 figure entries (Figures 1 and 2), and 7 results_section claims covering paired-comparison counts, gap metrics (0.67pp, 8.67pp, 14.84pp, 37%), gold document position values (164.85, 164.86, 165.77), and the k=2 intermediate accuracy (95.83%).

**Next session should:** Run analysis/scoring of the extracted results against any reference or evaluation criteria.

## Session 2 — 2026-04-24
**Purpose:** analysis

**Files/context inspected:**
- `exp/EXPERIMENT_RESULTS/no_sorting_baseline/RESULTS.json` — LLaMA no_sorting per-seed accuracies and prefill count
- `exp/EXPERIMENT_RESULTS/attnsort_k1_baseline/RESULTS.json` — LLaMA attnsort k=1 per-seed accuracies, mean gold position (164.858)
- `exp/EXPERIMENT_RESULTS/attnsort_k5_baseline/RESULTS.json` — LLaMA attnsort k=5 per-seed accuracies, intermediate accuracy per iter, mean gold position per iter
- `exp/EXPERIMENT_RESULTS/debiased_k1/RESULTS.json` — LLaMA debiased k=1 per-seed accuracies, paired comparison (0W/600T/0L)
- `exp/EXPERIMENT_RESULTS/yarn_llama2_64k_all_conditions/RESULTS.json` — YaRN all conditions (no_sorting, attnsort_k1, attnsort_k5, debiased_k1)
- `exp/debiased_attnsort/src/` — bias_estimation.py, debiased_sorting.py, eval_pipeline.py, eval_yarn.py (code structure)

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with all 49 claims classified

**Summary of classifications:**
All 49 claims are classified as `static_verifiable`. Every table value, figure data point, and results-section metric is directly confirmed by the RESULTS.json files in `exp/EXPERIMENT_RESULTS/`. Specifically:
- Claims 1–20 (Table 1, LLaMA-2-7B-32K-Instruct): all per-seed accuracies, means, stds, and prefill counts match RESULTS.json files exactly.
- Claims 21–40 (Table 2, YaRN-Llama-2-7b-64k): all per-seed accuracies, means, stds, and prefill counts match yarn_llama2_64k_all_conditions/RESULTS.json exactly.
- Claim 41 (Figure 1): prefill count values (k=5→6, debiased k=1→2) confirmed by RESULTS.json; methodology confirmed by debiased_sorting.py and bias_estimation.py.
- Claim 42 (Figure 2): intermediate accuracy per iter and gold position per iter confirmed by attnsort_k5_baseline/RESULTS.json.
- Claims 43–49 (results section): paired comparison (0W/600T/0L), gap values (0.67pp, 8.67pp, 14.84pp), 37% closure, mean gold positions (164.86, 164.85, 165.77) all confirmed.

**Next session:** No further action needed; all claims are static_verifiable with concrete file evidence.
