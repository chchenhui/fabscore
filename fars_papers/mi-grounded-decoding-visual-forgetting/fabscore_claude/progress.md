# Session Log

## Session 1
- **Purpose**: extraction
- **Paper inspected**: paper.pdf — "Time-Varying Mutual Information Decoding for Mitigating Visual Forgetting in Vision-Language Models" (FARS / Analemma)
- **JSON files created**:
  - `fabscore_claude/fs_extracted.json` — extracted tables (21 entries across 4 tables), figures (2 entries: Figure 1 and Figure 2), and results_section claims (2 entries from mechanistic analysis text not duplicated in tables)
- **Next session**: Run analysis/scoring (fabscore) on the extracted results. Consider checking whether any ablation or PDM-H trajectory numbers were missed from additional subsections.

## Session 2 (2026-04-24)
- **Purpose**: analysis (static code audit)
- **Files inspected**:
  - `exp/EXPERIMENT_RESULTS/vanilla_vlaa_thinker_7b/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/visual_replay_vlaa_thinker_7b/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/adaptive_mi_vlaa_thinker_7b/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/generality_check_qwen25vl/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/ablation_fixed_vs_adaptive/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/pdm_h_analysis/RESULTS.json`
  - `exp/mi_decoding/decoding/mi_decoding.py` (implementation verification)
  - `exp/mi_decoding/evaluation/metrics.py`, `pdm_h.py`
  - Full repository directory tree via Explore agent
- **JSON files created**:
  - `fabscore_claude/fs_analysis.json` — analysis classifications for all 25 claims
- **Summary of classifications**:
  - All 25 claims classified as `static_verifiable`
  - Claims 1–21 (table claims): Exact matching values found in respective RESULTS.json files:
    - vanilla_vlaa_thinker_7b/RESULTS.json for claims 1-2, 7-9
    - visual_replay_vlaa_thinker_7b/RESULTS.json for claims 3-4, 10-12
    - adaptive_mi_vlaa_thinker_7b/RESULTS.json for claims 5-6, 13-15
    - generality_check_qwen25vl/RESULTS.json for claims 16-17
    - ablation_fixed_vs_adaptive/RESULTS.json for claims 18-21
  - Claim 22 (Figure 1): mi_decoding.py implements dual-pass, threshold-α, time-varying γ_t=exp(-λ(t+t₀)) exactly as described
  - Claim 23 (Figure 2): pdm_h_analysis/RESULTS.json contains full trajectory data; AUC ratios 0.9755 (MMStar) and 0.9471 (HallusionBench) match paper's 0.976 and 0.947
  - Claims 24-25 (results_section): Manually verified decline calculations from PDM-H trajectory arrays: vanilla −0.088 ≈ −0.089, adaptive_mi −0.036 match; AUC ratios match exactly
- **Recommended next step**: No further analysis needed. All claims are fully supported by static artifacts. If execution verification is desired, the full eval pipeline exists under `exp/mi_decoding/scripts/` with corresponding shell scripts.
