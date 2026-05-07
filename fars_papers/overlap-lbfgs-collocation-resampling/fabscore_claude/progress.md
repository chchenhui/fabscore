# Progress Log

## Session 1
- **Purpose**: extraction
- **Paper inspected**: paper.pdf — "Overlap-Resampled L-BFGS for Physics-Informed Neural Networks" (Analemma / FARS)
- **Files created/updated**:
  - `fabscore_claude/fs_extracted.json` (created) — contains tables (25 entries across Tables 1–3), figures (2 entries: Figures 1 and 2), and results_section (7 entries from Sections 4.2–4.6)
- **Next session**: Run scoring/evaluation against fs_extracted.json, or perform any additional analysis (e.g., claim verification, comparison with baseline extractions).

## Session 2 — 2026-04-24
- **Purpose**: analysis (static code auditing of all 34 extracted claims)
- **Files inspected**:
  - `exp/EXPERIMENT_RESULTS/adam_resampling_ice_shelf/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/adam_lbfgs_ice_shelf/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/overlap_lbfgs_ice_shelf/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/adam_resampling_poisson2d/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/adam_lbfgs_poisson2d/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/overlap_lbfgs_poisson2d/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/ablation_o025/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/from_scratch_overlap_lbfgs_poisson2d/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/lbfgs_diagnostics/RESULTS.json`
  - `exp/overlap_lbfgs_pinn/trainers/pinn_trainer.py` (partial)
- **Files created/updated**:
  - `fabscore_claude/fs_analysis.json` (created) — full classification of all 34 claims
- **Summary**: All 34 claims classified as `static_verifiable`. Every numerical claim in Tables 1–3 and the results section was found to exactly match values in the corresponding RESULTS.json artifact files. Figure 2 data matches lbfgs_diagnostics/RESULTS.json. Figure 1 is a schematic pipeline supported by pinn_trainer.py code structure.
- **Recommended next step**: No further action needed for analysis. Proceed to scoring phase.
