# Progress Log

---

## Session: 2026-04-24

**Purpose:** extraction

**Paper inspected:** `paper.pdf` — "Anisotropic Spectral Error Dressing for Calibrated Ensemble Weather Forecasts" (FARS / Analemma)

**Context:** PDF paper proposing ASED, a training-free post-processing method for converting deterministic AI weather forecasts (GraphCast) into calibrated probabilistic ensembles via within-degree spectral anisotropy modeling. Evaluated on WeatherBench2 Z500 at 5-day lead time.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created. Contains:
  - 18 table entries from Table 1 (CRPS Global, CRPS Extra-tropics, SSR Global, SSR Extra-tropics for all 5 methods)
  - 3 figure entries (Figures 1, 2, 3)
  - 8 results_section claims (anisotropy index, zonal/meridional power ratios by scale band, gridpoint improvement fraction, per-latitude CRPS gains)

**Next session should:** Review or score the extracted results against ground truth annotations if available; or extend extraction to additional variables/lead times if the paper is expanded.

---

## Session: 2026-04-24 (Analysis)

**Purpose:** static analysis — classify all 29 extracted claims into verification buckets

**Files/context inspected:**
- `paper.pdf` — read paper context via prior extraction session
- `exp/EXPERIMENT_RESULTS/deterministic_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/isotropic_gp_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/sed_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/ased_main/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/ifs_ens_reference/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/anisotropy_diagnostic/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/crps_spatial_analysis/RESULTS.json`
- `exp/ased/scripts/` — all run scripts
- `exp/ased/perturbations/` — ased.py, sed.py, isotropic_gp.py
- `exp/ased/analysis/anisotropy_diagnostic.py`

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with classifications for all 29 claims

**Summary of classifications:**
- All 29 claims classified as `static_verifiable`
- Every table metric (claims 1–18) directly matches corresponding RESULTS.json file value:
  - Deterministic (160.93, 239.96) → deterministic_baseline/RESULTS.json
  - Isotropic GP (143.47±0.74, 182.94±0.90, SSR 1.127/1.009) → isotropic_gp_baseline/RESULTS.json
  - SED (143.80±0.52, 182.90±0.51, SSR 1.129/1.009) → sed_baseline/RESULTS.json
  - ASED (139.60±0.52, 181.54±0.46, SSR 1.123/1.010) → optimize_trace/iteration_0/RESULTS.json
  - IFS-ENS (117.34, 174.75, SSR 0.989/1.010) → ifs_ens_reference/RESULTS.json
- Figure claims (19–21) confirmed by anisotropy ratio data in RESULTS.json files and crps_spatial_analysis/RESULTS.json
- Results-section claims (22–29):
  - A_cal = -0.276±0.005 → anisotropy_diagnostic/RESULTS.json (A_cal=-0.2762, se=0.0053)
  - 4.26× power ratio → optimize_trace/iteration_0/RESULTS.json (w_ratio=4.257)
  - Synoptic ratio 5.99 → optimize_trace/iteration_0/RESULTS.json (5.9945...)
  - Planetary 4.12 → optimize_trace/iteration_0/RESULTS.json (4.1242...)
  - Small-scale 1.82 → optimize_trace/iteration_0/RESULTS.json (1.8221...)
  - 82.4% gridpoints, Δ3.17 → crps_spatial_analysis/RESULTS.json
  - Tropics 7.05 → crps_spatial_analysis/RESULTS.json
  - NH 0.94, SH 1.53 → crps_spatial_analysis/RESULTS.json

**Recommended next step:** No further sessions required for analysis. All claims are statically verifiable from pre-computed RESULTS.json artifacts.
