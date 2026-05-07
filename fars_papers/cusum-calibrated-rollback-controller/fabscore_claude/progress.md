## Session: 2026-04-24 — extraction

**Purpose:** Extract experimental results from paper.pdf

**Paper inspected:** `paper.pdf` — "CUSUM-ϵ: False-Alarm-Calibrated Rollback Thresholds for Runtime Training Stability Controllers" (Analemma/FARS). Compares CUSUM-ϵ vs. Or-ϵ one-step threshold rollback controllers on ResNet-18/CIFAR-10 with synthetic perturbations.

**JSON files created:**
- `fabscore_claude/fs_extracted.json` — tables (40 entries from Tables 1–3), figures (1 entry: Figure 1), results_section (5 claims from Sections 4.2 and 4.3).

**Key findings extracted:**
- Table 1: calibration parameters for both controllers (Or-ϵ ε=1.181, actual rate 0.16%; CUSUM-ϵ h=18.0, actual rate 0.20%)
- Table 2: main results — Or-ϵ step peak 1.85±0.92 vs CUSUM-ϵ 3.47±0.81; CUSUM false rollbacks 37.0±38.0 vs Or-ϵ 87.5±44.6
- Table 3: FIR partial reset reduces original CUSUM peak from 5.40 to 3.47 (step), −36%/−35%/−34%/−22% improvements
- Results section: Or-ϵ achieves 1.88× lower peak excess loss, 1.47× lower AUC (step); 1.77×/1.85× for ramp; CUSUM has 58% fewer false rollbacks; both controllers give >80% reduction vs no controller

**Next session should:** run analysis/scoring on `fs_extracted.json` or perform any additional review tasks as directed.

---

## Session: 2026-04-24 — analysis

**Purpose:** Static analysis of all 46 extracted claims against repository artifacts.

**Files inspected:**
- `paper.pdf` (via prior extraction)
- `exp/cusum_controller/results/calibration/calibration_params.json`
- `exp/cusum_controller/results/no_controller/summary_metrics.json`
- `exp/cusum_controller/results/or_epsilon/summary_metrics.json`
- `exp/cusum_controller/results/cusum_epsilon/metrics_summary.json`
- `exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/REPORT.md`
- `exp/EXPERIMENT_RESULTS/optimize_trace/optimize_summary_0.json`
- `exp/EXPERIMENT_RESULTS/cusum_epsilon/RESULTS.json`

**JSON files created:**
- `fabscore_claude/fs_analysis.json` — classification of all 46 claims

**Summary of classifications:**
- **All 46 claims: `static_verifiable`**
  - Claims 1–12 (Table 1 calibration params): all verified from `calibration_params.json` + `summary_metrics.json` files
  - Claims 13–35 (Table 2 main results): all verified from `no_controller/summary_metrics.json`, `or_epsilon/summary_metrics.json`, `cusum_epsilon/metrics_summary.json`
  - Claims 36–40 (Table 3 FIR ablation): verified from `REPORT.md` comparison table and `RESULTS.json` improvement_vs_original fields; back-derivation independently confirms original CUSUM baseline numbers
  - Claim 41 (Figure 1 architecture): verified from code structure in `or_controller.py`, `cusum_controller.py`, `calibrate.py`
  - Claims 42–46 (results section ratios): all derivable arithmetically from the above JSON files

**Key numerical confirmations:**
- ε = 1.1805 ≈ 1.181, h = 18.0, µ₀ = -0.041, σ₀ = 0.226, α = 0.1 all match
- All Table 2 means/stds match to rounding precision (2 decimal places)
- CUSUM delay -18.25 → rounds to -18.3; ramp delay -22.05 → rounds to -22.1
- FIR improvement pct: -35.7/-35.0/-33.7/-21.9 round to -36/-35/-34/-22 as reported
- Original CUSUM (h=14) baseline numbers cross-verified via back-derivation from pct changes

**Recommended next step:** No further analysis needed. All claims are static_verifiable with strong provenance chains. If execution is requested, full reproduction scripts are available at `exp/cusum_controller/scripts/`.
