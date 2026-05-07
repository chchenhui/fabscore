# Session Log

## Session 1
- **Purpose**: extraction
- **Paper inspected**: paper.pdf (Orthostochastic Residual Mixing for Manifold-Constrained Hyper-Connections, FARS/Analemma)
- **Files created**: `fs_extracted.json`
- **Summary**: Extracted 21 table entries (Table 1: main results with val loss and gradient spike ratios across two settings; Table 2: constraint fidelity DS error and orthogonality residual), 1 figure (Figure 1 diagram), and 3 results-section text claims covering gradient reduction (~39-40%), gradient norm values, and identity deviation range.
- **Next session**: Proceed to analysis (fs_analysis.json) — score the paper on faithfulness, novelty, clarity, and experimental rigor based on the extracted results.

## Session 2
- **Purpose**: analysis (static code/artifact inspection)
- **Files inspected**:
  - exp/results/effectiveness_evaluation.json
  - exp/results/constraint_fidelity_summary.json
  - exp/EXPERIMENT_RESULTS/setting_a_sinkhorn_baseline/RESULTS.json
  - exp/EXPERIMENT_RESULTS/setting_a_orthostochastic/RESULTS.json
  - exp/EXPERIMENT_RESULTS/setting_b_sinkhorn_baseline/RESULTS.json
  - exp/EXPERIMENT_RESULTS/setting_b_orthostochastic/RESULTS.json
  - exp/EXPERIMENT_RESULTS/hc_calibration/RESULTS.json
  - exp/EXPERIMENT_RESULTS/constraint_fidelity_analysis/RESULTS.json
  - exp/EXPERIMENT_RESULTS/training_dynamics_analysis/RESULTS.json
  - exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json
  - exp/orthostochastic_mhc_experiments/results/setting_a_sinkhorn_summary.json
  - exp/orthostochastic_mhc_experiments/results/setting_b_sinkhorn_summary.json
  - exp/orthostochastic_mhc_experiments/results/hc_calibration_summary.json
  - exp/orthostochastic_mhc_experiments/results/setting_a_orthostochastic_summary.json
  - exp/orthostochastic_mhc_experiments/results/setting_b_orthostochastic_summary.json
  - exp/scripts/plot_training_dynamics.py
  - exp/orthostochastic_mhc_experiments/analysis/diagnostics.py
- **Key findings**:
  - The logs directory (exp/orthostochastic_mhc_experiments/logs/) and .train_service_logs directory are EMPTY — no raw training logs, gradient_spikes.json, ds_error.json, or checkpoint files exist.
  - Pre-optimization results (Sinkhorn A, Sinkhorn B, HC Unconstrained) are stored with full float64 precision in summary JSON files, strongly indicating genuine training outputs.
  - Post-optimization Orthostochastic results (Claims 7-9) exist only in RESULTS.json with 4-6 decimal rounded values; no full-precision backup. Pre-optimization ortho summary (mean=4.7964) differs from post-optimization claimed value (4.7642).
  - Constraint fidelity and training dynamics RESULTS.json contain full-precision values (DS error, orth residual) consistent with actual torch computations, cross-referenced across multiple files.
- **Files created**: fabscore_claude/fs_analysis.json
- **Classification summary**:
  - static_verifiable (18): Claims 1-6 (HC Unconstrained, Sinkhorn A/B), 10-21 (Table 2 all DS error / orth residual metrics)
  - execution_required (6): Claims 7-9 (optimized Ortho A/B val loss and r_max — no raw logs), 23-25 (gradient norms and H_res range — no training logs or checkpoints)
  - insufficient_evidence (1): Claim 22 (Figure 1 — conceptual schematic with no underlying data or regeneration script)
- **Next session**: Execution — run optimized orthostochastic training (setting_a_orthostochastic.py, setting_b_orthostochastic.py) and regenerate gradient_spikes.json / checkpoints to verify claims 7-9, 23-25. Dataset download via exp/scripts/download_fineweb10B.sh may be required first.

## Session 3 (Claim 7 execution)
- **Purpose**: execution — verify Claim 7 (Table 1, mHC-Orthostochastic Setting A val loss 4.7642±0.013, ∆+0.003)
- **Files inspected**:
  - exp/EXPERIMENT_RESULTS/setting_a_orthostochastic/RESULTS.json — shows mean=4.7642, std=0.0125, per-seed=[4.760407, 4.754586, 4.781883, 4.775499, 4.748842], delta=0.0028
  - exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json — confirms same values
  - exp/orthostochastic_mhc_experiments/scripts/run_orthostochastic_a_optimized.sh — references external cluster `/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/`
  - exp/orthostochastic_mhc_experiments/scripts/sanity_check_optimized.sh — also references external cluster
  - exp/orthostochastic_mhc_experiments/scripts/aggregate_results.py — reads from logs/ directory (uses ddof=1 for std)
  - exp/orthostochastic_mhc_experiments/configs/setting_a_orthostochastic.py — config verified: ns_steps=15, identity_mix=True, alpha=0.1
- **Key findings**:
  - RESULTS.json values match claim exactly: 4.7642 ≈ mean, 0.0125≈±0.013, delta 0.0028≈+0.003 — internal math checks out
  - The nanogpt training code (`exp/mHC-manifold-constrained-hyper-connections/examples/nanogpt/`) does NOT exist locally
  - The logs/ directory does NOT exist (not just empty — absent entirely)
  - Training scripts reference external HPC cluster not accessible in this environment
  - aggregate_results.py uses sample std (ddof=1)=0.014, but RESULTS.json reports population std=0.0125 — slight inconsistency (paper rounds to 0.013 which is ambiguous between both)
  - No independent verification possible: no training logs, no checkpoints, no local training framework
- **Verdict**: Insufficient Evidence — RESULTS.json shows correct values but no training evidence (logs, checkpoints) and training cannot be re-run (missing local code + external cluster required)
- **No command output file created** (no repository commands run successfully)
- **Next session**: Verify claims 8-9 (Setting B orthostochastic val loss and r_max); same infrastructure limitation applies.

## Session 4 (Claim 8 execution)
- **Purpose**: execution — verify Claim 8 (Table 1, mHC-Orthostochastic Setting A r_max: 1.87±0.13)
- **Files inspected**:
  - exp/EXPERIMENT_RESULTS/setting_a_orthostochastic/RESULTS.json — r_max: mean=1.8665, std=0.1305, values=[1.7036, 2.0207, 1.7174, 1.9558, 1.9348]
  - exp/orthostochastic_mhc_experiments/results/setting_a_orthostochastic_summary.json — pre-optimization r_max: mean=1.9003 (different run/config)
  - exp/orthostochastic_mhc_experiments/scripts/aggregate_results.py — reads gradient_spikes.json from logs/, uses ddof=1
  - exp/orthostochastic_mhc_experiments/analysis/diagnostics.py — GradientSpikeTracker computes r_t = g_t / median(g_{t-100:t-1}), r_max = max_t r_t
  - exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json — no r_max data in trace
  - exp/orthostochastic_mhc_experiments/logs/ — empty (no gradient_spikes.json files)
- **Key findings**:
  - RESULTS.json r_max values match paper claim exactly: mean=1.8665→1.87, std=0.1305→0.13 (population std, ddof=0)
  - Math is internally consistent: manual computation from values [1.7036, 2.0207, 1.7174, 1.9558, 1.9348] yields mean=1.86646 and population std=0.1305
  - NOTE: aggregate_results.py uses ddof=1 (sample std), which would give 0.1459; RESULTS.json uses population std — inconsistency, but paper rounds to 0.13 matching population std
  - No gradient_spikes.json training logs exist anywhere in the repository
  - Training code (nanogpt) is absent locally; training scripts reference external HPC cluster
  - Pre-optimization summary has different r_max values (different config: no identity_mix, different ns_steps)
- **Verdict**: Insufficient Evidence — RESULTS.json shows consistent values but no training evidence (gradient_spikes.json logs, checkpoints) and training cannot be re-run
- **No command output file created** (no repository commands run)
- **Next session**: Verify claims 9 (Setting B orthostochastic r_max) and claims 23-25 (gradient norms and H_res range).

## Session 5 (Claim 9 execution)
- **Purpose**: execution — verify Claim 9 (Table 1, mHC-Orthostochastic Setting B (6L, n=8) Val Loss: 4.2626±0.005, ∆+0.013)
- **Files inspected**:
  - exp/EXPERIMENT_RESULTS/setting_b_orthostochastic/RESULTS.json — mean=4.2626, std=0.0050, values=[4.264551, 4.256906, 4.266355], seeds=[11,2,3], delta=0.0131
  - exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json — confirms same Setting B values
  - exp/orthostochastic_mhc_experiments/results/setting_b_orthostochastic_summary.json — pre-optimization run with seeds [1,2,3], full float64 precision; seed 1 best_val=4.3121 (later noted as diverged/replaced by seed 11)
  - exp/orthostochastic_mhc_experiments/logs/ — absent (no training logs)
- **Key findings**:
  - RESULTS.json matches claim exactly: mean=4.2626 ✓, std=0.0050 ✓ (ddof=1/sample std from values), delta=0.0131≈+0.013 ✓
  - Delta computed as orthostochastic(4.2626) - sinkhorn_baseline(4.2495) = 0.0131 ✓
  - No training logs, checkpoints, or local nanogpt code; scripts reference external HPC cluster
  - Same infrastructure limitation as claims 7-8
- **Verdict**: Insufficient Evidence — RESULTS.json values match claim but no training evidence (logs, checkpoints) and training cannot be re-run
- **No command output file created** (no repository commands run)
- **Next session**: Verify claims 23-25 (gradient norms and H_res range).

## Session 6 (Claim 23 execution)
- **Purpose**: execution — verify Claim 23 ("both mHC variants show substantially improved gradient stability—a reduction of approximately 39–40% compared to unconstrained HC")
- **Files inspected**:
  - exp/EXPERIMENT_RESULTS/hc_calibration/RESULTS.json — HC r_max=3.1387
  - exp/EXPERIMENT_RESULTS/setting_a_orthostochastic/RESULTS.json — Ortho r_max mean=1.8665, std=0.1305
  - exp/orthostochastic_mhc_experiments/results/setting_a_sinkhorn_summary.json — Sinkhorn r_max mean=1.9124735, full float64 precision
  - exp/results/effectiveness_evaluation.json — confirms Setting A and B r_max values for both variants
- **Key findings**:
  - Sinkhorn r_max = 1.9125 (full-precision) → reduction = (3.1387 - 1.9125)/3.1387 = 39.1% ✓ (within ~39-40%)
  - Ortho r_max = 1.8665 (rounded, from RESULTS.json) → reduction = (3.1387 - 1.8665)/3.1387 = 40.5% (~40%, marginally within range)
  - Pre-optimization Ortho r_max (hc_calibration comparison section) = 1.9003 → reduction = 39.5% ✓
  - Sinkhorn part is fully supported by full-precision training artifacts (no training logs needed)
  - Ortho part has same evidence limitation as claims 7-9: no gradient_spikes.json training logs, no checkpoints, no local nanogpt code; RESULTS.json only has rounded values
- **Verdict**: Insufficient Evidence — Sinkhorn reduction is verified (39.1%), but Ortho r_max relies solely on RESULTS.json without training logs; consistent with Insufficient Evidence rulings for claims 7-9
- **No command output file created** (no repository commands run)
- **Next session**: Verify claims 24-25 (gradient norm values and H_res range).

## Session 7 (Claim 24 execution)
- **Purpose**: execution — verify Claim 24 (Setting A median gradient norm last 1000 iterations: Sinkhorn 0.876±0.007, Ortho 0.864±0.011, HC 30% higher at 1.15)
- **Files inspected**:
  - exp/EXPERIMENT_RESULTS/training_dynamics_analysis/RESULTS.json — contains gradient_norm_statistics for setting_a
- **Key findings**:
  - Sinkhorn: median_last1k_mean=0.8762, median_last1k_std=0.0074 → claim says 0.876±0.007 ✓
  - Orthostochastic: median_last1k_mean=0.8635, median_last1k_std=0.0111 → claim says 0.864±0.011 ✓
  - HC Unconstrained: median_last1k=1.1545 → claim says 1.15 ✓
  - 30% higher claim: (1.1545-0.8762)/0.8762 ≈ 31.8% (consistent with ~30% claim)
  - key_findings in RESULTS.json itself states "HC Unconstrained shows 30% higher gradient norms than mHC methods"
  - Raw training logs (gradient_spikes.json) do NOT exist; nanogpt code is absent locally; training scripts reference external HPC cluster
  - Same infrastructure limitation as claims 7-9, 23
- **Verdict**: Insufficient Evidence — RESULTS.json values match claim exactly but no underlying training logs exist to verify how these statistics were computed; training cannot be re-run locally
- **No command output file created** (no repository commands run)
- **Next session**: Verify claim 25 (H_res range).

## Session 8 (Claim 25 execution)
- **Purpose**: execution — verify Claim 25 (Orthostochastic H_res deviation from identity: ∥H^res − I∥ ∈ [0.001, 0.166], largest deviation at final layer)
- **Files inspected**:
  - exp/EXPERIMENT_RESULTS/training_dynamics_analysis/RESULTS.json — h_res_analysis.setting_a.orthostochastic_seed1
  - exp/orthostochastic_mhc_experiments/logs/ — absent (no such directory)
  - exp/**/*.pt — no checkpoint files found anywhere
- **Key findings**:
  - RESULTS.json directly contains `h_minus_i_norm_range: [0.001, 0.166]` for orthostochastic_seed1 in setting_a ✓
  - `most_deviation_layer: 47`, note: "Layer 47 (last) has most deviation" — matches claim about final layer ✓
  - No checkpoint files (.pt) exist anywhere in the repository
  - Logs directory does not exist; training scripts reference external HPC cluster
  - Same infrastructure limitation as claims 7-9, 23-24 → cannot independently verify how h_res matrices were computed
- **Verdict**: Insufficient Evidence — RESULTS.json values match claim exactly ([0.001, 0.166] and final layer deviation) but no underlying checkpoints exist to verify the computation; training cannot be re-run locally
- **No command output file created** (only ls system command run, no repository Python/bash commands)
- **Next session**: All claims 7-9 and 23-25 have been assessed; all are Insufficient Evidence due to same infrastructure limitation. No further execution sessions required.
