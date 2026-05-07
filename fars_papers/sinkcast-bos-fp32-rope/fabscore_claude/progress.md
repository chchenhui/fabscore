# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** paper.pdf (SinkCast: An Empirical Study of Inference-Time Correction for BF16 RoPE Shift-Invariance)
**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries

**Summary:**
- Extracted 109 table entries from Tables 1, 2, and 3 covering: BF16 RoPE shift-error distribution (Dlogit values and j0-fractions), SinkCast gap closure results across K values, and downstream evaluation on RULER 4K/8K and LongBench for Llama-3.1-8B and Mistral-7B-v0.3.
- Extracted 2 figure entries (Figure 1: SinkCast pipeline; Figure 2: shift-error distribution bar chart).
- results_section is empty: all numerical results in the Experiments section body text are already captured in the tables (5.0%/8.5% j0-fractions in Table 1; 23.6%/36.3% gap closure in Table 2; −0.91 points overall in Table 3). No additional numerical claims exist in the body text that are not table duplicates.

**Next session should:**
- Verify or score the extracted claims against ground truth if available.
- No further extraction needed unless a revised paper version is provided.

## Session 2 — 2026-04-24
**Purpose:** analysis
**Files inspected:**
- `exp/EXPERIMENT_RESULTS/bf16_flash_microbench/RESULTS.json` — BF16 microbench data (D_logit values, j0_fraction) for Table 1
- `exp/EXPERIMENT_RESULTS/sinkcast_k1_microbench/RESULTS.json` — SinkCast gap closure results for Table 2 (all K values)
- `exp/EXPERIMENT_RESULTS/sinkcast_k1_downstream/RESULTS.json` — Downstream evaluation for Table 3 (RULER + LongBench)
- `exp/EXPERIMENT_RESULTS/bf16_downstream_baseline/RESULTS.json` — BF16 baseline downstream results (cross-checked)
- `exp/sinkcast/core/sinkcast.py`, `rope_utils.py`, `sinkcast_hooks.py` — Core algorithm implementation (Figure 1)
- `fabscore_claude/progress.md` — Previous session log

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — Created with all 111 claims classified

**Summary of classifications:**
- All 111 claims classified as `static_verifiable`.
- Table 1 (claims 1–12): D_logit and j0_fraction values match exactly in bf16_flash_microbench/RESULTS.json at seq_len=2048, shift_pair=(0,4096).
- Table 2 (claims 13–28): Gap closure (max/mean drift) for all K values match exactly in sinkcast_k1_microbench/RESULTS.json under `all_K_results`.
- Table 3 (claims 29–109): All per-benchmark BF16/SC drops and improvements match exactly in sinkcast_k1_downstream/RESULTS.json; averages (-0.22, -1.60, -0.91) verified arithmetically.
- Figure 1 (claim 110): 3-stage pipeline description consistent with code in sinkcast.py/rope_utils.py/sinkcast_hooks.py.
- Figure 2 (claim 111): j0-fraction values (5.0%, 8.5%) directly in bf16_flash_microbench/RESULTS.json.

**Next session should:**
- No further action required. All claims are fully supported by static artifacts in the repository.
