# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** paper.pdf (Silence-Conditional Output Suppression for Training-Free Whisper Hallucination Mitigation)
**Files created:**
- `fabscore_claude/fs_extracted.json` — full extraction of tables, figures, and results_section claims

**Summary of extraction:**
- Tables: 30 entries from Table 1 (main results: hallucination rate, WER, FP rate across 5 conditions) and Table 2 (per-class trigger rates for 10 UrbanSound8K classes)
- Figures: 3 figures (method overview, tradeoff curve, per-class trigger rate bar chart)
- Results section: 3 non-duplicate numerical claims (Always-Mask WER deltas, FP counts at τ=0.3, FP counts at τ=0.6)

**Next session should:** perform analysis/scoring (fs_analysis.json) comparing extracted results against any reference or rubric.

## Session 2 — 2026-04-24
**Purpose:** analysis
**Files inspected:**
- `exp/EXPERIMENT_RESULTS/condition_a_default_whisper/RESULTS.json` — Condition A baseline metrics
- `exp/EXPERIMENT_RESULTS/condition_b_always_mask/RESULTS.json` — Condition B always-mask metrics
- `exp/EXPERIMENT_RESULTS/condition_c_schm/RESULTS.json` — Condition C SCHM multi-tau metrics
- `exp/EXPERIMENT_RESULTS/phase1_diagnostic/RESULTS.json` — Per-class trigger rates at tau=0.5
- `exp/schm/inference/run_schm.py`, `run_default.py`, `run_always_mask.py` — inference implementations
- `exp/schm/inference/compute_p_nospeech.py` — p_no_speech extraction
- `exp/schm/analysis/plot_tau_tradeoff.py`, `run_diagnostic.py` — analysis scripts
- `fabscore_claude/progress.md` — prior session state

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — full classification of all 36 claims

**Summary of classifications:**
All 36 claims classified as `static_verifiable`:
- Claims 1–20 (Table 1): All hallucination rates, WER values, and FP rates match exactly with RESULTS.json values in condition_a, condition_b, condition_c subdirectories.
- Claims 21–30 (Table 2): All per-class trigger rates at tau=0.5 match exactly with phase1_diagnostic/RESULTS.json.
- Claims 31–33 (Figures): Method description (Fig.1) confirmed by run_schm.py/compute_p_nospeech.py; tradeoff trend (Fig.2) and per-class ordering (Fig.3) confirmed by RESULTS.json data.
- Claims 34–36 (Results section): WER deltas, false positive counts all exactly derivable from condition_b and condition_c RESULTS.json.

**Key observations:**
- No conflicting data found between paper claims and repository artifacts.
- The repo contains complete RESULTS.json files with exact matching numbers for every quantitative claim.
- FP Rate=0% for Condition A and FP Rate=100% for Condition B are logically derivable from inference code (no suppression vs. always-suppress).

**Next session:** No further action required; all claims are resolved.
