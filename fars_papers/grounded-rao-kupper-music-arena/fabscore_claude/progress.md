## Session: 2026-04-24

**Purpose:** extraction

**Paper inspected:** paper.pdf — "Grounded Rao-Kupper Leaderboards for Music Arena" (FARS / Analemma). 8-page PDF proposing GRK, a ranking model for arena-style evaluation that treats BOTH BAD votes as outside options anchored to model quality. Evaluated on Music Arena (3,274 battles, 12 text-to-music systems).

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (41 entries from Tables 1–3), figures (2 entries: Figure 1 overview diagram, Figure 2 acceptability correlation scatter), and results_section (5 entries: bootstrap CIs for NLL difference, instrumental/vocal NLL reduction percentages, 23% BOTH BAD reduction, ablation CI, Pearson r correlation).

**Next session should:**
- Run analysis/scoring against fs_extracted.json (e.g., fs_analysis.json, fs_summary.json).
- Verify any auto-generated claims against the raw table values if needed.

---

## Session: 2026-04-24 (analysis)

**Purpose:** analysis (static verification of all 48 extracted claims)

**Files/context inspected:**
- `exp/grk_music_arena/results/grk_global.json`
- `exp/grk_music_arena/results/bt_baseline_global.json`
- `exp/grk_music_arena/results/abmnl_baseline_global.json`
- `exp/grk_music_arena/results/grk_instrumental.json`, `grk_vocal.json`
- `exp/grk_music_arena/results/bt_baseline_instrumental.json`, `bt_baseline_vocal.json`
- `exp/grk_music_arena/results/abmnl_baseline_instrumental.json`, `abmnl_baseline_vocal.json`
- `exp/grk_music_arena/results/grk_no_ground_global.json`
- `exp/grk_music_arena/results/main_results_table.json`
- `exp/grk_music_arena/results/abmnl_reg_sensitivity.json`
- `exp/grk_music_arena/results/empirical_bothbad_rates.json`
- `exp/grk_music_arena/results/rho_beta_correlation.json`
- `exp/grk_music_arena/results/leaderboard_comparison.json`
- `exp/grk_music_arena/scripts/run_acceptability_analysis.py`
- Repository structure (via Explore agent)

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with all 48 claims classified as `static_verifiable`

**Summary of classifications:**
All 48 claims are `static_verifiable`:
- Claims 1–18 (Table 1): NLL, Brier, ECE with CIs verified against per-split result JSON files. All match within rounding precision.
- Claims 19–30 (Table 2): Per-class NLL values verified from global result JSON files.
- Claims 31–41 (Table 3): Ablation results verified from grk_no_ground_global.json (GRK vs no-grounding) and abmnl_reg_sensitivity.json (AB-MNL L2=0 and L2→∞ variants).
- Claim 42 (Figure 1): Conceptual diagram; model equations verified from grk_model.py code.
- Claim 43 (Figure 2): Pearson r=0.60 (p=0.041) manually computed from empirical_bothbad_rates.json × grk_global.json acceptability (12 systems), confirming the paper's scatter plot claim.
- Claims 44–48 (results section): All bootstrap CIs, NLL reduction percentages, and Pearson r confirmed from pre-computed JSON files.

**Recommended next step:** Analysis complete. All 48 claims are statically verifiable. No execution is required.
