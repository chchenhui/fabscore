## Session: 2026-04-24 — extraction

**Purpose:** Extract experimental results from paper.pdf (ScaffoldSwap ablation study).

**Paper inspected:** `paper.pdf` — "ScaffoldSwap: Are Discrete Speech Units Necessary as a Temporal Scaffold for Audio-Driven 3D Facial Animation?" (FARS / Analemma)

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (16 entries from Table 1 and Table 2), figures (1 entry: Figure 1), and results_section (3 entries with statistical significance values not captured in tables).

**Summary of extracted results:**
- Table 1: Main LVE results for SSL (WavLM), Phoneme+Timing (MFA), and Discrete Units (HuBERT+k-means) on BIWI and VOCASET.
- Table 2: Ablation results on BIWI — discretization effect (+17.5%), timing features (+0.01%), temporal shuffle (+14.9%).
- Figure 1: ScaffoldSwap framework diagram.
- Results section: Per-dataset gap percentages (1.3% BIWI, 0.5% VOCASET) and statistical significance factors (7.1× BIWI, 5.2× VOCASET) not captured in tables.

**Next session should:** Run analysis/scoring on fs_extracted.json, or compare extracted results against claims in the abstract/conclusion for consistency checking.

---

## Session: 2026-04-24 — analysis

**Purpose:** Static analysis to classify all 20 extracted claims from the ScaffoldSwap ablation study.

**Files inspected:**
- `paper.pdf` (via prior extraction session context)
- `exp/EXPERIMENT_RESULTS/condA_biwi/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condA_vocaset/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condB_biwi/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condB_vocaset/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condC_biwi/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condC_vocaset/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condC_no_timing_biwi/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/hubert_continuous_biwi/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/sanity_check_temporal_shuffle/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json`
- Repository code: frontends (ssl_frontend.py, unit_frontend.py, phoneme_frontend.py), model (scaffold_model.py, decoder.py, frequency_adaptor.py)

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with all 20 claims classified as `static_verifiable`

**Summary of classifications:**
- All 20 claims classified as `static_verifiable`.
- Every numerical claim (Table 1, Table 2, results section) is directly backed by precise values in RESULTS.json files in `exp/EXPERIMENT_RESULTS/`. The repository stores per-seed and averaged LVE metrics for all conditions (A/B/C on BIWI and VOCASET), plus ablation runs (C w/o timing, HuBERT-continuous, temporal shuffle). All delta percentages were independently verified by calculation from stored means and are consistent.
- Figure 1 (framework overview) is confirmed by the code structure implementing all three audio conditioning approaches with a shared TCN decoder.
- Results section claims about gap magnitude and statistical significance (7.1×, 5.2×) are confirmed from the RESULTS.json files and effectiveness_evaluation_result.json.

**Next session should:** No further action needed — all claims are fully verified through static analysis.
