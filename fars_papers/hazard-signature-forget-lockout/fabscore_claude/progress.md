## Session: 2026-04-24

**Purpose:** extraction

**Paper inspected:** paper.pdf — "Hazard-Signature Tombstones: Commit-Time Forget Lockout for LLM Agent Memory" (FARS / Analemma)

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created; contains tables (30 entries from Table 1 and Table 2), figures (2 entries: Figure 1 system overview, Figure 2 backfill ablation bar chart), and results_section (3 entries: index pollution percentages, backfill ablation average results per query, slot waste statistic).

**Summary of extraction:**
- Table 1: Main results for 5 conditions (No Forget, ID-Delete, Retrieval-Time Exact, Retrieval-Time Fuzzy, HST Commit-Time) across PRP@3, Benign Recall@3, WriteBlockRate, BenignFalseBlock, HS-Stability.
- Table 2: Embedding ablation (BGE-M3 vs Qwen3-Embedding-4B) for 3 conditions on PRP@3 and Benign Recall@3.
- Figure 1: HST system architecture diagram.
- Figure 2: Backfill ablation bar chart showing avg results per query for B(δ=0)=0.17, B(δ=6)=1.00, C(HST)=3.00.
- Results section: 3 non-table numerical claims about index pollution ratios and backfill ablation query results.

**Next session should:** run analysis/scoring against this extraction (fs_analysis.json), or verify extracted values against any supplementary material if available.

---

## Session: 2026-04-24 (analysis)

**Purpose:** analysis — static inspection of repository artifacts to classify all 35 extracted claims.

**Files inspected:**
- `exp/EXPERIMENT_RESULTS/no_forget_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condition_a_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condition_b_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condition_c_hst/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/backfill_ablation/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/alt_embedding_ablation/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/condition_b_optimized_results.json`
- `exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/condition_c_optimized_results.json`
- `exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json`
- Code: `exp/hst/memory/store.py`, `policies.py`, `exp/hst/data/hazard_signature.py`

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created; contains classification of all 35 claims.

**Summary of classifications:**
- All 35 claims classified as `static_verifiable`.
- Table 1 claims (indices 1–18): each directly matched to a stored RESULTS.json in EXPERIMENT_RESULTS/. Numerical values match within standard rounding (e.g., 0.9167 → 0.92, 0.6667 → 0.67).
- Table 2 claims (indices 19–30): all matched to `alt_embedding_ablation/RESULTS.json`, which explicitly records per-model (BGE-M3, Qwen3) results for conditions A, B, C.
- Figure 1 (index 31): architecture diagram verifiable from code structure (store.py, policies.py, hazard_signature.py).
- Figure 2 (index 32): bar chart data verified from `backfill_ablation/RESULTS.json` (B_delta_0=0.1667, B_delta_6=1.0, C=3.0 avg results per query).
- Results section claims (indices 33–35): all verified from `condition_a_baseline/RESULTS.json` and `backfill_ablation/RESULTS.json`.

**Next session:** No further action required; all claims are statically verifiable from saved JSON artifacts. If independent re-execution is desired, entrypoints are `exp/hst/scripts/run_*.py`.
