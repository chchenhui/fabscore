# Progress Log

## Session 1 — 2026-04-24

**Purpose:** extraction

**Paper inspected:** paper.pdf — "FCBoost: Static Frequency-Aware Channel Selection for 2-Bit KV Cache Quantization" (Analemma / FARS)

**Context:** 8-page paper proposing FCBoost, a static channel-selection method for 2-bit KV cache quantization using Contextual Agreement (CA) scores derived from RoPE frequency analysis, evaluated on Qwen3-8B with AIME24/25 and GSM8K benchmarks.

**JSON files created:**
- `fabscore_claude/fs_extracted.json` — full extraction with tables, figures, and results_section entries

**Extraction summary:**
- Tables: 21 entries from Table 1 (main AIME24/25 results: FCBoost, Kitty, KIVI-KV2*) and Table 2 (ablation: CA mask vs random masks vs no-boost)
- Figures: 1 entry — Figure 1 (FCBoost framework overview)
- Results section: 3 entries not duplicated in tables — Jaccard overlap analysis (mean=0.299, std=0.122), Spearman rank correlation (mean ρ=0.670, std=0.185), and GSM8K short-context sanity check results (FCBoost 89.84%, FP16 90.30%, Kitty/KIVI-KV2* 88.48%)

**Next session should:** perform analysis or scoring of the extracted results (e.g., running fs_analysis or fs_summary steps).

---

## Session 2 — 2026-04-24

**Purpose:** analysis (static code audit of all 25 extracted claims)

**Files inspected:**
- `exp/EXPERIMENT_RESULTS/fcboost_main/RESULTS.json` — FCBoost v2 AIME24/25/avg results
- `exp/EXPERIMENT_RESULTS/kivi_kv2star_baseline/RESULTS.json` — KIVI-KV2* baseline results
- `exp/EXPERIMENT_RESULTS/kitty_baseline/RESULTS.json` — Kitty baseline results
- `exp/EXPERIMENT_RESULTS/random_mask_ablation/RESULTS.json` — Table 2 ablation data
- `exp/EXPERIMENT_RESULTS/overlap_analysis/RESULTS.json` — Jaccard/Spearman metrics
- `exp/EXPERIMENT_RESULTS/gsm8k_sanity_check/RESULTS.json` — GSM8K accuracy results
- `exp/results/random_mask_ablation.json` — cross-validated ablation data
- `exp/results/overlap_analysis.json` — cross-validated overlap analysis
- `exp/fcboost/profiling/ca_profiler.py`, `mask_generator.py`, `fcboost_cache.py` — implementation code
- `exp/fcboost/analysis/overlap_analysis.py` — overlap computation code

**JSON files created:**
- `fabscore_claude/fs_analysis.json` — full classification of all 25 claims

**Classification summary:**
- All 25 claims classified as `static_verifiable`
- Table 1 claims (1–12): exact matches in kivi_kv2star_baseline/RESULTS.json, kitty_baseline/RESULTS.json, fcboost_main/RESULTS.json
- Table 2 claims (13–21): exact matches in random_mask_ablation/RESULTS.json (two copies: EXPERIMENT_RESULTS and results/)
- Figure 1 (claim 22): architectural diagram; all described components (offline CA profiling, static mask generation, mixed-precision boosting of K and V caches) implemented in corresponding Python modules
- Results section claims (23–25): exact matches in overlap_analysis/RESULTS.json and gsm8k_sanity_check/RESULTS.json
- Notable: Table 1 and Table 2 use different seed runs for KIVI-KV2* (AIME24: 67.78 vs 65.56; AIME25: 64.44 vs 66.67) but the same average (66.11); this is consistent within their respective result files and plausible under different seed runs

**Next session should:** run fs_summary or scoring step.
