## Session: 2026-04-24

**Purpose:** extraction

**Paper inspected:** paper.pdf — "Tiny-LR Proxy SFT for Dataset Ranking: An Empirical Investigation" (FARS / Analemma). The paper compares Standard-LR Proxy (5e-5), Tiny-LR Proxy (1e-5), and Training-Free NLL baselines for SFT dataset ranking using Qwen2.5-1.5B proxy and Qwen2.5-7B target across 12 math datasets.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created; contains tables (44 entries from Tables 1–3), figures (3 entries: Figures 1–3), and results_section (1 entry: Standard-LR Proxy's MATH-500-specific PDA=0.864, ρ=0.874 not captured in any table).

**Notes:**
- Table 1: overall ranking agreement (PDA, Spearman ρ, p-value, Top-1) for 4 methods.
- Table 2: per-benchmark analysis of Tiny-LR Proxy (MATH-500 Only vs GSM8K Only).
- Table 3: full dataset rankings by all methods vs target ground truth.
- The only numerical result in the body text NOT already in tables: Standard-LR Proxy MATH-500-only performance (PDA=0.864, ρ=0.874), mentioned in Section 4.2.

**Next session:** analysis or scoring against ground truth results.

---

## Session: 2026-04-24 (analysis)

**Purpose:** static analysis — classify all 48 extracted claims against repository artifacts

**Files/Context Inspected:**
- `exp/tlr_proxy_sft/results/main_results_table.csv` — master results table with PDA, Spearman ρ, p-values for all methods
- `exp/tlr_proxy_sft/results/ranking_agreement_std.json` — Standard-LR proxy ranking, PDA=0.7121, ρ=0.5944, p=0.04152
- `exp/tlr_proxy_sft/results/ranking_agreement_tiny_v2.json` — Tiny-LR (1e-5) proxy ranking, PDA=0.5, ρ=-0.0909, p=0.7787
- `exp/tlr_proxy_sft/results/ranking_agreement_nll.json` — NLL baseline ranking, PDA=0.6364, ρ=0.3706, p=0.23562
- `exp/tlr_proxy_sft/results/per_benchmark_pda.json` — per-benchmark breakdown (MATH-500, GSM8K) for Standard-LR and Tiny-LR
- Repository tree (via subagent Explore)

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created; all 48 claims classified

**Summary of Classifications:**
- **static_verifiable: 48** (all claims)
- **no_code_files: 0**, **obvious_hallucination: 0**, **insufficient_evidence: 0**, **execution_required: 0**, **error: 0**

**Key findings:**
- All Table 1 numerical values (PDA, CI, Spearman ρ, p-value) for Random, NLL, Standard-LR, and Tiny-LR conditions match the JSON result files exactly (within rounding to 3 decimal places).
- All Table 2 per-benchmark values (MATH-500: PDA=0.818, ρ=0.846, p≈0.0005; GSM8K: PDA=0.515, ρ=0.063, p=0.846) match `per_benchmark_pda.json`.
- All Table 3 dataset ranks (12 datasets × 2 methods = 24 rank claims) match the `proxy_ranking` arrays in `ranking_agreement_std.json` and `ranking_agreement_tiny_v2.json`.
- Figure claims (2, 3) are backed by the same underlying JSON data files.
- Results-section claim (PDA=0.864, ρ=0.874 for Standard-LR on MATH-500) matches `per_benchmark_pda.json` standard_lr math500 (0.8636, 0.8741).

**Next session:** execution/scoring or no further action needed (all claims static_verifiable).
