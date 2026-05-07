## Session: 2026-04-24 — extraction

**Purpose:** Extract experimental results from paper.pdf (Public-Anchor Drift Adapters for Privacy-Limited Embedding Model Upgrades).

**Paper inspected:** `paper.pdf` (6 pages, PDF)

**JSON files created:**
- `fabscore_claude/fs_extracted.json` — contains tables (23 entries from Table 1), figures (2 entries: Figure 1, Figure 2), and results_section (7 entries from sections 4.2 and 4.4).

**Summary of extracted content:**
- Table 1: nDCG@10 results for 5 methods (Oracle, Misaligned, In-domain Adapter, PADA, Shuffled-Pair Null) across 4 BEIR datasets (SciFact, TREC-COVID, FiQA-2018, ArguAna) plus average ρ.
- Figure 1: PADA framework overview diagram.
- Figure 2: Sample efficiency curve (anchor corpus size vs. retrieval recovery).
- Results section claims: PADA recovery ratios 1.11–1.31, per-dataset nDCG improvements, oracle recovery percentages, sample efficiency thresholds.

**Next session should:** Run scoring/analysis (fs_analysis.json) or any further downstream evaluation steps.

---

## Session: 2026-04-24 — analysis

**Purpose:** Static analysis of all 32 extracted claims against repository artifacts.

**Files inspected:**
- `exp/EXPERIMENT_RESULTS/task2_in_domain/RESULTS.json` — Oracle, Misaligned, In-domain adapter nDCG@10 values with per-seed breakdown
- `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json` — PADA ρ values and mean nDCG@10 per dataset
- `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_report.md` — Full results table (all 5 methods × 4 datasets) with mean ± std, oracle recovery %, per-seed ranges
- `exp/EXPERIMENT_RESULTS/size_ablation/RESULTS.json` — N_p sensitivity data: nDCG@10 and ρ for {500,1000,2000,5000,10000}
- `exp/EXPERIMENT_RESULTS/size_ablation/REPORT.md` — Detailed ablation analysis
- `exp/pada/adapters/residual_mlp.py`, `exp/pada/trainers/adapter_trainer.py`, etc. — Implementation cross-check

**JSON files created:**
- `fabscore_claude/fs_analysis.json` — Full classification of all 32 claims

**Summary of classifications:**
- **static_verifiable: 32/32** — All claims verified against stored JSON/Markdown artifacts with exact numerical matches (within rounding).
  - Claims 1–4 (Oracle): match task2_in_domain/RESULTS.json oracle baselines
  - Claims 5–8 (Misaligned): match task2_in_domain/RESULTS.json misaligned baselines
  - Claims 9–13 (In-domain adapter): match task2_in_domain/RESULTS.json per-seed means/stds; ρ=1.00 by construction
  - Claims 14–18 (PADA): match effectiveness_evaluation_report.md table; ρ avg 1.22 computed from individual ρ values
  - Claims 19–23 (Shuffled-Pair): match effectiveness_evaluation_report.md (ArguAna 0.0003 rounds to 0.000)
  - Claim 24 (Figure 1): code architecture matches described training/deployment/insight framework
  - Claim 25 (Figure 2): size_ablation/RESULTS.json provides underlying data; transitions match
  - Claims 26–32 (Results section): all computed metrics and thresholds match stored result files
- **No obvious_hallucination, no_code_files, insufficient_evidence, or execution_required**

**Recommended next step:** Proceed to execution phase (fs_execution.json) if required by harness, or finalize summary (fs_summary.json). No discrepancies found; static verification is complete.
