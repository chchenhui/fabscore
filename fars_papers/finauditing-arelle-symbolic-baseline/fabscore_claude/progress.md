## Session: 2026-04-24 — extraction

**Purpose:** Extract experimental results from paper.pdf (Executable FinMR: Arelle-Based Symbolic Baselines and an Executability Audit for XBRL Mathematical Reasoning, FARS / Analemma).

**Paper inspected:** `paper.pdf` (8-page PDF, FinMR benchmark evaluation of an Arelle-based symbolic XBRL baseline vs. LLM baselines).

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (42 entries from Tables 1–3), figures (2 entries: Figure 1 pipeline overview, Figure 2 failure taxonomy bar chart), and results_section (4 entries: 29-instance intersection accuracy, projected executability after taxonomy bundling, failure cause counts).

**Key results captured:**
- Arelle symbolic baseline: 42.17% ACC (full set), 71.79% ACC (executable subset, 0% SER)
- Best LLM (Fin-o1-14B): 13.86% ACC; SC LLM (gpt-4.1): 8.67%±0.94%
- Per-DQC: DQC 0015 → 83.33%, DQC 0117 → 90.00%, DQC 0126 → 52.94% (Arelle)
- Executability: 58.73% (195/332); 64.2% of failures from external taxonomy deps; projected 85.24% with bundling
- href rewriting ablation: 0% executability without it

**Next session should:** Perform analysis or scoring (fs_analysis.json / fs_summary.json) against ground-truth results if available, or cross-check extraction completeness.

---

## Session: 2026-04-24 — analysis

**Purpose:** Static analysis of all 48 extracted claims against repository artifacts.

**Files inspected:**
- `paper.pdf` (via extraction context)
- `exp/EXPERIMENT_RESULTS/arelle_symbolic_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/regex_message_only_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/per_dqc_rule_breakdown/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/sc_llm_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/sc_llm_baseline/REPORT.md`
- `exp/EXPERIMENT_RESULTS/ablation_no_rewrite/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/failure_taxonomy/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json`
- `exp/executable_finmr/` (module structure explored via agent)

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with full 48-claim classification

**Classification summary:**
- `static_verifiable`: 42 claims (indices 1–15, 21–42, 43, 44, 46, 47, 48)
  - All Table 1–3 metrics (Arelle, Regex, SC LLM baselines) confirmed directly in RESULTS.json files
  - Per-DQC breakdown, ablation, and failure taxonomy claims all match RESULTS.json exactly
  - Figure 1 pipeline matches code structure in exp/executable_finmr/
  - Figure 2 failure distribution data confirmed in failure_taxonomy/RESULTS.json
- `no_code_files`: 5 claims (indices 16–20)
  - Fin-o1-14B (published) and Other LLMs (published) — cited from external FinMR benchmark paper, no code/artifacts in this repo
- `insufficient_evidence`: 1 claim (index 45)
  - 29-instance SC-exec intersection (89.66% vs 14.94%) — mentioned in effectiveness_evaluation_result.json but no dedicated RESULTS.json; per-instance prediction files referenced but unconfirmed
- `obvious_hallucination`: 0
- `execution_required`: 0
- `error`: 0

**Key findings:**
- The repository contains pre-computed RESULTS.json files for every baseline and ablation that directly support all paper claims except the published LLM baselines and the intersection analysis.
- All arithmetic checks out: per-DQC counts sum to overall metrics; failure taxonomy counts (88+13+18+18=137=332-195).
- The DQC 0117 Regex exec ACC of 66.00% (exact) is directly in RESULTS.json.
- SC LLM CER mean=47.33% is confirmed by three-seed average (48%+46%+48%)/3.

**Next session should:** Generate fs_summary.json with final verdict/scoring if required. Or examine whether the sc_llm_baseline_results.jsonl file exists to potentially upgrade claim 45 to static_verifiable.
