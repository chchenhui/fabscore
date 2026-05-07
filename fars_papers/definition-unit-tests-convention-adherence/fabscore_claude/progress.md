# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** paper.pdf — "Definition Unit Tests Improve LLM Convention Adherence" (FARS, Analemma)
**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` (created)

**Summary of extraction:**
- Extracted 56 table entries from Table 1 (main results on ErdosConventionsBench, 300 items, two models × five conditions × five metrics) and Table 2 (ablation on number of discriminative checks k, unique k=1 rows only to avoid duplication).
- Extracted 2 figures: Figure 1 (DUT prompt structure diagram) and Figure 2 (per-family accuracy bar chart).
- Extracted 7 results_section claims covering: neutral-checks baseline within 2pp of glossary-only; majority vote comparisons (+53.7pp/+44.3pp on Qwen, +22.7pp/+20.7pp on Llama); relative alt-rate reductions (81% on Qwen, 78% on Llama); error-analysis percentages (29% arithmetic vs 5% convention on Llama convolution; 18% parsing failures vs 0% convention on Llama completeness).

**Next session should:** run analysis/scoring on the extracted results (fs_analysis.json), or verify extraction completeness against additional paper sections if needed.

---

## Session 2 — 2026-04-24
**Purpose:** analysis (static verification of all 65 claims)

**Files inspected:**
- `exp/results/effectiveness_evaluation.json` — main consolidated results (Table 1: overall + per-family accuracy, alternate convention rates for both models across all 5 conditions)
- `exp/results/analysis/per_family_table.json` — per-family accuracy breakdown (also verifies per-family numbers)
- `exp/results/analysis/k_ablation.json` — Table 2 k=1 ablation results for both models
- `exp/results/analysis/error_breakdown.json` — error category breakdown (arithmetic, convention, parsing) for Condition C outputs
- `exp/dut_project/prompts/templates.py` — prompt structure implementation (Figure 1 verification)
- `exp/results/figures/per_family_accuracy.png` — Figure 2 output artifact

**JSON files created/updated in this session:**
- `fabscore_claude/fs_analysis.json` (created)

**Summary of classifications:**
All 65 claims classified as `static_verifiable`. Every numerical claim in Tables 1 and 2 was found verbatim in either `effectiveness_evaluation.json` (Table 1) or `k_ablation.json` (Table 2). Results-section claims (indices 59–65) were verified against stored diffs in `effectiveness_evaluation.json` and error rates in `error_breakdown.json`. Figure 1 (prompt structure) verified against `prompts/templates.py`. Figure 2 (per-family accuracy) verified against `per_family_table.json` and figure PNG.

**Next session should:** No further analysis needed. All claims are statically verifiable from existing JSON artifacts.
