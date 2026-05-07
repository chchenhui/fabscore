# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** results/paper.md — "Execution-Trace Alignment: Fine-tuning Code LLMs with Step-wise Causal Program Feedback"
**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` (created)

**Summary of extraction:**
- 1 table found (Table 1): SFT (gpt2) on 10-example HumanEval subset — pass@1: 0.0%, Pass Count: 0, Fail Count: 10.
- 1 figure found: `sft_baseline/pass_fail_ratio.png` (Pass/Fail Ratio for SFT baseline).
- No results_section entries: all numerical body-text claims (0% pass@1, 100% failure rate) were already captured in Table 1 and thus excluded to avoid duplication.

**Next session should:**
- Proceed to analysis/scoring phase using `fs_extracted.json`.

## Session 2 — 2026-04-24
**Purpose:** analysis
**Files inspected:**
- `results/paper.md` — full paper read; Table 1 and Figure 1 verified
- `results/sft_baseline/evaluation_results.json` — 10 test cases, all FAIL (SyntaxError), confirming 0 passes / 10 failures
- `results/sft_baseline/pass_fail_ratio.png` — figure artifact confirmed present
- `gemini/src/evaluate.py` — plotting script (generation path for the figure)
- `fabscore_claude/fs_extracted.json` — confirmed 4 claims to analyze
- `results/log.txt` — training log confirming SFT run completed

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` (created)

**Summary of classifications:**
All 4 claims classified as `static_verifiable`:
- Claims 1–3 (Table 1 values): Directly confirmed by `results/sft_baseline/evaluation_results.json`, which contains exactly 10 entries all with status "FAIL", yielding pass@1=0.0%, Pass Count=0, Fail Count=10.
- Claim 4 (Figure sft_baseline/pass_fail_ratio.png): PNG artifact exists; underlying data (0 pass, 10 fail) fully present in `evaluation_results.json`. No execution needed.

**Next session should:**
- No further action needed; analysis is complete. All claims are statically verifiable.
