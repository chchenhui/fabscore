# Progress Log

## Session 1
- **Purpose**: extraction
- **Paper inspected**: paper.pdf — "Escrowed Batch Reveal: Eliminating First-Proposal Bias in Agentic Marketplaces Through Visibility Protocol Design" (FARS, Analemma)
- **Files created/updated**:
  - `fabscore_claude/fs_extracted.json` — created with tables (36 entries from Tables 1–3), figures (3 entries: Figures 1–3), and results_section (3 entries from Sections 4.2 and 4.4)
- **Summary of extraction**:
  - Table 1: Main results for 4 protocols (HardGate, SoftWait, ITS, EBR) on gemini-2.5-flash: first-arrival rate, completion rate, p-value
  - Table 2: Per-scenario breakdown across contractors_first/second/third for all 4 protocols
  - Table 3: Cross-model generalization comparing HardGate vs EBR on gemini-2.5-flash and claude-sonnet-4-5
  - Figures 1–3: Protocol comparison diagram, rank-selection bar chart, reveal-position bar chart
  - results_section: chi-squared p=0.494 uniformity result, EBR rank distribution (24.4%/35.6%/37.8%), reveal-position primacy (72.7%, p=1.1e-7)
- **Next session**: No further extraction needed. Could verify results against any supplementary materials if added.

## Session 2
- **Purpose**: analysis (static)
- **Files inspected**:
  - `exp/EXPERIMENT_RESULTS/ebr_gemini_flash/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/hardgate_gemini_flash/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/softwait_gemini_flash/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/its_gemini_flash/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/cross_model_robustness/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/reveal_order_analysis/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json`
  - `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_report.md`
  - `exp/EXPERIMENT_RESULTS/ebr_gemini_flash/REPORT.md`
  - `exp/EXPERIMENT_RESULTS/softwait_gemini_flash/REPORT.md`
  - `exp/multi-agent-marketplace/experiments/ebr/analysis/statistical_test_results.json`
  - `exp/multi-agent-marketplace/experiments/ebr/analysis/statistical_tests.py`
  - `exp/multi-agent-marketplace/experiments/ebr/analysis/main_results_table.py`
- **JSON files created**:
  - `fabscore_claude/fs_analysis.json` — created with full classification of all 42 claims
- **Summary of classifications**:
  - **static_verifiable** (40/42): All table values (Table 1, 2, 3), all figure data claims (Figures 1-3), and all results-section claims are directly confirmed by RESULTS.json artifacts in EXPERIMENT_RESULTS/.
  - **execution_required** (2/42): Claims 8 and 12 (SoftWait and ITS p vs HardGate = 0.38). The repository's statistical_test_results.json records one-sided z-test p=0.178762; two-sided z-test gives ~0.357. Neither matches the paper's 0.38 exactly. The discrepancy could come from a Fisher's exact two-sided test (not computed for secondary comparisons in statistical_tests.py), so execution is needed to verify.
- **Key findings**:
  - The repository has comprehensive RESULTS.json files for all 4 conditions (HardGate, SoftWait, ITS, EBR) and cross-model robustness and reveal-order analysis, with values matching every major claim.
  - The only unclear point is the exact statistical test used for the Table 1 p-values for secondary comparisons (claims 8, 12). The stored one-sided z-test (0.178762) and two-sided z-test (0.357) both differ slightly from the paper's 0.38.
- **Recommended next step**: Run `python exp/multi-agent-marketplace/experiments/ebr/analysis/statistical_tests.py` and additionally compute Fisher's exact two-sided p-value for SoftWait/ITS vs HardGate comparisons to verify whether p=0.38 is reproducible.

## Session 3
- **Purpose**: execution (claim 8: Table 1, SoftWait p vs. HardGate = 0.38)
- **Files inspected**:
  - `exp/EXPERIMENT_RESULTS/softwait_gemini_flash/RESULTS.json` — SoftWait: 19/30 earliest arrival
  - `exp/EXPERIMENT_RESULTS/hardgate_gemini_flash/RESULTS.json` — HardGate: 33/45 earliest arrival
  - `exp/multi-agent-marketplace/experiments/ebr/analysis/statistical_tests.py` — only one-sided z-test for secondary comparisons
  - `exp/multi-agent-marketplace/experiments/ebr/analysis/statistical_test_results.json` — one-sided p=0.178762 stored
- **Execution artifacts created**:
  - `fabscore_claude/workspace/claim_8_command_output.txt` — all statistical test results
- **Execution findings**:
  - SoftWait: 19/30, HardGate: 33/45
  - Repository z-test one-sided: p=0.178762 (matches stored JSON)
  - Z-test two-sided: p=0.357524 (≈0.36)
  - Fisher exact two-sided: p=0.445328 (≈0.45)
  - Boschloo exact two-sided: p=0.376193 (rounds to 0.38)
  - Paper claims p=0.38; ONLY Boschloo's exact (not implemented in repo) gives 0.376 ≈ 0.38
  - Repository code uses z-test → p=0.357 (two-sided), which conflicts with paper's p=0.38
- **Verdict**: Result Fabrication — the repository's implemented test (z-test) gives p=0.357 two-sided (stored as 0.179 one-sided), neither matching the paper's claimed p=0.38. No test implemented in the repository produces 0.38. While Boschloo's exact (two-sided) gives 0.376 ≈ 0.38, this test is not implemented or referenced in the repository.
- **Next session**: Can verify claim 12 (ITS p vs HardGate = 0.38) using same data — same comparison since ITS has identical counts (19/30) to SoftWait.

## Session 4
- **Purpose**: execution (claim 12: Table 1, Inference-Time Scaling, p vs. HardGate = 0.38)
- **Files inspected**:
  - `exp/EXPERIMENT_RESULTS/its_gemini_flash/RESULTS.json` — ITS: 19/30 earliest arrival (confirmed identical to SoftWait)
  - `fabscore_claude/workspace/claim_8_command_output.txt` — reused artifact from Session 3
- **Execution artifacts reused**:
  - `fabscore_claude/workspace/claim_8_command_output.txt` — directly applicable since ITS (19/30) = SoftWait (19/30), yielding identical p-values vs HardGate (33/45)
- **Key findings**:
  - ITS: 19/30, HardGate: 33/45 (same data as SoftWait vs HardGate)
  - Repository z-test one-sided: p=0.178762 (stored in statistical_test_results.json)
  - Z-test two-sided: p=0.357524 (≈0.36, not 0.38)
  - Fisher exact two-sided: p=0.445328 (not 0.38)
  - Boschloo exact two-sided: p=0.376193 (rounds to 0.38, but NOT implemented in repository)
  - Paper claims p=0.38; no test implemented in repository produces this value
- **Verdict**: Result Fabrication — same reasoning as claim 8. The repository's statistical_tests.py implements only a one-sided z-test (p=0.179) for secondary comparisons. The two-sided z-test gives p≈0.357, not 0.38. The only test yielding 0.38 is Boschloo's exact (p=0.376), which is not implemented or referenced in the repository.
- **Next session**: No further action needed for claims 8 or 12.
