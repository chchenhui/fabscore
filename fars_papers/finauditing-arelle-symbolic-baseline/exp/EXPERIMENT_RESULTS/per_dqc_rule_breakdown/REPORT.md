# Per-DQC Rule Family Breakdown Analysis

## Experiment Overview

Fine-grained analysis of executability coverage and accuracy broken down by the three DQC rule families in the FinMR benchmark (332 instances total). Compares Arelle symbolic baseline and Regex message-only baseline across full-set and executable-subset evaluations.

## Setup

- **Dataset**: TheFinAI/FinMR test split (332 instances)
- **Rule families**: DQC_0015 (sign/negativity, N=110), DQC_0117 (dimensional cross-check, N=120), DQC_0126 (calculation-consistency, N=102)
- **Baselines**: Arelle symbolic baseline, Regex message-only baseline
- **Evaluator**: Deterministic judge (A/S/E/C labels per FINAUDITING Appendix C.3)
- **Source data**: `outputs/arelle_baseline_results.jsonl`, `outputs/regex_baseline_results.jsonl`

## Key Results

### Full-Set Metrics (All Instances)

| Rule Family | N | Exec% | Arelle ACC | Arelle SER | Arelle EER | Arelle CER | Regex ACC | Regex SER | Regex EER | Regex CER |
|---|---|---|---|---|---|---|---|---|---|---|
| DQC_0015 | 110 | 54.5% | 45.5% | 45.5% | 9.1% | 0.0% | 36.4% | 45.5% | 18.2% | 0.0% |
| DQC_0117 | 120 | 41.7% | 37.5% | 58.3% | 0.0% | 4.2% | 29.2% | 54.2% | 0.0% | 16.7% |
| DQC_0126 | 102 | 83.3% | 44.1% | 16.7% | 0.0% | 39.2% | 71.6% | 15.7% | 0.0% | 12.7% |

### Executable Subset Metrics

| Rule Family | N_exec | Arelle ACC | Arelle EER | Arelle CER | Regex ACC | Regex EER | Regex CER |
|---|---|---|---|---|---|---|---|
| DQC_0015 | 60 | 83.3% | 16.7% | 0.0% | 66.7% | 33.3% | 0.0% |
| DQC_0117 | 50 | 90.0% | 0.0% | 10.0% | 66.0% | 0.0% | 34.0% |
| DQC_0126 | 85 | 52.9% | 0.0% | 47.1% | 84.7% | 0.0% | 15.3% |

### Failure Taxonomy (Non-Executable Instances)

| Rule Family | N_fail | External Dep | Missing DTS | Malformed XML | Unknown |
|---|---|---|---|---|---|
| DQC_0015 | 50 | 12 (24.0%) | 10 (20.0%) | 11 (22.0%) | 17 (34.0%) |
| DQC_0117 | 70 | 63 (90.0%) | 0 (0.0%) | 7 (10.0%) | 0 (0.0%) |
| DQC_0126 | 17 | 13 (76.5%) | 3 (17.6%) | 0 (0.0%) | 1 (5.9%) |

## Key Observations

### Executability Coverage

1. **DQC_0126 is easiest to execute** (83.3%, 85/102). Calculation-consistency rules have self-contained calculation linkbases that reconstruct well.
2. **DQC_0117 is hardest to execute** (41.7%, 50/120). Dimensional cross-check rules depend heavily on external taxonomy dimensions (90% of failures are `external_dependency`).
3. **DQC_0015 is intermediate** (54.5%, 60/110). Sign/negativity rules have diverse failure modes -- the most balanced spread across all 4 failure categories.

### Accuracy on Executable Subset (Where Arelle Shines)

1. **DQC_0117: Arelle 90.0% vs Regex 66.0%** (+24.0pp). Dimensional cross-checks benefit most from structural XBRL traversal; regex cannot reliably parse dimensional aggregation.
2. **DQC_0015: Arelle 83.3% vs Regex 66.7%** (+16.7pp). Sign/negativity rules benefit from Arelle's fact extraction; regex has double the extraction error rate (33.3% vs 16.7%).
3. **DQC_0126: Arelle 52.9% vs Regex 84.7%** (-31.8pp). Arelle underperforms on calculation-consistency due to systematic calc-linkbase traversal bugs (CER=47.1%). Regex's simple message parsing is more robust here.

### Where Benchmark Difficulty Concentrates

- **Executability gap**: DQC_0117 has the largest gap between full-set and exec-subset accuracy (37.5% -> 90.0%), indicating 58.3% of difficulty is from non-executability, not rule logic.
- **Benchmark packaging issues**: DQC_0117's 63/70 `external_dependency` failures suggest FinMR's dimensional cross-check instances often lack necessary taxonomy dimension definitions.
- **Implementation bugs**: DQC_0126's 47.1% CER on executable instances is a code-level issue (Arelle calc-linkbase traversal), not a benchmark issue.

### Summary

- **Most amenable to symbolic execution**: DQC_0117 (90% ACC when executable) and DQC_0015 (83.3% ACC when executable)
- **Least amenable**: DQC_0126 -- high executability but low accuracy due to calc-linkbase bugs
- **Hardest to make executable**: DQC_0117 -- only 41.7% coverage, dominated by external dependency failures
- **Most benchmark packaging issues**: DQC_0117 (external_dependency) and DQC_0015 (diverse failure mix)

## Figures

- `figures/dqc_rule_judge_labels.png` -- Stacked bar chart of A/S/E/C label distribution for Arelle vs Regex per rule family
- `figures/dqc_rule_executability.png` -- Pie charts of executable vs non-executable breakdown per rule family
