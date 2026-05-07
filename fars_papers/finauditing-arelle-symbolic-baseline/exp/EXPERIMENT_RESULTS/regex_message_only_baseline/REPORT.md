# Regex Message-Only Heuristic Baseline on FinMR

## Experiment Overview

This experiment implements and evaluates a trivial heuristic baseline (Level 0 in the baseline ladder) for the FinMR benchmark from FINAUDITING. The baseline extracts `extracted_value` and `calculated_value` purely from the textual content embedded in each FinMR query using regex and simple arithmetic -- no proper XML parser or XBRL tooling is used.

The purpose is to test whether the DQC message text (and embedded XBRL instance data) alone leaks enough information to solve the task, providing a lower bound for reasoning difficulty.

## Setup

- **Dataset**: TheFinAI/FinMR (332 instances, test split)
- **DQC Rule Distribution**: DQC_US_0015 (110), DQC_US_0117 (120), DQC_US_0126 (102)
- **Method**: Regex-based extraction from query text with rule-family-specific logic:
  - DQC_0015 (sign/negativity): Extract reported value from instance XML via regex, compute abs() for calculated_value
  - DQC_0117 (dimensional cross-check): Extract default fact value, sum dimensional member facts
  - DQC_0126 (calculation consistency): Extract parent fact value, parse calc linkbase arcs via regex to find children+weights, sum weighted child values
- **Evaluation**: Deterministic evaluator replicating FINAUDITING Appendix C.3 (Structure -> Extraction -> Calculation -> Accurate)
- **Deterministic**: No random seeds needed; fully reproducible

## Key Results

### Aggregate Metrics

| Metric | Value |
|--------|-------|
| ACC    | 44.58% (148/332) |
| SER    | 39.46% (131/332) |
| EER    | 6.02% (20/332) |
| CER    | 9.94% (33/332) |

### Per-DQC Rule Breakdown

| DQC Rule | N | ACC | SER | EER | CER |
|----------|---|-----|-----|-----|-----|
| DQC_US_0015 | 110 | 36.36% | 45.45% | 18.18% | 0.00% |
| DQC_US_0117 | 120 | 29.17% | 54.17% | 0.00% | 16.67% |
| DQC_US_0126 | 102 | 71.57% | 15.69% | 0.00% | 12.75% |

### Comparison with Published Baselines

| Method | ACC | SER | Source |
|--------|-----|-----|--------|
| Fin-o1-14B (best LLM) | 13.86% | 71% | FINAUDITING paper |
| **Regex baseline (ours)** | **44.58%** | **39.46%** | This experiment |

## Key Observations

1. **The regex baseline (44.58% ACC) dramatically outperforms the best published LLM baseline (13.86%)**. This confirms that the FinMR query text leaks substantial information -- a simple regex can extract answers without any reasoning.

2. **DQC_0126 is most amenable to regex extraction (71.57% ACC)**: The calculation linkbase XML and instance document facts are usually present and well-structured enough for regex parsing.

3. **DQC_0117 has the highest SER (54.17%)**: Many DQC_0117 instances have no schema document (`None`) and truncated instance documents lacking fact elements, making regex extraction impossible.

4. **DQC_0015 has the highest EER (18.18%)**: The sign/negativity rule requires matching the correct fact to the target concept, which sometimes fails due to ambiguous context matching (multiple facts for the same concept in different dimensional contexts).

5. **SER dominates errors (39.46%)**: The main failure mode is inability to extract any values at all (empty prediction), not incorrect computation. This is driven by missing/truncated XBRL artifacts in the query text.

6. **Zero CER for DQC_0015**: Once the extracted_value is correctly found, computing abs() is trivially correct, confirming this rule type's simplicity.
