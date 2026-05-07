# Self-Consistency LLM Baseline on FinMR (50-Instance Subset)

## Experiment Overview

Evaluates a frontier long-context LLM (gpt-4.1) with best-of-4 self-consistency on a 50-instance stratified subset of FinMR. This provides a Level 4 baseline in the baseline ladder, testing whether inference-time scaling (multiple samples + majority vote) improves LLM performance on XBRL numerical reasoning.

## Setup

- **Model**: gpt-4.1 (via LEMMA MAAS API, OpenAI-compatible)
- **Method**: Self-consistency with 4 independent samples per instance, majority vote among parseable JSON outputs
- **Temperature**: 0.7, max_tokens: 512
- **Subset**: 50 instances, stratified by DQC rule family (17 DQC_0015, 18 DQC_0117, 15 DQC_0126)
- **Seeds**: [42, 123, 456] (3 runs for variance estimation)
- **Evaluator**: Deterministic judge (FINAUDITING Appendix C.3)
- **Concurrency**: asyncio with 10-concurrent-request semaphore, 5 retries with exponential backoff

## Key Results

### Aggregate (mean +/- std across 3 seeds)

| Metric | Value |
|--------|-------|
| ACC    | 8.67% +/- 0.94% |
| SER    | 0.00% +/- 0.00% |
| EER    | 44.00% +/- 0.00% |
| CER    | 47.33% +/- 0.94% |

### Per-Seed Results

| Seed | ACC | SER | EER | CER |
|------|-----|-----|-----|-----|
| 42   | 8.00% | 0.00% | 44.00% | 48.00% |
| 123  | 10.00% | 0.00% | 44.00% | 46.00% |
| 456  | 8.00% | 0.00% | 44.00% | 48.00% |

### Per-DQC Breakdown (averaged across seeds)

| DQC Rule | N | ACC Range | EER | CER Range |
|----------|---|-----------|-----|-----------|
| DQC_0015 | 17 | 0-5.88% | 47.06% | 47.06-52.94% |
| DQC_0117 | 18 | 11.11% | 55.56% | 33.33% |
| DQC_0126 | 15 | 6.67-20.00% | 26.67% | 53.33-66.67% |

## Key Observations

1. **gpt-4.1 with self-consistency achieves only 8.67% ACC**, which is lower than the published best LLM (Fin-o1-14B at 13.86% ACC on the full 332 instances) and far below the regex baseline (44.58% on the full set).

2. **Zero structural errors (SER=0%)**: The model consistently produces valid JSON output, showing that the prompt format is well-understood.

3. **Extraction is the main bottleneck (EER=44%)**: Nearly half the instances have extraction errors, meaning the model fails to identify the correct reported value from the XBRL filing. This is consistent across all seeds.

4. **Calculation errors are also prevalent (CER=47.33%)**: Even when the model extracts the right value, it frequently computes the wrong calculated value.

5. **DQC_0117 (dimensional) is hardest for extraction** (55.56% EER) but easiest for calculation when extraction succeeds. **DQC_0126 (calculation linkbase)** has the lowest EER (26.67%) but highest CER (up to 66.67%).

6. **Low variance across seeds** (ACC std = 0.94%): Self-consistency with 4 samples provides stable results. The EER is identical across all seeds, suggesting extraction failures are deterministic for this model.

7. **Self-consistency does not help much**: The 4-sample majority vote provides marginal improvement over what a single sample would give, as the model tends to make the same errors consistently.

## Comparison with Other Baselines

| Method | Dataset | ACC |
|--------|---------|-----|
| Published Fin-o1-14B (zero-shot) | Full 332 | 13.86% |
| **gpt-4.1 SC (best-of-4)** | **50-instance subset** | **8.67%** |
| Regex message-only heuristic | Full 332 | 44.58% |

## Files

- `RESULTS.json`: Full metrics including per-seed and per-DQC breakdowns
- Per-instance predictions: `executable_finmr/outputs/sc_llm_baseline_results.jsonl`
- Subset IDs: `executable_finmr/configs/sc_subset_ids.json`
- Implementation: `executable_finmr/baselines/self_consistency_baseline.py`
- Runner: `executable_finmr/scripts/run_sc_baseline.py`
