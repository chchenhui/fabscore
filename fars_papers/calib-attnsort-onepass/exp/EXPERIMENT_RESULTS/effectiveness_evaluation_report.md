# Effectiveness Evaluation Report

## Verdict: bad

## Summary

The Debiased One-Pass Attention Sorting hypothesis is **refuted**. The proposed method -- subtracting a per-prompt position-bias curve from raw attention scores to enable a single sorting pass (k=1, 2 prefills) that matches iterative Attention Sorting (k=5, 6 prefills) -- fails on both tested models. On the primary model (LLaMA-2-7B-32K-Instruct), debiasing produces results **identical** to uncalibrated k=1 (94.83% vs 94.83%, 0 wins / 600 ties / 0 losses). On the secondary model (YaRN-Llama-2-7b-64k), debiasing shows a meaningful +8.67pp gain over uncalibrated k=1, but remains 14.84pp behind k=5, far exceeding the 2-3pp pivot threshold. The core claim that position-bias correction can substitute for iterative sorting is not supported.

## Experiment Feasibility Check

All experiments completed successfully. No infrastructure or environment issues.

- **No Sorting Baseline**: 3 seeds x 200 examples = 600 examples evaluated (LLaMA + YaRN)
- **Attention Sorting k=1**: 600 examples (LLaMA + YaRN)
- **Attention Sorting k=5**: 600 examples (LLaMA + YaRN)
- **Debiased k=1 (proposed)**: 600 examples per model, with 2 optimization iterations on LLaMA and 1 on YaRN
- All methods produced valid accuracy measurements. The experimental pipeline is sound.

## Results Analysis

### Consolidated Comparison Table

#### LLaMA-2-7B-32K-Instruct (Primary Model)

| Method | Seed 42 | Seed 123 | Seed 456 | Mean Accuracy | Std | Prefill Passes |
|--------|---------|----------|----------|---------------|-----|----------------|
| No Sorting | 70.5% | 76.0% | 72.0% | 72.83% | 2.84% | 1 |
| Attn Sort k=1 (uncalibrated) | 94.0% | 96.5% | 94.0% | 94.83% | 1.44% | 2 |
| **Debiased k=1 (proposed, best)** | **94.0%** | **96.5%** | **94.0%** | **94.83%** | **1.44%** | **2** |
| Attn Sort k=5 (iterative) | 96.5% | 95.0% | 95.0% | 95.50% | 0.87% | 6 |

#### YaRN-Llama-2-7b-64k (Secondary Model)

| Method | Seed 42 | Seed 123 | Seed 456 | Mean Accuracy | Std | Prefill Passes |
|--------|---------|----------|----------|---------------|-----|----------------|
| No Sorting | 36.0% | 33.5% | 38.0% | 35.83% | 2.25% | 1 |
| Attn Sort k=1 (uncalibrated) | 47.0% | 45.0% | 49.5% | 47.17% | 2.25% | 2 |
| **Debiased k=1 (proposed, best)** | **57.5%** | **53.5%** | **56.5%** | **55.83%** | **2.08%** | **2** |
| Attn Sort k=5 (iterative) | 71.5% | 73.5% | 67.0% | 70.67% | 3.33% | 6 |

### Regime Check (Precondition)

Both models pass the regime precondition (k=5 must improve over no-sorting by >= 3.0pp):

| Model | k=5 Accuracy | No Sorting | Gap | Threshold | Pass? |
|-------|-------------|------------|-----|-----------|-------|
| LLaMA-2-7B-32K-Instruct | 95.50% | 72.83% | 22.67pp | 3.0pp | **YES** |
| YaRN-Llama-2-7b-64k | 70.67% | 35.83% | 34.84pp | 3.0pp | **YES** |

The sorting regime is highly informative for both models.

### Decision Rule Application

#### Criterion (a): Debiased k=1 within 1-2pp of k=5

| Model | Debiased k=1 | k=5 | Gap | Within 2pp? |
|-------|-------------|-----|-----|-------------|
| LLaMA | 94.83% | 95.50% | 0.67pp | **YES** |
| YaRN | 55.83% | 70.67% | 14.84pp | **NO** |

LLaMA passes, but this is misleading -- the debiased method matches k=1 uncalibrated exactly, so the closeness to k=5 is inherited from k=1, not earned by debiasing.

#### Criterion (b): Debiased k=1 wins or ties vs k=5 on >= 80% of prompts

- **LLaMA**: Paired comparison vs k=5 not available at prompt level. However, debiased k=1 is identical to uncalibrated k=1 (0W/600T/0L), so this comparison reduces to "k=1 vs k=5" -- which is already known to have a 0.67pp deficit.
- **YaRN**: Debiased k=1 loses to k=5 on all 3 seeds (0 wins, 0 ties, 3 losses at seed level). The per-seed gaps are -14.0pp, -20.0pp, -10.5pp.

**FAIL on both models** (LLaMA inherits k=1's result, YaRN loses decisively).

#### Criterion (c): Debiased k=1 improves over uncalibrated k=1 by margin outside std range

| Model | Debiased k=1 | Uncalibrated k=1 | Improvement | k=1 Std | Outside Std? |
|-------|-------------|-------------------|-------------|---------|--------------|
| LLaMA | 94.83% | 94.83% | +0.00pp | 1.44% | **NO** (zero improvement) |
| YaRN | 55.83% | 47.17% | +8.67pp | 2.25% | **YES** (3.85x std) |

LLaMA: Debiasing has literally zero effect -- the optimized minimal-swap strategy produces identical outputs on all 600 examples.
YaRN: Debiasing shows a genuine, large improvement, well outside the noise band.

#### Refute Criteria Assessment

1. **"Debiased k=1 is statistically indistinguishable from uncalibrated k=1"**: **TRUE on LLaMA** -- 0 wins, 600 ties, 0 losses. The methods are not just statistically indistinguishable; they are byte-for-byte identical.

2. **"Debiased k=1 is >= 3pp worse than k=5 while k=5 clears precondition"**: **TRUE on YaRN** -- debiased k=1 is 14.84pp worse than k=5, and k=5 clears the precondition by 34.84pp.

Both refute criteria are met (one per model).

#### Pivot Assessment

The pivot criterion requires debiased k=1 to be only 2-3pp behind k=5. On YaRN the gap is 14.84pp -- far too large for a k=2 hybrid to plausibly close. On LLaMA the gap is 0.67pp but debiasing contributes nothing to get there.

### Optimization History

Two rounds of optimization were applied:

**LLaMA-2-7B-32K-Instruct (2 iterations):**
1. Original full-sort debiased: 94.33% (0.50pp below uncalibrated k=1)
2. Optimized minimal-swap: 94.83% (matches uncalibrated k=1 exactly)

The optimization succeeded in recovering accuracy lost by full re-sorting, but the ceiling effect means debiasing adds nothing. The swap rate was 1.67% (10/600 examples), and all swaps were correct (debiased top-1 was the gold doc), but swapping already had no effect on generation because the model is robust to gold doc being in top-5 positions rather than strictly last.

**YaRN-Llama-2-7b-64k (1 iteration):**
1. Initial minimal-swap: 47.83% (matched k=1 uncalibrated)
2. Optimized full-sort with divisive debiasing: 55.83% (+8.67pp over k=1)

Three fixes were applied: (a) switched from median to mean aggregation with finer bins, (b) changed from additive to divisive debiasing, (c) used full-sort instead of minimal-swap. This closed 37% of the k=1-to-k=5 gap -- meaningful but insufficient.

## Statistical Significance

### LLaMA-2-7B-32K-Instruct

No statistical test is needed: debiased k=1 and uncalibrated k=1 produce **identical outputs on all 600 examples** (0 wins, 600 ties, 0 losses). The debiasing intervention has exactly zero effect on this model's predictions.

The gap between debiased k=1 (94.83%) and k=5 (95.50%) is 0.67pp. With k=5 std of 0.87%, this gap is within 1 standard deviation -- not statistically significant in a 3-seed design but consistently present across seeds.

### YaRN-Llama-2-7b-64k

**Debiased k=1 vs uncalibrated k=1**: All 3 seeds improve (+10.5pp, +8.5pp, +7.0pp). The mean improvement of 8.67pp is 3.85x the uncalibrated k=1's std (2.25%). This is clearly significant.

**Debiased k=1 vs k=5**: All 3 seeds show k=5 winning (-14.0pp, -20.0pp, -10.5pp). The 14.84pp deficit is 4.5x the k=5 std (3.33%). This gap is highly significant.

## Verdict Justification

### Verdict: **bad**

The core hypothesis is that explicit per-prompt position-bias subtraction can enable a single sorting pass (k=1, 2 prefill passes) to match the accuracy of iterative Attention Sorting (k=5, 6 prefill passes). This hypothesis is **refuted** on both tested models:

**Evidence for refutation:**

1. **Primary model (LLaMA-2-7B-32K-Instruct)**: Debiasing has zero effect. The optimized method produces byte-identical results to uncalibrated k=1 on all 600 test examples. The reason: on this model, uncalibrated k=1 already places the gold document near the end (mean position ~165 out of ~166), so the raw attention scores are already highly accurate. Position bias is not the bottleneck -- the remaining 0.67pp gap to k=5 comes from the refinement of document ordering across iterations, not from position-bias corruption of the initial sort.

2. **Secondary model (YaRN-Llama-2-7b-64k)**: Debiasing helps substantially (+8.67pp over k=1), confirming that position bias is a real and correctable phenomenon. However, the debiased method remains 14.84pp behind k=5 -- far beyond the 2-3pp "pivot" threshold, let alone the 1-2pp "success" threshold. This means position bias accounts for roughly 37% of the gap between k=1 and k=5; the remaining 63% comes from other iterative sorting benefits.

**What the results reveal about iterative sorting:**

The experimental findings suggest that iterative sorting's improvements over single-pass sorting are NOT primarily due to position-bias correction. Instead, they likely arise from:
- **Attention context refinement**: Each sorting iteration changes which documents are adjacent, altering cross-document attention patterns and providing progressively more informative relevance signals.
- **Error accumulation reduction**: Multiple passes allow the sorting to converge on a stable ordering, averaging out attention noise from any single forward pass.
- **Non-linear attention dynamics**: The attention pattern after re-sorting is not a simple bias-corrected version of the original -- it reflects genuinely new information from the reorganized context.

**Why "bad" rather than "good" with partial success:**

Although debiasing produces a genuine +8.67pp improvement on YaRN, the proposed method's core claim is specifically about matching k=5 with k=1. The method achieves only 37% of the needed improvement, with a 14.84pp residual gap that is too large for any reasonable pivot (e.g., k=2 hybrid) to close. The fundamental assumption -- that position bias is the primary barrier to one-pass sorting matching iterative sorting -- is contradicted by the evidence. The approach would need a fundamentally different mechanism to close the remaining gap, not just parameter tuning.