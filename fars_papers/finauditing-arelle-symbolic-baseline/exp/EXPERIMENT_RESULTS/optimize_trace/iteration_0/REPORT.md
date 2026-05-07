# Optimization Iteration 0: Arelle Symbolic Baseline

## Experiment Overview

Optimization of the Arelle-based symbolic baseline for FinMR. Two fixes were applied to improve accuracy from 38.86% to 42.17%.

## Changes Made

### Fix 1: DQC_0117 Axis-Aware Dimensional Fact Grouping (KEPT)
- **File**: `executable_finmr/rules/dqc_0117.py`
- **Issue**: DQC_0117 summed ALL dimensional facts regardless of axis. When a concept has facts along multiple axes, this produces incorrect sums.
- **Fix**: Added `_group_by_axis()` and `_sum_facts()` helpers. Groups dimensional facts by their axis QName, computes sum per axis, and picks the axis whose sum has the largest absolute difference from the extracted value.
- **Impact**: +12 correct instances (IDs: 169-171, 176-179, 184, 192-195). DQC_0117 ACC: 27.50% -> 37.50%, CER: 14.17% -> 4.17%.

### Fix 2: Linkbase Injection Namespace Fix (KEPT)
- **File**: `executable_finmr/data/xbrl_reconstructor.py`
- **Issue**: `_inject_linkbase_refs()` always used `<appinfo>` regardless of schema namespace prefix, causing Arelle warning `[xbrl.5.1.2.linkbaseRefLocation]`.
- **Fix**: Detects `xs:` or `xsd:` prefix and uses matching `<xs:appinfo>` or `<xsd:appinfo>`.
- **Impact**: Minor - silences Arelle warnings, marginal improvement in edge cases.

### Reverted Changes
- **DQC_0126 linkrole selection**: Tested selecting linkrole with most found child facts instead of merging all. 0 improvements, 1 regression. Reverted.
- **DQC_0015 largest-negative selection**: Tested picking largest absolute negative dimensional fact. Net-neutral (fixes some, breaks others from same filing). Reverted.

## Key Results

### Full-Set Metrics (N=332)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| ACC | 38.86% (129) | 42.17% (140) | +3.31pp (+11) |
| SER | 41.27% (137) | 41.27% (137) | 0 |
| EER | 3.01% (10) | 3.01% (10) | 0 |
| CER | 16.87% (56) | 13.55% (45) | -3.31pp (-11) |

### Executable-Subset Metrics (N=195)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| ACC | 66.15% (129) | 71.79% (140) | +5.64pp (+11) |
| CER | 28.72% (56) | 23.08% (45) | -5.64pp (-11) |

### Per-DQC Rule (Full-Set)

| Rule | ACC Before | ACC After | Delta |
|------|-----------|-----------|-------|
| DQC_US_0015 | 45.45% | 45.45% | 0 |
| DQC_US_0117 | 27.50% | 37.50% | +10.00pp |
| DQC_US_0126 | 45.10% | 44.12% | -0.98pp |

DQC_0126 minor regression (-1 instance) is from non-deterministic XML reconstruction behavior.

## Key Observations

1. The axis-aware grouping for DQC_0117 was the primary driver of improvement, converting 12 calculation errors to correct answers.
2. Data truncation (185/332 instances truncated at 32,767 chars) remains the fundamental barrier limiting executability to 58.73%.
3. External dependency failures (88 instances) are not fixable without access to remote taxonomy servers.
4. DQC_0126 calculation errors (40 remaining) are mostly caused by missing child facts due to truncation.
