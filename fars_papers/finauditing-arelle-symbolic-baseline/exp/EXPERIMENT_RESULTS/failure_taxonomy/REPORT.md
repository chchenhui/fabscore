# Failure Taxonomy Analysis of Non-Executable FinMR Instances

## Experiment Overview

Analyzed the distribution of failure causes across all 137 non-executable FinMR instances (out of 332 total) to characterize the benchmark's packaging quality and provide actionable recommendations for benchmark maintainers.

## Setup

- **Data source**: `outputs/arelle_baseline_results.jsonl` (332 instances from Arelle symbolic baseline)
- **Failure classifier**: `executable_finmr/engine/failure_classifier.py` -- assigns one of 6 priority-ordered labels based on Arelle error logs
- **Analysis script**: `executable_finmr/analysis/failure_taxonomy.py`

## Key Results

### Overall Failure Distribution (N=137 non-executable)

| Failure Category       | Count | Percentage | Top Affected DQC Rules             |
|------------------------|------:|-----------:|------------------------------------|
| External Dependency    |    88 |     64.2%  | DQC_0117(63), DQC_0126(13), DQC_0015(12) |
| Malformed XML/Escaping |    18 |     13.1%  | DQC_0015(11), DQC_0117(7)          |
| Unknown                |    18 |     13.1%  | DQC_0015(17), DQC_0126(1)          |
| Missing DTS Artifact   |    13 |      9.5%  | DQC_0015(10), DQC_0126(3)          |

### Per-DQC Rule Failure Breakdown

| DQC Rule | N_fail | External Dep | Missing DTS | Malformed XML | Unknown |
|----------|-------:|-------------:|------------:|--------------:|--------:|
| DQC_0015 |     50 |    12 (24%)  |   10 (20%)  |    11 (22%)   | 17 (34%)|
| DQC_0117 |     70 |    63 (90%)  |    0 (0%)   |     7 (10%)   |  0 (0%) |
| DQC_0126 |     17 |    13 (76%)  |    3 (18%)  |     0 (0%)    |  1 (6%) |

### External Dependency Analysis

- **88 instances** fail due to unresolvable external taxonomy URLs in offline mode
- **978 distinct external URLs** referenced across these instances
- **3 distinct domains**: `xbrl.fasb.org` (most frequent), `www.xbrl.org`, `xbrl.sec.gov`
- **100% recoverable**: All 88 external-dependency failures reference only standard taxonomy URLs; bundling US-GAAP + SRT + DEI + XBRL base taxonomy packages would recover all of them
- This would raise executability from 58.73% (195/332) to **85.24% (283/332)**

### Instance Characteristics by Failure Category

| Category              |   N | Mean Query Length | Median Query Length | Mean Sections |
|-----------------------|----:|------------------:|--------------------:|--------------:|
| External Dependency   |  88 |          121,747  |           124,236   |          7.0  |
| Missing DTS Artifact  |  13 |          119,948  |           115,300   |          7.0  |
| Malformed XML         |  18 |          118,667  |           120,934   |          7.0  |
| Unknown               |  18 |          113,149  |           112,699   |          7.0  |
| Executable (baseline) | 195 |           99,991  |           110,322   |          7.0  |

Non-executable instances tend to have longer queries (more complex filings) but the same number of XBRL sections (all have 7 sections). External dependency instances have the longest queries on average.

## Key Observations

1. **External dependencies dominate**: 64.2% of all failures are due to unresolvable taxonomy URLs. DQC_0117 is most affected (90% of its failures).

2. **DQC_0117 is disproportionately impacted**: 70 of 120 DQC_0117 instances (58.3%) fail, almost entirely due to external dependencies. Its dimensional cross-check semantics require definition linkbase arcs that depend on taxonomy schemas.

3. **100% of external dependency failures are recoverable** by bundling standard US-GAAP/SEC taxonomy packages. This is the single highest-impact fix.

4. **DQC_0015 has the most diverse failure modes**: Unlike DQC_0117 and DQC_0126, DQC_0015 failures are spread across all categories, with "unknown" being the largest (34%).

5. **Query length correlates with failure**: Non-executable instances have ~20% longer queries on average, suggesting more complex filings with more taxonomy imports.

## Recommendations for Benchmark Maintainers

1. **Include US-GAAP taxonomy packages (2020-2024 vintages)** in the benchmark release. This alone would recover 88 of 137 non-executable instances (raising executability from 58.7% to 85.2%).
2. **Include SEC DEI, SRT, and Country taxonomy packages** alongside US-GAAP.
3. **Fix malformed XML** in 18 instances (double-escaped entities, unclosed tags).
4. **Resolve 13 instances with missing DTS artifacts** (broken schemaRef, missing linkbase files).
5. **Investigate 18 unknown-category failures** (concentrated in DQC_0015).

## Figures

- `results/figures/failure_category_bar.png` -- Horizontal bar chart of failure category distribution
- `results/figures/failure_sankey.png` -- Flow diagram: Instance -> Executable/Non-executable -> Failure Category -> DQC Rule
- `results/figures/failure_per_rule_stacked.png` -- Stacked bar chart of failure categories by DQC rule family

## Full Report

- `results/failure_taxonomy_report.json` -- Complete structured report with all statistics
