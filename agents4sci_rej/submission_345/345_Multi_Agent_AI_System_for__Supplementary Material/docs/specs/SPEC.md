# SPEC — Reusable prompt templates (single source of truth)

All agents must select prompts from the sections below. When iterating, refine templates here rather than crafting custom one-offs. Include versioned headers when materially changing a template.

## PLANNING v1
- Goal: Produce minimal plan to pass gates G1–G5.
- Inputs: repo state summary, data profile, open TODOs.
- Output: ordered tasks with owners (Execution/Writing/Reviewer), risk notes, success criteria.

## ETL v1
- Goal: Normalize raw CSVs into parquet tables with strict schema validation.
- Inputs: paths to data_raw, schema definitions.
- Output: data_proc/*.parquet, results/data_profile.json, validation log.

## BASELINE v1
- Goal: Implement peak_sales_heuristic and year2_naive.
- Inputs: launches.parquet row, access/pricing assumptions.
- Output: scalar (peak) or 1x5 array (years) with units and assumptions.

## ANALOGS v1
- Goal: Retrieve and weight analogs; project adjusted curve.
- Inputs: drug row, launches_df, revenues_df.
- Output: np.ndarray (5 years), list of analog ids and weights.

## PATIENT_FLOW v1
- Goal: Build patient-flow forecast using access ceilings.
- Inputs: drug row, access tier, diagnosis/treatment/adherence priors.
- Output: np.ndarray (5–10 years), intermediate assumptions.

## EVAL v1
- Goal: Evaluate models on held-out test, bootstrap CIs, correct for multiple comparisons.
- Inputs: predictions, actuals, split indices, alpha.
- Output: metrics.json with medians, 95% CIs, Holm–Bonferroni adjusted p-values.

## POWER v1
- Goal: Detect 10% absolute MAPE improvement at alpha=0.05, power=0.8.
- Inputs: effect size, variance estimate, desired power.
- Output: required N and achieved power given N.

## COVERAGE v1
- Goal: Compute 80% PI coverage and width.
- Inputs: predictive distributions or intervals, actuals.
- Output: coverage %, width stats, calibration notes.

## WRITING.INTRO v1
- Goal: Update INTRODUCTION.md using latest gates/metrics; no over-claims.
- Inputs: results_summary.json, data profile, related work anchors.
- Output: concise, sourced intro; bold primary claims gated.

## WRITING.METHODS v1
- Goal: Update METHODS.md reflecting exact implementation and stats protocol.
- Inputs: code modules/versions, seeds, config snapshot.
- Output: reproducible methods section.

## WRITING.RESULTS v1
- Goal: Update RESULTS.md with current metrics and figures; add limitations.
- Inputs: eval artifacts and figures.
- Output: sober results text aligned to gates.

## REVIEW v1
- Goal: Enforce gates and statistical hygiene, flag risks.
- Inputs: artifacts, logs, diffs.
- Output: review report with pass/fail and fixes.
