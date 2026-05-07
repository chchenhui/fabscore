# Arelle Symbolic Baseline for FinMR

## Experiment Overview

This experiment implements and evaluates the proposed Arelle-based symbolic baseline for the FinMR (Financial Mathematical Reasoning) benchmark. The pipeline reconstructs per-instance XBRL packages from FinMR query text, loads them with the Arelle XBRL engine in offline mode, and computes extracted/calculated values using rule-family-specific XBRL semantics. It also produces an executability audit classifying which instances can be successfully processed.

## Setup

- **Dataset**: TheFinAI/FinMR (332 instances, test split)
- **DQC Rule Distribution**: DQC_US_0015 (110), DQC_US_0117 (120), DQC_US_0126 (102)
- **Engine**: Arelle v2.38.17, Python 3.12
- **Mode**: Fully deterministic, no LLM, no random seeds, CPU-only
- **Evaluation**: Deterministic judge (A=Accurate, S=Structural Error, E=Extraction Error, C=Calculation Error)

### Pipeline Steps
1. **XBRL Package Reconstruction**: Parse query sections, write 6 XML files per instance, rewrite hrefs, inject linkbaseRef entries, handle truncated XML and missing schemas
2. **Target Extraction**: Extract concept QName, period, DQC rule family from parsed instance fields
3. **Arelle Loading**: Load instance in offline mode with cached US-GAAP taxonomies (2021-2024)
4. **DQC Rule Execution**:
   - DQC_0015: extracted_value = target fact, calculated_value = abs(target fact)
   - DQC_0117: extracted_value = default-context fact, calculated_value = sum of dimensional member facts (axis-aware grouping)
   - DQC_0126: extracted_value = parent concept fact, calculated_value = weighted sum of calculation linkbase children
5. **Failure Classification**: Non-executable instances classified by failure cause

## Key Results

### Full-Set Metrics (N=332)

| Metric | Value | Count |
|--------|-------|-------|
| ACC    | 42.17% | 140/332 |
| SER    | 41.27% | 137/332 |
| EER    | 3.01%  | 10/332  |
| CER    | 13.55% | 45/332  |

### Executable-Subset Metrics (N=195)

| Metric | Value | Count |
|--------|-------|-------|
| ACC    | 71.79% | 140/195 |
| SER    | 0.00%  | 0/195   |
| EER    | 5.13%  | 10/195  |
| CER    | 23.08% | 45/195  |

### Executability Coverage

- **Coverage**: 58.73% (195/332)
- Executable instances span all three DQC rule families

### Per-DQC Rule Breakdown (Full-Set)

| Rule | N | Executable | ACC | SER | EER | CER |
|------|---|-----------|-----|-----|-----|-----|
| DQC_US_0015 | 110 | 60 (54.5%) | 45.45% | 45.45% | 9.09% | 0.00% |
| DQC_US_0117 | 120 | 50 (41.7%) | 37.50% | 58.33% | 0.00% | 4.17% |
| DQC_US_0126 | 102 | 85 (83.3%) | 44.12% | 16.67% | 0.00% | 39.22% |

### Per-DQC Rule Breakdown (Executable Subset)

| Rule | N_exec | ACC | EER | CER |
|------|--------|-----|-----|-----|
| DQC_US_0015 | 60 | 83.33% | 16.67% | 0.00% |
| DQC_US_0117 | 50 | 90.00% | 0.00% | 10.00% |
| DQC_US_0126 | 85 | 52.94% | 0.00% | 47.06% |

### Failure Taxonomy (N=137 non-executable)

| Failure Cause | Count | Description |
|---------------|-------|-------------|
| external_dependency | 88 | Remote taxonomy/schema URLs not resolvable offline |
| missing_dts_artifact | 13 | Schema or linkbase missing from reconstructed package |
| malformed_xml | 18 | XML parsing failures due to truncation or encoding |
| unknown | 18 | Other failures |

## Key Observations

1. **High executability for DQC_0126** (83.3%): Calculation linkbase-based rules have the most self-contained XBRL packages in FinMR, with complete schemas and linkbases.

2. **Low executability for DQC_0117** (41.7%): Dimensional aggregation rules often require external taxonomy components (definition linkbase entries referencing remote schemas) that are not included in the FinMR query.

3. **Data truncation is a major barrier**: 185/332 instances have instance documents truncated at exactly 32,767 characters (2^15 - 1), and 120/332 have missing or invalid schemas. This is a dataset-level limitation.

4. **DQC_0015 has best executable-subset ACC** (83.33%): The sign-check rule is the simplest -- once the target fact is located, the calculation (absolute value) is trivial. Remaining errors are due to dimensional ambiguity when multiple facts match the target concept and period.

5. **DQC_0117 has highest executable-subset ACC after optimization** (90.00%): Axis-aware dimensional grouping resolved most calculation errors. Only 5 CER instances remain, caused by incomplete dimensional data.

6. **DQC_0126 CER is highest** (47.06% on executable subset): Calculation errors arise when child facts in the summation-item relationships are missing from the instance or have different periods/contexts than the parent.

7. **SER is 0% on executable subset**: By construction, all executable instances produce valid JSON output, so no structural errors occur within the executable subset.

## Optimization History

- **Original**: ACC=38.86% (129/332)
- **After optimization (iter 0)**: ACC=42.17% (140/332), +3.31pp
  - Fix: Axis-aware dimensional fact grouping in DQC_0117 (+12 instances)
  - Fix: Namespace-aware linkbase injection in XBRL reconstructor (minor)

## Output Files

- `outputs/arelle_baseline_results.jsonl` - Per-instance results (332 rows)
- `results/arelle_baseline_metrics.json` - Aggregate metrics
- `results/executable_finmr_ids.json` - List of 195 executable instance IDs
