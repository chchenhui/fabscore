# Failure Taxonomy Analysis

## Experiment Overview

Classified all model outputs from Conditions A (Unconstrained), B (Best-of-2), and C (LAVE CFG) into a fine-grained error taxonomy with four mutually exclusive categories:

1. **Success** -- BFCL AST evaluator passes
2. **Parse Failure** -- `ast.parse()` raises an exception (syntactically invalid output)
3. **Wrong Function** -- parses but function name(s) do not match ground truth
4. **Wrong Arguments** -- correct function name(s) but BFCL eval fails (incorrect arguments)

This analysis directly addresses the core research question: are dLLM tool-calling failures dominated by parse/format errors or by semantic errors?

## Setup

- **Model**: LLaDA-8B-Instruct
- **Dataset**: BFCL-v3 Non-Live, 350 examples (7 categories x 50)
- **Seeds**: 42, 123, 456 (results averaged)
- **Conditions**: A (unconstrained diffusion), B (best-of-2 parseability filter), C (LAVE syntax-only CFG)
- **Classifier**: `bfcl_cfg_diffusion/analysis/failure_taxonomy.py`
- **Plots**: `bfcl_cfg_diffusion/analysis/plot_failure_taxonomy.py`

## Key Results

### Overall Failure Taxonomy (mean across 3 seeds)

| Error Type | (A) Unconstrained | (B) Best-of-2 | (C) LAVE CFG |
|---|---|---|---|
| **Success** | 126.7 (36.2%) | 127.0 (36.3%) | 128.7 (36.8%) |
| **Parse Failure** | 23.7 (6.8%) | 18.0 (5.1%) | 9.3 (2.7%) |
| **Wrong Function** | 45.0 (12.9%) | 47.0 (13.4%) | 54.3 (15.5%) |
| **Wrong Arguments** | 154.7 (44.2%) | 158.0 (45.1%) | 157.7 (45.0%) |

### Parse Failure Rate by BFCL Category (%)

| Category | (A) Unconstrained | (B) Best-of-2 | (C) LAVE CFG |
|---|---|---|---|
| simple_python | 4.7 | 3.3 | 0.0 |
| simple_java | 13.3 | 11.3 | 8.7 |
| simple_javascript | 10.7 | 9.3 | 4.7 |
| multiple | 3.3 | 2.0 | 2.7 |
| parallel | 2.7 | 1.3 | 2.7 |
| parallel_multiple | 12.7 | 8.7 | 0.0 |
| irrelevance | 0.0 | 0.0 | 0.0 |

## Key Observations

1. **Parse failures are the minority failure mode**: Only 6.8% of unconstrained outputs fail to parse, while 44.2% have wrong arguments and 12.9% have wrong function names. The dominant bottleneck is semantic, not syntactic.

2. **LAVE CFG effectively eliminates parse failures**: Parse failure drops from 6.8% (A) to 2.7% (C), a ~60% relative reduction. The remaining 2.7% comes primarily from Java/JavaScript categories where Python's `ast.parse` is less reliable.

3. **Parse failure reduction does not translate to success gains**: Despite reducing parse failures by 14.4 examples on average (from 23.7 to 9.3), success only increases by 2.0 examples (126.7 to 128.7). Most formerly-unparseable outputs, when made parseable by CFG constraints, reveal wrong function names (+9.3 examples in Wrong Function category).

4. **Wrong Arguments is the dominant failure mode at ~45% across all conditions**: This is stable regardless of parse constraints, confirming that the semantic understanding of argument names, values, and types is the true bottleneck.

5. **Category-level parse failures**: Java (13.3%) and JavaScript (10.7%) categories have the highest parse failure rates under unconstrained decoding, likely due to language-specific syntax differences. Parallel_multiple (12.7%) also has high parse failure, reduced to 0% by LAVE CFG.

6. **Verdict**: Syntax-only CFG constraints successfully reduce formatting errors but the tool-calling gap is overwhelmingly semantic. The ~4.1pp parse failure reduction from A to C translates to only ~0.6pp success improvement, because most outputs that become parseable still contain semantic errors.

## Output Files

- `bfcl_cfg_diffusion/scores/failure_taxonomy_results.json` -- Full numerical results
- `bfcl_cfg_diffusion/scores/failure_taxonomy_stacked.pdf` -- Stacked bar chart
- `bfcl_cfg_diffusion/scores/parse_fail_by_category.pdf` -- Per-category parse failure rates
