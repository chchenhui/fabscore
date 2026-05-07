# Per-Category Success Rate Breakdown and Inference Timing Analysis

## Experiment Overview

Detailed breakdown of BFCL success rates by the 7 Non-Live single-turn categories for all three conditions (A: Unconstrained, B: Best-of-2, C: LAVE CFG), plus wall-clock inference timing comparison to assess computational overhead of constrained decoding.

## Setup

- **Model**: LLaDA-8B-Instruct (diffusion LM)
- **Dataset**: BFCL-v3 Non-Live, 350 examples (7 categories x 50)
- **Seeds**: 42, 123, 456 (3 runs per condition)
- **Categories**: simple_python, simple_java, simple_javascript, multiple, parallel, parallel_multiple, irrelevance

## Key Results

### Per-Category Success Rate (%)

| Category | # Examples | (A) Unconstrained | (B) Best-of-2 | (C) LAVE CFG | Delta (C-A) |
|----------|-----------|-------------------|---------------|--------------|-------------|
| Simple (Python) | 50 | 50.7 +/- 0.9 | 50.7 +/- 0.9 | 50.0 +/- 0.0 | -0.7 |
| Simple (Java) | 50 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 1.3 +/- 0.9 | +1.3 |
| Simple (JavaScript) | 50 | 65.3 +/- 0.9 | 66.7 +/- 0.9 | 68.7 +/- 0.9 | +3.3 |
| Multiple | 50 | 35.3 +/- 0.9 | 36.0 +/- 0.0 | 36.0 +/- 0.0 | +0.7 |
| Parallel | 50 | 46.0 +/- 0.0 | 45.3 +/- 0.9 | **53.3 +/- 0.9** | **+7.3** |
| Parallel Multiple | 50 | 14.0 +/- 0.0 | 14.0 +/- 0.0 | 14.0 +/- 0.0 | 0.0 |
| Irrelevance | 50 | 42.0 +/- 0.0 | 41.3 +/- 0.9 | 34.0 +/- 0.0 | **-8.0** |
| **Simple (merged)** | 150 | 38.7 +/- 0.5 | 39.1 +/- 0.8 | 40.0 +/- 0.8 | +1.3 |
| **Overall** | **350** | 36.19 +/- 0.27 | 36.29 +/- 0.23 | 36.76 +/- 0.12 | +0.57 |

### Inference Timing Comparison

| Condition | Avg Time/Instance (s) | Total Time for 350 (s) | Relative Overhead vs (A) |
|-----------|----------------------|------------------------|--------------------------|
| (A) Unconstrained | 11.44 +/- 0.12 | 4,003 | 1.00x |
| (B) Best-of-2 | 22.95 +/- 0.09 | 8,034 | 2.01x |
| (C) LAVE CFG | 9.13 +/- 0.29 | 3,196 | **0.80x** |

## Key Observations

### Category-Level Findings

1. **Parallel category benefits most from LAVE CFG**: +7.3pp success rate improvement (46.0% -> 53.3%), the largest gain across all categories. Parallel function calls have complex multi-call output structure where CFG constraints help ensure syntactically valid outputs.

2. **Simple (JavaScript) shows consistent improvement**: +3.3pp (65.3% -> 68.7%) with LAVE CFG, likely because JS function call syntax benefits from grammar enforcement.

3. **Simple (Java) remains near-zero for all conditions**: The model fundamentally cannot produce correct Java function calls regardless of syntax constraints. This is a semantic limitation, not a formatting issue.

4. **Parallel Multiple shows no improvement**: All conditions yield exactly 14.0%, suggesting this category's difficulty is entirely semantic (choosing the right combination of functions and arguments).

5. **Irrelevance category degrades with LAVE CFG**: -8.0pp (42.0% -> 34.0%). This is expected — irrelevance requires outputting "no function call" (empty list `[]`), but LAVE CFG uses unconstrained generation for this category (bypass). The degradation likely comes from the different diffusion parameters (steps=256, gen_length=512 vs A's steps=128, gen_length=256) affecting output quality for non-constrained categories.

6. **Best-of-2 provides marginal benefit**: Only Simple (JavaScript) shows meaningful improvement (+1.3pp); most categories are unchanged, confirming that parse failures are rare and filtering rarely selects an alternative.

### Timing Findings

1. **LAVE CFG is faster than unconstrained (0.80x)**: Counter-intuitively, LAVE CFG runs in 9.13s/instance vs 11.44s/instance for unconstrained. This is because LAVE's constrained decoding produces valid outputs quickly without retries, and the reduced retry budget (max_retry_num_total=10) limits wasted computation.

2. **Best-of-2 is exactly 2x as expected**: At 22.95s/instance (2.01x), best-of-2 simply generates two samples sequentially, confirming the expected 2x overhead.

3. **Per-category timing varies significantly**: Multiple and Parallel Multiple categories take longest (~15s unconstrained) due to longer output sequences. LAVE CFG reduces all category times substantially, except Irrelevance (27.2s) which uses unconstrained mode with larger generation parameters.

### Implications

- LAVE CFG is **practical**: it is faster than unconstrained baseline and much faster than best-of-2, while achieving the highest success rate.
- The benefit concentrates in categories with complex output structure (Parallel), while categories where the bottleneck is entirely semantic (Parallel Multiple, Simple Java) see no benefit.
- The Irrelevance regression suggests that the larger generation parameters used for LAVE CFG (steps=256, gen_length=512) may hurt non-constrained categories; future work could use condition-specific parameters.

## Artifacts

- Analysis script: `bfcl_cfg_diffusion/analysis/category_breakdown.py`
- Visualization script: `bfcl_cfg_diffusion/analysis/plot_category_breakdown.py`
- Machine-readable results: `bfcl_cfg_diffusion/scores/category_breakdown_results.json`
- Category breakdown chart: `bfcl_cfg_diffusion/scores/category_breakdown.pdf`
- Timing comparison chart: `bfcl_cfg_diffusion/scores/timing_comparison.pdf`
