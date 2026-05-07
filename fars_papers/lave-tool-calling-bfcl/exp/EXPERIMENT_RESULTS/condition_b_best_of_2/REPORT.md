# Condition B: Best-of-2 with AST-Parseability Filter

## Experiment Overview

Evaluated LLaDA-8B-Instruct with best-of-2 inference-time scaling using AST-parseability filtering on BFCL-v3 Non-Live subset. For each instance, 2 independent diffusion samples are generated using different sub-seeds, then the parseable one is selected (if exactly one parses); otherwise the first sample is selected. This controls for the "retry until parseable" deployment workaround without using ground-truth labels.

## Setup

- **Model**: GSAI-ML/LLaDA-8B-Instruct (bfloat16)
- **Decoding**: Semi-autoregressive diffusion, unconstrained (`constrain=False`), 2 samples per instance
- **Sub-seed scheme**: For global seed s, sub-seeds are s*2 and s*2+1
- **Selection rule**: If exactly one sample parses via `ast.parse()`, select it; otherwise select sample 1
- **Hyperparameters**: max_tokens=256, steps=128, block_length=32, temperature=0.2
- **Dataset**: BFCL-v3 Non-Live, 350 examples (50 per category x 7 categories)
- **Categories**: simple_python, simple_java, simple_javascript, multiple, parallel, parallel_multiple, irrelevance
- **Seeds**: 42, 123, 456
- **GPU**: 1x GPU per seed, 3 seeds in parallel
- **Inference time**: ~23.0s per example (2 samples), ~134 min total per seed

## Key Results

| Metric | Mean | Std |
|--------|------|-----|
| Overall Success Rate | 36.29% | 0.23% |
| Overall AST Parse Rate (selected) | 93.62% | 0.27% |
| Mean Inference Time | 22.95s | - |

### Comparison with Condition A (Unconstrained)

| Metric | Condition A | Condition B | Delta |
|--------|------------|------------|-------|
| Success Rate | 36.19% +/- 0.27% | 36.29% +/- 0.23% | +0.10pp |
| AST Parse Rate | 91.71% +/- 1.02% | 93.62% +/- 0.27% | +1.91pp |
| Inference Time | 11.44s | 22.95s | +11.51s (2x) |

### Per-Category Success Rates (Mean +/- Std)

| Category | Success Rate | AST Parse Rate |
|----------|-------------|----------------|
| simple_python | 50.7% +/- 0.9% | 96.7% +/- 0.9% |
| simple_java | 0.0% +/- 0.0% | 88.7% +/- 0.9% |
| simple_javascript | 66.7% +/- 0.9% | 90.7% +/- 0.9% |
| multiple | 36.0% +/- 0.0% | 98.0% +/- 1.6% |
| parallel | 45.3% +/- 0.9% | 98.7% +/- 0.9% |
| parallel_multiple | 14.0% +/- 0.0% | 91.3% +/- 0.9% |
| irrelevance | 41.3% +/- 0.9% | 91.3% +/- 0.9% |

### Seed-Level Results

| Seed | Success Rate | AST Parse Rate |
|------|-------------|----------------|
| 42 | 36.29% | 93.43% |
| 123 | 36.57% | 94.00% |
| 456 | 36.00% | 93.43% |

### Pre-Selection Statistics

| Metric | Mean | Std |
|--------|------|-----|
| Individual sample 1 parse rate | 92.86% | 0.40% |
| Individual sample 2 parse rate | 92.48% | 0.59% |
| Average individual parse rate | 92.67% | - |
| Both samples parse | 91.71% | - |
| Neither sample parses | 6.38% | - |
| Exactly one parses (filter helps) | 1.90% | 0.88% |
| Selected output parse rate | 93.62% | 0.27% |

## Key Observations

1. **Negligible success rate improvement (+0.10pp)**: Best-of-2 with parseability filtering provides essentially no improvement in BFCL success rate over unconstrained decoding. The gains are within noise.

2. **Small AST parse rate improvement (+1.91pp)**: The selected output parse rate increases from 91.71% to 93.62%, a modest but consistent improvement.

3. **Filtering rarely helps (1.9% of cases)**: In only ~7 out of 350 examples does exactly one of the two samples parse. Most of the time (91.7%) both parse, and 6.4% of the time neither parses. This means the filter has very limited opportunity to improve outcomes.

4. **2x inference cost**: Best-of-2 doubles the inference time (~23s vs ~11.4s per example) for negligible benefit.

5. **Per-category results nearly identical to Condition A**: No category shows meaningful improvement from the retry strategy.

6. **Low temperature produces similar samples**: At temperature=0.2, the two diffusion samples are often identical or very similar, limiting the diversity needed for effective best-of-N.

## Files

- Raw outputs: `bfcl_cfg_diffusion/results/best_of_2/seed_{42,123,456}.jsonl`
- Evaluation results: `bfcl_cfg_diffusion/results/best_of_2/eval_results.json`
- Inference script: `bfcl_cfg_diffusion/inference/run_best_of_2.py`
- Shell script: `bfcl_cfg_diffusion/scripts/run_best_of_2.sh`
- Evaluation script: `bfcl_cfg_diffusion/evaluation/run_bfcl_eval.py`
- WandB logs: `wandb/` (offline mode)
