# Condition A: Unconstrained Diffusion Decoding Baseline

## Experiment Overview

Evaluated LLaDA-8B-Instruct with standard unconstrained diffusion decoding on BFCL-v3 Non-Live subset. This establishes the floor performance for comparison with constrained diffusion decoding (Conditions B and C).

## Setup

- **Model**: GSAI-ML/LLaDA-8B-Instruct (bfloat16)
- **Decoding**: Semi-autoregressive diffusion, unconstrained (`constrain=False`)
- **Hyperparameters**: max_tokens=256, steps=128, block_length=32, temperature=0.2
- **Dataset**: BFCL-v3 Non-Live, 350 examples (50 per category x 7 categories)
- **Categories**: simple_python, simple_java, simple_javascript, multiple, parallel, parallel_multiple, irrelevance
- **Seeds**: 42, 123, 456
- **GPU**: 1x GPU per seed, 3 seeds in parallel
- **Inference time**: ~11.4s per example, ~67 min total per seed

## Key Results

| Metric | Mean | Std |
|--------|------|-----|
| Overall Success Rate | 36.19% | 0.27% |
| Overall AST Parse Rate | 91.71% | 1.02% |
| Mean Inference Time | 11.44s | - |

### Per-Category Success Rates (Mean +/- Std)

| Category | Success Rate | AST Parse Rate |
|----------|-------------|----------------|
| simple_python | 50.7% +/- 0.9% | 95.3% +/- 0.9% |
| simple_java | 0.0% +/- 0.0% | 86.7% +/- 0.9% |
| simple_javascript | 65.3% +/- 0.9% | 89.3% +/- 0.9% |
| multiple | 35.3% +/- 0.9% | 96.7% +/- 1.9% |
| parallel | 46.0% +/- 0.0% | 97.3% +/- 0.9% |
| parallel_multiple | 14.0% +/- 0.0% | 87.3% +/- 2.5% |
| irrelevance | 42.0% +/- 0.0% | 89.3% +/- 1.9% |

### Seed-Level Results

| Seed | Success Rate | AST Parse Rate |
|------|-------------|----------------|
| 42 | 36.57% | 93.14% |
| 123 | 36.00% | 90.86% |
| 456 | 36.00% | 91.14% |

## Key Observations

1. **AST parse rate is high (91.7%)** despite unconstrained decoding, showing LLaDA can generate syntactically valid function calls most of the time.

2. **simple_java has 0% success** despite 87% AST parse rate. The model generates Python-style function calls when Java-style is expected, causing semantic failures.

3. **Parallel calls work well (46%)** - the model can generate multiple function calls in a single output.

4. **parallel_multiple is hardest (14%)** - combining parallel + multiple selection is challenging.

5. **Irrelevance detection at 42%** - the model incorrectly generates function calls for irrelevant queries 58% of the time.

6. **Very stable across seeds** - standard deviations are very small (<1pp for most categories).

## Comparison with Published Anchor

The Bitter Lesson paper (arXiv:2601.12979) reports LLaDA-8B Non-Live = 23.0% (Table 2). Our result of 36.2% is 13.2pp higher, exceeding the +/-5pp tolerance. Likely explanations:

- **Different category set**: Bitter Lesson uses 6 categories (300 examples) without irrelevance; we use 7 (350).
- **Different sampling**: They may use different random seeds or full categories.
- **BFCL version differences**: They report S=8%, P=0% which are dramatically lower than ours (S=39%, P=46%), suggesting possible differences in prompt formatting or BFCL data version.
- **Our setup is internally consistent** and will be the baseline for Conditions B and C comparison.

## Files

- Raw outputs: `bfcl_cfg_diffusion/results/unconstrained/seed_{42,123,456}.jsonl`
- Evaluation results: `bfcl_cfg_diffusion/results/unconstrained/eval_results.json`
- Inference script: `bfcl_cfg_diffusion/inference/run_unconstrained.py`
- Evaluation script: `bfcl_cfg_diffusion/evaluation/run_bfcl_eval.py`
- WandB logs: `wandb/` (offline mode)
