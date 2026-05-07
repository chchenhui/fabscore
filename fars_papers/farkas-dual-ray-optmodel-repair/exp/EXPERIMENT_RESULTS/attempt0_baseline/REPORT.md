# Attempt-0 Baseline: Qwen2.5-7B-Instruct LP Generation on MAMO

## Experiment Overview

Generate LP models from natural-language optimization problems using Qwen2.5-7B-Instruct with greedy decoding (temperature=0) and evaluate pass@1 on the MAMO-Optimization benchmark (EasyLP + ComplexLP). This establishes the base performance without any repair, providing the attempt-0 results and identifying the infeasible subset for downstream repair experiments.

## Setup

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B parameters)
- **Decoding**: Greedy (temperature=0.0, top_p=1.0)
- **Max tokens**: 4096
- **Benchmark**: MAMO-Optimization (863 instances: 652 EasyLP + 211 ComplexLP)
- **Solver**: HiGHS via highspy (presolve=off, time_limit=60s)
- **Objective tolerance**: relative 1e-6, absolute 1.0
- **GPU**: 1x GPU (96GB), vLLM offline inference
- **Prompt**: MAMO few-shot (1 example) with constraint-naming convention (c0001, c0002, ...)
- **Total runtime**: ~12 minutes (generation + evaluation)

## Key Results

| Metric | Overall | EasyLP | ComplexLP |
|--------|---------|--------|-----------|
| Pass@1 | 58.05% (501/863) | 71.17% (464/652) | 17.54% (37/211) |
| Infeasible rate | 3.59% (31/863) | 1.38% (9/652) | 10.43% (22/211) |

### Status Distribution

| Classification | Count | Rate |
|---------------|-------|------|
| pass | 501 | 58.05% |
| fail-wrong-objective | 269 | 31.17% |
| fail-error | 54 | 6.26% |
| fail-infeasible | 31 | 3.59% |
| fail-unbounded | 8 | 0.93% |

## Key Observations

1. **EasyLP vs ComplexLP gap**: EasyLP pass@1 (71.2%) is much higher than ComplexLP (17.5%), consistent with the difficulty difference.
2. **Low infeasibility rate**: Only 3.6% of instances produce infeasible LPs (31 total). This is the subset that all repair experiments will operate on.
3. **Wrong objective dominates failures**: 31.2% of instances produce optimal but incorrect objectives, suggesting modeling errors rather than syntax errors.
4. **Error rate**: 6.3% fail with parse/solver errors, and 0.9% are unbounded.
5. **Comparison to MIND**: MIND reports 89.4% (EasyLP) / 31.5% (ComplexLP) under its evaluation, but uses a different pipeline (unified evaluation, not direct .lp format). Our lower numbers reflect the stricter `.lp` format generation requirement.

## Files

- `dualrayrank/outputs/attempt0/` - Generated .lp files (863 files, named `{difficulty}_{id}.lp`)
- `dualrayrank/outputs/attempt0/generation_log.jsonl` - Full generation log with raw outputs
- `dualrayrank/outputs/attempt0/infeasible_instances.json` - List of 31 infeasible instances
- `dualrayrank/results/attempt0_results.json` - Full evaluation results with per-instance details
