# Zero-Shot Log Parsing Baseline

## Experiment Overview

Evaluate zero-shot log parsing (no rule repository) on three LogHub datasets (BGL, Linux, HDFS) using Qwen2.5-7B-Instruct served via vLLM. This establishes the lower-bound parsing performance when the deduction model receives no induced rules.

## Setup

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B parameters)
- **Inference**: vLLM v0.12.1, 1x GPU, temperature=0, top_p=1, max_tokens=2048
- **Prompt**: Zero-shot deduction prompt (no rules, no examples) following LogRules case-based structure
- **Datasets**: BGL (supercomputer), Linux (OS), HDFS (distributed storage) -- each 2000 logs from LogHub
- **Splits**: K=10 induction, 50 canary, ~1940 test per seed (induction/canary unused in this baseline)
- **Seeds**: 42, 123, 456 (control different random splits of induction/canary/test)
- **Metrics**: PA (Parsing Accuracy), FTA (F1 of Template Accuracy), wildcard_ratio

## Key Results

| Dataset | PA (mean +/- std) | FTA (mean +/- std) | wildcard_ratio (mean +/- std) |
|---------|-------------------|---------------------|-------------------------------|
| BGL     | 0.5414 +/- 0.0037 | 0.1981 +/- 0.0038  | 0.0565 +/- 0.0011            |
| Linux   | 0.0584 +/- 0.0003 | 0.2414 +/- 0.0113  | 0.1138 +/- 0.0004            |
| HDFS    | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000  | 0.1726 +/- 0.0005            |

## Key Observations

1. **BGL achieves moderate PA (54.1%)** because many BGL templates are simple static strings with no variables, which the model can reproduce exactly. FTA is low (19.8%) because the model fails to consistently produce the correct template across all logs in a template group.

2. **Linux PA is very low (5.8%)** despite moderate FTA (24.1%). The model partially understands the structure but uses inconsistent wildcard formats, resulting in few exact string matches.

3. **HDFS achieves 0% PA and 0% FTA**. The model uses diverse non-standard wildcard formats (e.g., `<*.*:*>`, `<*elongated number*>`, `<*blk_*>`) instead of the standard `<*>` placeholder. HDFS logs contain complex structured data (IP addresses, block IDs, file paths) that the model struggles to abstract correctly without guidance.

4. **Low wildcard_ratio across all datasets** (5.7-17.3%) indicates the model is conservative about abstracting variables without explicit rules. Compare with the ground truth which has significantly more wildcards in HDFS templates.

5. **Very low variance across seeds** confirms that the random split of induction/canary/test has minimal impact on zero-shot performance (since induction/canary are unused).

6. These results establish a clear lower-bound. The LogRules paper reports PA of 98.7% (BGL), 86.2% (Linux), 92.4% (HDFS) with rules, demonstrating the dramatic improvement that rule-based prompting provides.
