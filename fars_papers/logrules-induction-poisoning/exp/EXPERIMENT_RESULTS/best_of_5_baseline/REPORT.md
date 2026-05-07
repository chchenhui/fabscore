# Best-of-5 Inference-Time Scaling Baseline

## Experiment Overview

Evaluate a best-of-5 decoding strategy for zero-shot log parsing on three LogHub datasets (BGL, Linux, HDFS) using Qwen2.5-7B-Instruct served via vLLM. For each log line, generates 5 candidate templates at temperature=0.7 and selects the first one that passes a template-format verifier. This represents an inference-time scaling baseline that improves over pure zero-shot without using any induced rules.

## Setup

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B parameters)
- **Inference**: vLLM v0.12.1, 1x GPU, n=5 candidates, temperature=0.7, max_tokens=2048
- **Prompt**: Same zero-shot deduction prompt as baseline (no rules, no examples)
- **Verifier**: `is_valid_template(template, raw_log)` checks:
  1. Only `<*>` placeholders (no `<*blk*>`, `<*.*:*>`, etc.)
  2. Constant tokens are a subset of raw log tokens (structural consistency)
  3. Template is not entirely wildcards
- **Selection**: First candidate passing verifier; fallback to first candidate if none pass
- **Datasets**: BGL, Linux, HDFS -- each 2000 logs from LogHub
- **Splits**: K=10 induction, 50 canary, ~1940 test per seed (induction/canary unused)
- **Seeds**: 42, 123, 456
- **Metrics**: PA (Parsing Accuracy), FTA (F1 of Template Accuracy), wildcard_ratio

## Key Results

| Dataset | PA (mean +/- std) | FTA (mean +/- std) | wildcard_ratio (mean +/- std) |
|---------|-------------------|---------------------|-------------------------------|
| BGL     | 0.4844 +/- 0.0075 | 0.1350 +/- 0.0127  | 0.1085 +/- 0.0018            |
| Linux   | 0.0703 +/- 0.0035 | 0.1848 +/- 0.0041  | 0.1606 +/- 0.0010            |
| HDFS    | 0.0007 +/- 0.0003 | 0.0000 +/- 0.0000  | 0.2338 +/- 0.0008            |

### Comparison with Zero-Shot Baseline

| Dataset | Zero-Shot PA | Best-of-5 PA | Delta PA |
|---------|-------------|-------------|----------|
| BGL     | 0.5414      | 0.4844      | -0.0570  |
| Linux   | 0.0584      | 0.0703      | +0.0119  |
| HDFS    | 0.0000      | 0.0007      | +0.0007  |

## Key Observations

1. **BGL PA decreases slightly (-5.7 pp)** with best-of-5. The verifier's structural consistency check filters out some templates that would have matched exactly under greedy decoding (temperature=0). The higher temperature introduces noise in templates that were already correct under greedy decoding. FTA also drops from 0.198 to 0.135.

2. **Linux PA improves modestly (+1.2 pp)** from 5.8% to 7.0%. The verifier successfully filters non-standard placeholder formats (a major issue in zero-shot), and the multiple sampling provides more diverse candidates. FTA drops slightly from 0.241 to 0.185.

3. **HDFS remains near zero** (0.07% PA). The verifier catches some format violations but HDFS logs require understanding complex structures (IPs, block IDs, file paths) that the model cannot handle without rules regardless of sampling strategy.

4. **Wildcard ratio increases** across all datasets (BGL: 5.7% -> 10.9%, Linux: 11.4% -> 16.1%, HDFS: 17.3% -> 23.4%), indicating the verifier selects templates with more `<*>` placeholders by filtering out non-standard wildcards.

5. **Very low variance across seeds** (< 1 pp for PA), consistent with zero-shot results since induction/canary splits are unused.

6. **Best-of-5 does not meaningfully improve over zero-shot** -- the inference-time scaling approach has limited value without rule guidance. The main bottleneck is not template format quality but structural understanding of log patterns, which requires induced rules.
