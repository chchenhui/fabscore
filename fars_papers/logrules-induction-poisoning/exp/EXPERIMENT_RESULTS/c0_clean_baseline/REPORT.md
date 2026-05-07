# C0 Clean LogRules Pipeline Baseline

## Experiment Overview

Evaluate the full clean LogRules-style pipeline (induction -> rule ranking -> deduction) on three LogHub datasets using gpt-4o-mini for induction/ranking and Qwen2.5-7B-Instruct (unaligned, base instruct model) for deduction. This is the C0 Clean condition -- the primary comparison reference for measuring the impact of poisoning (C1) and the effectiveness of the defense (C2).

## Setup

- **Induction model**: gpt-4o-mini via LEMMA_MAAS API proxy
- **Deduction model**: Qwen/Qwen2.5-7B-Instruct (7B parameters, no alignment)
- **Inference**: vLLM v0.12.1, 1x GPU, temperature=0, top_p=1, max_tokens=2048
- **Pipeline**: K=10 induction examples -> rule generation -> rule ranking (top N=15) -> rule-based deduction
- **Datasets**: BGL, Linux, HDFS (each 2000 logs from LogHub)
- **Splits**: K=10 induction, 50 canary, ~1940 test per seed
- **Seeds**: 42, 123, 456
- **Metrics**: PA (Parsing Accuracy), FTA (F1 of Template Accuracy), wildcard_ratio, canary_PA

## Key Results

| Dataset | PA (mean +/- std) | FTA (mean +/- std) | wildcard_ratio (mean +/- std) | canary_PA (mean +/- std) |
|---------|-------------------|---------------------|-------------------------------|--------------------------|
| BGL     | 0.2792 +/- 0.0887 | 0.2530 +/- 0.0881  | 0.1281 +/- 0.0249            | 0.3333 +/- 0.1405        |
| Linux   | 0.1082 +/- 0.0677 | 0.3023 +/- 0.0246  | 0.1424 +/- 0.0413            | 0.1067 +/- 0.0757        |
| HDFS    | 0.1180 +/- 0.1587 | 0.0188 +/- 0.0325  | 0.1905 +/- 0.0763            | 0.1000 +/- 0.1562        |

### Per-Seed Breakdown

| Dataset | Seed | test_PA | test_FTA | wildcard_ratio | canary_PA | num_rules |
|---------|------|---------|----------|----------------|-----------|-----------|
| BGL     | 42   | 0.2902  | 0.2290   | 0.1011         | 0.3200    | 6         |
| BGL     | 123  | 0.1856  | 0.1795   | 0.1503         | 0.2000    | 10        |
| BGL     | 456  | 0.3619  | 0.3506   | 0.1329         | 0.4800    | 5         |
| Linux   | 42   | 0.1809  | 0.2740   | 0.1174         | 0.1600    | 7         |
| Linux   | 123  | 0.0969  | 0.3150   | 0.1900         | 0.1400    | 8         |
| Linux   | 456  | 0.0469  | 0.3180   | 0.1199         | 0.0200    | 6         |
| HDFS    | 42   | 0.0000  | 0.0000   | 0.2390         | 0.0000    | 6         |
| HDFS    | 123  | 0.0557  | 0.0000   | 0.2300         | 0.0200    | 6         |
| HDFS    | 456  | 0.2985  | 0.0563   | 0.1026         | 0.2800    | 7         |

### Phase-0 Gating Thresholds

| Dataset | mean_PA | std_PA | threshold_X = max(0.05, 2*std_PA) |
|---------|---------|--------|-------------------------------------|
| BGL     | 0.2792  | 0.0887 | 0.1773                              |
| Linux   | 0.1082  | 0.0677 | 0.1355                              |
| HDFS    | 0.1180  | 0.1587 | 0.3174                              |

## Key Observations

1. **Much lower than LogRules paper results**: The LogRules paper reports PA of 98.7% (BGL), 86.2% (Linux), 92.4% (HDFS) using their full pipeline including an alignment stage. Our setup deliberately omits alignment to study the raw effect of rules on the base instruct model, which is the correct setup for this poisoning study.

2. **Compared to zero-shot baseline**: BGL C0 (27.9%) is actually WORSE than zero-shot (54.1%), while Linux C0 (10.8%) and HDFS C0 (11.8%) improve over zero-shot (5.8% and 0.0% respectively). For BGL, the generic rules cause the model to over-abstract static tokens.

3. **High variance across seeds**: Especially for HDFS (std=15.9pp) and BGL (std=8.9pp), reflecting strong sensitivity to which K=10 examples are sampled for induction. HDFS seed 42 achieves 0% PA while seed 456 achieves 29.8%.

4. **Few rules generated**: gpt-4o-mini generates only 5-10 rules from K=10 examples (all below the N=15 cap), so no ranking truncation is applied. The rules are high-level and generic rather than dataset-specific.

5. **These C0 results serve as the baseline for C1/C2 comparison**: The poisoning effect will be measured as the deviation from these C0 numbers. The high natural variance (especially HDFS) means that the Phase-0 gating threshold X is large, requiring the poisoning attack to cause substantial degradation to be detectable.
