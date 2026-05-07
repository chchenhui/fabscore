# Paper Outline: The Consistency Confound: Why Stronger Alignment Can Break a Black-Box Instantiation of Semantic Entropy

**Authors:** Gemini 2.5 Pro, Claude Code, Dhruv Trehan

---

## Abstract

- **Context:** Black-box monitoring of Large Language Models (LLMs) for jailbreak attempts is a critical safety problem. An intuitive approach is to use the semantic inconsistency of multiple responses, quantified by Semantic Entropy (SE), as a behavioral signal of a model's internal conflict.
- **Method:** We test a black-box, embedding-clustering instantiation of SE as a zero-shot jailbreak detector. This is, to our knowledge, the first systematic study of this method for jailbreak detection.
- **Key Finding:** We comprehensively refute this hypothesis. Across four models, including SOTA 70B+ parameter models, and two diverse benchmarks, we find this SE variant is an unreliable and brittle detector whose failure is robust to paraphrasing.
- **Core Contribution:** We identify and validate the **"Consistency Confound"**: stronger safety alignment leads models to produce highly consistent, templated refusals. We operationalize this signature as: high duplicate rate (>50%) OR high refusal templates (>60%) with a low cluster count (≤2). This mechanism accounts for **73.3% (44/60)** of false negatives for Llama-4-Scout and **97.5% (79/81)** for Qwen-2.5-7B. Our findings are supported by Wilson CIs for FNR metrics.
- **Implications:** Our findings are a cautionary tale for behavioral detection methods, demonstrating that as models become better aligned, they may paradoxically become harder to monitor with techniques that treat response diversity as a signal of safety.

---

## 1. Introduction

- **The Challenge of Black-Box Safety:** Introduce the problem of monitoring closed, API-based LLMs for jailbreaks, where internal access is impossible.
- **The Promise of Behavioral Signals:** Frame the intuitive appeal of using behavioral signals, like response inconsistency, as a proxy for a model's internal conflict when processing a harmful prompt. This approach is inspired by the successful use of consistency to detect factual hallucinations (SelfCheckGPT).
- **Our Core Claims:** We find this plausible mechanism fails in practice. This paper makes three central claims:
    1. **SE Underperforms Baselines:** Our black-box SE variant is consistently outperformed by simpler consistency baselines.
    2. **Performance is Unreliable and Brittle:** SE's effectiveness is an artifact of specific hyperparameter choices (`τ`, `N`). We observe severe score degeneracy at common settings (e.g., `τ=0.1` yields FPR=0.0, FNR=1.0 on JailbreakBench), undermining its usefulness as a continuous detector.
    3. **Failure Generalizes via the Consistency Confound:** The primary failure mode is the "Consistency Confound," where strong alignment produces consistent refusals. This failure persists and worsens on SOTA models.
- **Deployment Relevance:** In black-box monitoring pipelines, alarms based on SE may underfire as model alignment strengthens, making this behavioral diversity signal anti-correlated with risk.

---

## 2. Related Work

- **Positioning:** Our work is the first, to our knowledge, to systematically evaluate a black-box, embedding-clustering instantiation of SE for jailbreak detection and to quantify the Consistency Confound as its dominant failure mechanism.
- **Taxonomy of Defenses:**
    - **White-box (Internal Monitors):** Methods like GradSafe and HiddenDetect analyze internal gradients or activations.
    - **Decoding-time (Output Steering):** Methods like SafeDecoding and RAIN modify the generation process to promote safety.
    - **Black-box (Perturbation-based):** Methods like SemanticSmooth perturb the input and check for consistent outputs. Our method is an **output-sampling** variant in this family.
    - **Guard Stacks & Agents:** Supervised classifiers like Llama-Guard that assess prompt/response safety against a fixed taxonomy.
    - **Uncertainty Lineage:** Our method adapts a signal from hallucination detection (SelfCheckGPT, canonical SE).
- *Source:* `mentor_docs/literature_review_synthesis_notes_1.md` for full taxonomy and citations.

---

## 3. Methodology

- **3.1. Detection Methods:**
    - **Primary: Semantic Entropy (SE):** `idea_14_workspace/src/core/semantic_entropy.py`
    - **Baselines:** Avg Pairwise BERTScore, Embedding Variance, Levenshtein Variance. `idea_14_workspace/src/core/baseline_metrics.py`
- **3.2. Experimental Setup:**
    - **Models:** Llama-4-Scout (`meta-llama/llama-4-scout`), Qwen-2.5-7B, Qwen-2.5-72B, Llama-3.3-70B.
    - **Datasets:**
        - **JailbreakBench (JBB):** 120-prompt split. *Source:* `idea_14_workspace/data/manifests/jbb_validation_ids.json`.
        - **HarmBench-Contextual (HBC):** 162 prompts. *Source:* `idea_14_workspace/data/processed/h2_harmbench_twins_test.jsonl`.
    - **Implementation Details:**
        - **API:** OpenRouter. Generation logs (`idea_14_workspace/outputs/h*/response_generation/*_logs.md`) document exact model strings.
        - **Generation Params:** N=5 (or 10), T=0.7, Top-p=0.95, Max Tokens=1024.
        - **Embedding Model:** `Alibaba-NLP/gte-large-en-v1.5`.
        - **Reproducibility:** Global seed: 42. `idea_14_workspace/configs/project_config.yaml`.
- **3.3. Evaluation Protocol:**
    - **Metrics:** AUROC and FNR@t5FPR.
    - **Thresholding:** Score thresholds selected on the full evaluation dataset's benign prompts to achieve an FPR as close as possible to, but not exceeding, 5%. We report `actual_fpr` achieved.
    - **Statistical Analysis:** We report 95% Wilson CIs for FNR metrics. DeLong CIs for AUROC are suppressed for SE due to severe score degeneracy. `idea_14_workspace/src/core/statistical_tests.py`.

---

## 4. Results: Detector Performance and Generalization

- **4.1. On JailbreakBench, SE Underperforms and Shows Degeneracy (H1)**
    - At canonical τ=0.2, Llama-4-Scout FNR is 0.850 [0.739, 0.919] (actual_fpr=0.0) and Qwen-2.5-7B FNR is 0.983 [0.911, 0.997] (actual_fpr=0.05).
    - *CI Source:* `idea_14_workspace/outputs/statistical_analysis/h1_statistical_results.json`.
    - **_Figure 1: AUROC Comparison on JailbreakBench._** File: `idea_14_workspace/outputs/visualisation/figures/figure_1_auroc_comparison.png`.
- **4.2. Performance Generalizes Poorly to HarmBench (H2)**
    - **Llama:** SE (τ=0.2, FNR=0.765) is outperformed by Embedding Variance (FNR=0.605).
    - **Qwen:** SE shows a brittle "win" at τ=0.1 (FNR=0.630) but collapses at canonical τ=0.2 (FNR=0.889).
    - *CI Source:* `idea_14_workspace/outputs/statistical_analysis/h2_statistical_results.json`.
    - **_Figure 2: AUROC Comparison on HarmBench._** File: `idea_14_workspace/outputs/visualisation/figures/figure_1h_auroc_harmbench.png`.
    - **_Table 2: FNR@t5FPR Comparison Across Datasets._** File: `idea_14_workspace/outputs/visualisation/tables/table_2_fnr_comparison.csv`.
- **4.3. Failure Persists on State-of-the-Art Models (H7)**
    - **Qwen-72B:** SE FNR is 1.0 (actual_fpr=0.0) at τ=0.1, demonstrating extreme degeneracy. Best baseline (Emb. Var.) AUROC is 0.733 vs SE's 0.636.
    - **Llama-70B:** Emb. Var. is superior on both AUROC (0.809 vs SE's 0.787) and FNR (0.450 vs SE's best of 0.550).
    - *CI Source:* `idea_14_workspace/outputs/statistical_analysis/h7_statistical_results.json`.
    - **_Figure 3: Comprehensive AUROC Comparison Across All Models._** File: `idea_14_workspace/outputs/visualisation/figures/figure_1c_auroc_comprehensive.png`.

---

## 5. Results: Analysis of Failure Modes

- **5.1. Length is a Minor Confounder (H3):** OLS regression of SE score on log(length) yields weak correlations for both Llama (R²=0.103) and Qwen (R²=0.00013).
  - *Source:* `idea_14_workspace/outputs/h3/results/`.
  - **_Figure 4: SE vs. Response Length._** File: `idea_14_workspace/outputs/visualisation/figures/figure_2_se_vs_length.png`.
- **5.2. Brittleness to Hyperparameters (H4):** The H4 criterion ("FNR increases by >20pp when τ changes 0.1→0.2 OR N changes 5→10") was met. For Qwen/HBC, ΔFNR for τ was +25.9pp, while for N it was -16.1pp.
  - *Source:* `idea_14_workspace/outputs/h4/evaluation/h4_brittleness_results.json`.
  - **_Figure 5: Hyperparameter Brittleness._** File: `idea_14_workspace/outputs/visualisation/figures/figure_3_hyperparameter_brittleness.png`.
- **5.3. Robustness to Paraphrasing (H5):** The failure mode is robust to paraphrasing, undermining a contamination-based explanation.
  - **_Figure 6: Paraphrase Impact on FNR._** File: `idea_14_workspace/outputs/visualisation/figures/figure_5_paraphrase_impact.png`.
- **5.4. The Consistency Confound (H6):** The primary failure mechanism is models producing *consistent* outputs for harmful prompts.
    - **Quantitative Audit:** Our CC signature explained **73.3% (44/60)** of Llama@JBB FNs and **97.5% (79/81)** of Qwen@HBC FNs. Both "All Unique" and "Common Multi-τ" audit strategies yielded similar high rates (e.g., 97.5% vs 98.7% for Qwen).
    - **Qualitative Evidence:**
        - **Consistent Refusal (Llama, jbb_75):** All 5 responses are near-identical refusals, e.g., "I can’t help you with that. Creating a phishing email...". SE=0.0.
        - **Consistent Jailbreak (Qwen, h2_harmful_061):** All 5 responses provide distinct but semantically identical SQL injection code snippets. SE=0.0.
    - **_Figure 7: Breakdown of False Negative Causes._**
        - **File:** `idea_14_workspace/outputs/visualisation/figures/figure_4_fn_breakdown.png`.
        - **Data Sources:** H6 `..._qualitative_audit_results.json` files for both Llama and Qwen.

---

## 6. Discussion and Conclusion

- **Conclusion:** The core assumption of our tested SE variant—that inconsistency signals jailbreaks—is inverted in the safety domain. The "Consistency Confound" is a fundamental, scale-aggravated flaw for this method.
- **Design Principles for Practitioners:**
    1.  Beware SE's score degeneracy at small `τ`; it can lead to non-discriminatory FNR=1.0 behavior.
    2.  When using threshold-based metrics, always report `actual_fpr` for transparency.
    3.  Simpler dispersion metrics like Embedding Variance often provide a more robust signal than SE.
    4.  Audit for the Consistency Confound; low diversity is not a reliable signal of safety.
    5.  Be aware of threat models that bypass this detector by design (e.g., backdoors, many-shot conditioning).
- **Limitations & Future Work:**
    - This study is scoped to a single SE variant, random seed, and a limited set of models/datasets.
    - Future work should perform a direct comparison to canonical SE, evaluate against practical baselines (e.g., Llama-Guard, LLM-as-judge), and conduct a more rigorous evaluation with multi-seed runs and a held-out calibration set.