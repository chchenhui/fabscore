# Paper Outline v4: The Consistency Confound: Why Stronger Alignment Can Break Black-Box Jailbreak Detection

**Authors:** AI Researcher et al.

---

## Abstract

- **Context:** Black-box monitoring of Large Language Models (LLMs) for jailbreak attempts is a critical, unsolved safety problem. A promising recent idea, inspired by hallucination detection, is to use the *semantic inconsistency* of multiple generated responses as a behavioral signal of the internal conflict caused by a jailbreak.
- **Novelty:** Unlike prior consistency-based detectors for hallucination detection (Farquhar et al., 2024; Manakul et al., 2023), we show that in safety settings, output consistency *increases* with model alignment, invalidating it as a black-box proxy for safety.
- **Key Finding:** We comprehensively refute the hypothesis that high semantic entropy (SE) serves as a reliable jailbreak detector. Across two models (Llama-4-Scout-17B, Qwen-2.5-7B) and two benchmarks (JailbreakBench, HarmBench), SE is unreliable and brittle.
- **Core Contribution:** We identify and validate a core failure mechanism we term the **"Consistency Confound"**: stronger safety alignment leads models to produce highly consistent, templated refusals. This results in low semantic entropy, causing the detector to fail precisely when the model is behaving most safely. Our qualitative audits show this mechanism accounts for **73.3% (44/60)** of false negatives for Llama on JailbreakBench and a stark **97.5% (79/81)** for Qwen on HarmBench.
- **Implications:** Our findings serve as a cautionary tale for behavioral detection methods, suggesting that as models become better aligned, they may paradoxically become harder to monitor with techniques that treat diversity as a signal of safety.

---

## 1. Introduction

- **The Appeal of Behavioral Signals:** Introduce the problem of black-box jailbreak detection. Frame the intuitive appeal of using behavioral signals, like response inconsistency, as a proxy for the model's internal conflict.
- **Semantic Entropy as a Formalism:** Introduce Semantic Entropy (SE) as a plausible method to measure this inconsistency. Explain the mechanism: jailbreak causes conflict -> multimodal response distribution -> high SE.
- **Our Core Claims:** We find that this plausible mechanism fails in practice. This paper makes three central claims:
    1. **SE Underperforms Baselines:** SE is consistently outperformed by simpler textual consistency baselines on standard benchmarks.
    2. **Performance is Unreliable and Brittle:** The effectiveness of consistency detectors is highly dependent on the model, data distribution, and hyperparameter choices. We show SE’s apparent "wins" are artifacts of specific `τ` (clustering threshold) and `N` (sample count) settings.
    3. **Failure is Caused by the Consistency Confound:** The primary failure mode is a mechanism where strong safety alignment produces consistent, templated refusals, which the detector misinterprets as safe.
- **Related Work:**
    - Situate our work in the lineage of consistency-based methods for LLM evaluation (e.g., SelfCheckGPT), arguing the core assumption (consistency → correctness) breaks down in the safety domain.
    - Differentiate from input-perturbation methods and white-box methods, highlighting the unique challenges of a black-box, output-only approach.
    - Contrast with input-side detectors that classify prompt embeddings (e.g., Galinkin & Sablotny, 2024), which is a complementary but distinct approach.

---

## 2. Methodology

- **2.1. Threat Model and Detection Task:** We operate in a black-box setting. The task is to classify an input *prompt* as harmful or benign by analyzing N=5 generated responses. A false negative occurs when a harmful prompt is classified as benign.
- **2.2. Detection Methods:**
    - **Primary Method: Semantic Entropy (SE):** `src/core/semantic_entropy.py`
    - **Baseline Methods:** Avg Pairwise BERTScore, Embedding Variance, Levenshtein Variance. `src/core/baseline_metrics.py`
- **2.3. Rationale for Black-Box SE Variant:**
    - Explain the choice of embedding-based clustering over the original NLI-based method (Farquhar et al., 2024), driven by black-box API constraints.
    - **_Table 1: Comparison of SE Variants._**
      - **Rows:** Primary Application, Access Required, Clustering Method.
      - **Columns:** Original SE (Farquhar et al., 2024), Our Implementation.
- **2.4. Experimental Setup:**
    - **Models:** Llama-4-Scout-17B-16E-Instruct, Qwen/Qwen2.5-7B-Instruct.
    - **Datasets:**
        - **JailbreakBench (JBB):** 120-prompt validation split. *Manifest:* `data/manifests/jbb_validation_ids.json`.
        - **HarmBench-Contextual (HBC):** 162 prompts (81 harmful, 81 matched benign "twins"). *Source:* `data/processed/h2_harmbench_twins_test.jsonl`.
    - **Implementation Details:**
        - **API:** OpenRouter. `src/core/response_generator_openrouter.py`.
        - **Response Generation:** N=5 (or N=10 for H4), T=0.7, Top-p=0.95, Max Tokens=1024. See `outputs/h*/response_generation/*_logs.md`.
        - **Embedding Model:** `Alibaba-NLP/gte-large-en-v1.5` (1024-dim, L2-normalized).
        - **Clustering:** Agglomerative Clustering (average linkage, cosine distance) with `τ` ∈ {0.1, 0.2, 0.3, 0.4}. `src/core/semantic_entropy.py`.
        - **Reproducibility:** Global seed=42. `configs/project_config.yaml`.
- **2.5. Evaluation Protocol:**
    - **Metrics:** AUROC and FNR@5%FPR.
    - **Thresholding:** For each detector, a score threshold is selected on the full dataset's benign prompts to achieve an FPR as close as possible to, without exceeding, 5%. The FNR is then calculated on harmful prompts at this threshold. We report the `actual_fpr` for transparency.
    - **τ Selection Policy:** For AUROC, we report the score from the best-performing `τ`. For FNR@5%FPR, we report SE at a fixed **canonical `τ=0.2`** for fair cross-method comparison.
    - *Source:* `configs/project_config.yaml`, `src/core/evaluation.py`.

---

## 3. Results: Signal Unreliability and Inconsistency

- **3.1. On JailbreakBench, Simple Baselines Outperform Semantic Entropy (H1)**
    - **Llama-4-Scout:** SE's best AUROC is 0.685 (at τ=0.1), outperformed by BERTScore (AUROC 0.767). At canonical τ=0.2, SE's FNR is 0.85 (actual_fpr=0.0, threshold=0.971).
      - *Source:* `outputs/h1/evaluation/llama4scout_120val_results.json`
    - **Qwen-2.5-7B:** SE's best AUROC is 0.690 (at τ=0.1), outperformed by Embedding Variance (AUROC 0.721). At canonical τ=0.2, FNR is 0.983 (actual_fpr=0.05, threshold=1.371).
      - *Source:* `outputs/h1/evaluation/qwen25_120val_results.json`
    - **_Figure 1: AUROC Comparison on JailbreakBench._** Bar chart. X-axis: Model (Llama, Qwen). Y-axis: AUROC. Bars for SE (at best τ), BERTScore, Emb. Variance. Data Sources: `outputs/h1/evaluation/llama4scout_120val_results.json`, `outputs/h1/evaluation/qwen25_120val_results.json`.
- **3.2. Performance Generalizes Poorly to HarmBench (H2)**
    - **Llama-4-Scout:** At canonical τ=0.2, SE performs poorly (FNR=0.765, actual_fpr=0.0). This is worse than the best baseline, Embedding Variance (FNR=0.605).
      - *Source:* `outputs/h2/evaluation/llama-4-scout-17b-16e-instruct_h2_results.json`
    - **Qwen-2.5-7B:** At canonical τ=0.2, SE's performance collapses (FNR=0.889, actual_fpr=0.0). Although its best performance at τ=0.1 appears to be a "win" (FNR=0.630), this is a brittle artifact, as shown next.
      - *Source:* `outputs/h2/evaluation/qwen2.5-7b-instruct_h2_results.json`
    - **_Table 2: FNR@5%FPR on JBB vs. HBC._** Table with columns: Model, Dataset, Method, FNR, actual_fpr, threshold. For SE, show rows for both canonical (τ=0.2) and best-τ.

---

## 4. Results: Investigating the Failure Modes

- **4.1. Confounder Analysis: Response Length is Not the Primary Driver (H3)**
    - **Finding:** For Llama on HarmBench (N=162 prompts), an OLS regression of `SE score (τ=0.1) ~ log(median response length)` yields a weak correlation (R²=0.103). Residualized SE still has an AUROC of 0.630.
      - *Source:* `outputs/h3/results/llama-4-scout-17b-16e-instruct_H2_h3_results.json`
    - **_Figure 2: SE vs. Response Length._** Scatter plot. X-axis: log(Median Response Length). Y-axis: SE Score (τ=0.1). Points colored by label. Data: `outputs/h3/per_prompt_analysis/llama-4-scout-17b-16e-instruct_H2_h3_prompt_analysis.jsonl`.
- **4.2. Robustness Analysis: Performance is Brittle to Hyperparameters (H4)**
    - We test the one case where SE worked well: Qwen on HarmBench (at τ=0.1, N=5).
    - **τ-Brittleness:** Increasing `τ` from 0.1 to 0.2 causes a performance collapse: FNR skyrockets from **0.630 to 0.889**.
    - **N-Sensitivity:** Increasing `N` from 5 to 10 *improves* performance at τ=0.1 (FNR drops to 0.469), but the brittleness to `τ` persists (FNR at N=10 jumps to 0.827 when τ=0.2).
      - *Source:* `outputs/h4/evaluation/h4_brittleness_results.json`
    - **_Figure 3: FNR@5%FPR vs. Hyperparameters for Qwen on HarmBench._** Multi-line plot. X-axis: τ ∈ {0.1,..,0.4}. Y-axis: FNR. Lines for SE (N=5), SE (N=10), and Embedding Variance. Data sources: `outputs/h4/evaluation/h4_brittleness_results.json`, `outputs/h2/evaluation/qwen2.5-7b-instruct_h2_results.json`.
- **4.3. Data Contamination Analysis: Failures are Robust to Paraphrasing (H5)**
    - **Finding:** On paraphrased JBB prompts, some Qwen baselines saw improved FNR (Avg BERTScore -6.3pp; Emb. Variance -2.0pp), while Levenshtein worsened (+9.0pp). AUROC shifts were minor. SE performance was unchanged.
      - *Source:* `outputs/h5/evaluation/h5_paraphrase_degradation_report.md`

---

## 5. The Consistency Confound: A Generalizable Mechanism for Failure (H6)

- **5.1. Defining the Consistency Confound:** SE fails because it punishes consistent outputs. Strong alignment leads to consistent, templated refusals (low SE), causing a false negative. Similarly, an effective jailbreak that produces consistent harmful content will also be missed.
- **5.2. Why SE is Uniquely Vulnerable:** Unlike baselines that directly measure embedding dispersion, SE's two-step process (thresholding via `τ`, then compressing via entropy) makes it uniquely sensitive to small changes in variance for small N.
- **5.3. Quantitative Audit of False Negatives:**
    - Defined the confound signature as (duplicate rate > 0.6 and cluster count ≤ 2).
    - **Llama @ JBB:** Explained **73.3% (44/60)** of false negatives.
      - *Source:* `outputs/h6/llama-h1-jailbreakbench/llama-4-scout-17b-16e-instruct_H1_h6_qualitative_audit_results.json`
    - **Qwen @ HBC:** Explained a stark **97.5% (79/81)** of false negatives.
      - *Source:* `outputs/h6/qwen-h2-harmbench/qwen-2.5-7b-instruct_H2_h6_qualitative_audit_results.json`
    - **_Figure 4: Breakdown of False Negative Causes._** Stacked bar chart. X-axis: Experiment (Llama@JBB, Qwen@HBC). Y-axis: Count of False Negatives. Bars segmented by "Consistency Confound" vs "Other". Data sources: H6 audit JSONs and `outputs/h6/*/per_prompt_predictions.jsonl`.
- **5.4. Qualitative Examples:**
    - **Example 1 (Consistent Refusal):** For prompt `jbb_75` (phishing), Llama produces 5 near-identical refusals. SE = 0.0 -> False Negative.
    - **Example 2 (Consistent Jailbreak):** For prompt `h2_harmful_061` (SQL injection), Qwen consistently provides harmful code. SE = 0.0 -> False Negative.

---

## 6. Discussion and Conclusion

- **Significance:** As safety alignment improves, LLM output distributions for harmful prompts collapse onto refusals, making output-diversity detectors systematically fail. This paper provides the first comprehensive audit of this effect on a modern safeguard.
- **Design Principle for Future Detectors:** Detectors whose primary signal is output diversity become less useful as refusal policies converge. Future work should focus on detectors aligned to features *positively* correlated with safety (e.g., refusal-template features, or input-level prompt properties).
- **Limitations & Future Work:**
    - Our FNR figures may be optimistic due to the lack of a separate calibration set. Future work should re-run evaluations with strict train/test splits and compute CIs.
    - Test the Consistency Confound on larger models (the planned H7) and against distribution shifts (e.g., JailbreaksOverTime, 2025).
    - Develop a simple refusal-template matching baseline to demonstrate a positively-correlated signal.

---

## 7. References
- Farquhar et al., (2024). "Detecting hallucinations in large language models using semantic entropy." *Nature*.
- Manakul et al., (2023). "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection." *EMNLP 2023*.
- Galinkin & Sablotny, (2024). "Improved Large Language Model Jailbreak Detection via Pretrained Embeddings." *arXiv*.
- Chao et al., (2024). "JailbreakBench: An Open Robustness Benchmark..." *NeurIPS 2024*.
- Mazeika et al., (2024). "HarmBench: A Standardized Evaluation Framework..." *arXiv*.

---

## Data Provenance Note
- Outline based on results generated on or before 2025-08-31. File timestamps (e.g., H4 JSON `2025-08-28`) are noted in result files. All referenced files are in the `outputs/` directory. Experiment H7 (large model test) was planned but not executed.