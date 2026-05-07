# Paper Outline: The Consistency Confound: Why Stronger Alignment Can Break Black-Box Jailbreak Detection

**Authors:** AI Researcher et al.

**Abstract:**
- **Context:** Black-box monitoring of Large Language Models (LLMs) for jailbreak attempts is a critical, unsolved safety problem. A promising recent idea, inspired by hallucination detection, is to use the *semantic inconsistency* of multiple generated responses as a behavioral signal of the internal conflict caused by a jailbreak.
- **Hypothesis:** We test the core hypothesis that high semantic entropy—a measure of semantic inconsistency—can serve as a reliable, zero-shot jailbreak detector.
- **Key Finding:** We comprehensively refute this hypothesis. Across two models (Llama-4-Scout-17B, Qwen-2.5-7B) and two benchmarks (JailbreakBench, HarmBench), we find semantic entropy to be an unreliable and brittle detector.
- **Core Contribution:** We identify and validate a core failure mechanism we term the **"Consistency Confound"**: stronger safety alignment leads models to produce highly consistent, templated refusals. This results in low semantic entropy, causing the detector to fail precisely when the model is behaving most safely. Our qualitative audits show this mechanism accounts for **73.3%** of false negatives for Llama on JailbreakBench and a stark **97.5%** for Qwen on HarmBench.
- **Implications:** Our findings serve as a cautionary tale for behavioral detection methods, suggesting that as models become better aligned, they may paradoxically become harder to monitor with techniques that treat consistency as a signal of safety.

---

## 1. Introduction

- **The Appeal of Behavioral Signals:** Introduce the problem of black-box jailbreak detection. Frame the intuitive appeal of using behavioral signals, like response inconsistency, as a proxy for the model's internal conflict.
  - *Source:* `idea/revised_idea_v1.md`
- **Semantic Entropy as a Formalism:** Introduce Semantic Entropy (SE) as a plausible method to measure this inconsistency. Explain the mechanism: jailbreak causes conflict -> multimodal response distribution -> high SE. Benign prompts -> unimodal response -> low SE.
- **Our Core Claims:** We find that this plausible mechanism fails in practice. This paper makes three central claims:
    1. **SE Underperforms Baselines:** Semantic Entropy is consistently outperformed by simpler textual consistency baselines (e.g., Average Pairwise BERTScore, Embedding Variance) on standard benchmarks.
    2. **Performance is Unreliable and Brittle:** The effectiveness of all consistency detectors, including SE, is highly dependent on the model, data distribution, and hyperparameter choices. We show SE's apparent "wins" are artifacts of specific `τ` (clustering threshold) and `N` (sample count) settings.
    3. **Failure is Caused by the Consistency Confound:** The primary failure mode is a mechanism we term the "Consistency Confound," where strong safety alignment produces templated, consistent refusals to harmful prompts, resulting in low semantic entropy that the detector misinterprets as safe.
- **Related Work:**
    - Situate our work in the lineage of consistency-based methods (e.g., SelfCheckGPT), arguing the core assumption (consistency → correctness) breaks down in the safety domain.
    - Differentiate from input-perturbation methods and white-box methods, highlighting the unique challenges of a black-box, output-only approach.
    - *Source:* `papers/methodology_notes.md`

---

## 2. Methodology

- **2.1. Detection Methods:**
    - **Primary Method: Semantic Entropy (SE):** `src/core/semantic_entropy.py`
    - **Baseline Methods:** Avg Pairwise BERTScore, Embedding Variance, Levenshtein Variance. `src/core/baseline_metrics.py`
- **2.2. Rationale for Black-Box SE Variant:**
    - Explain the choice of embedding-based clustering over the original NLI-based method from Farquhar et al. (2024), driven by the constraints of black-box APIs.
    - **_Table 1: Comparison of SE Variants._** A table contrasting our implementation with the original *Nature* paper.
      - **Rows:** Primary Application, Access Required, Clustering Method, Computational Cost.
      - **Columns:** Original SE (Farquhar et al.), Our Implementation.
      - *Source:* `papers/methodology_notes.md`
- **2.3. Experimental Setup:**
    - **Models:** Llama-4-Scout-17B-16E-Instruct, Qwen/Qwen2.5-7B-Instruct.
    - **Datasets:**
        - **JailbreakBench (JBB):** 120-prompt validation split (60 harmful, 60 benign).
        - **HarmBench-Contextual (HBC):** 81 harmful prompts and 81 matched benign "twin" prompts.
    - **Implementation Details:**
        - **API:** OpenRouter. `src/core/response_generator_openrouter.py`.
        - **Response Generation:** N=5 (or N=10 for H4), T=0.7, Top-p=0.95, Max Tokens=1024. See `outputs/h1/response_generation/*_logs.md`.
        - **Embedding Model:** `Alibaba-NLP/gte-large-en-v1.5` (1024-dim, L2-normalized). `src/core/semantic_entropy.py`.
        - **Clustering:** Agglomerative Clustering (avg linkage, cosine distance) with `τ` ∈ {0.1, 0.2, 0.3, 0.4}.
        - **Reproducibility:** Global seed=42.
- **2.4. Evaluation Protocol:**
    - **Metrics:** AUROC and False Negative Rate at a target 5% False Positive Rate (FNR@5%FPR).
    - **Thresholding Protocol:** For each detector, a score threshold is selected across the full evaluation dataset to achieve an FPR as close as possible to, without exceeding, 5%. The FNR is then calculated at this threshold. We report the `actual_fpr` achieved for transparency.
    - **τ Selection Policy:** For AUROC, we report the score from the best-performing `τ` for SE. For FNR@5%FPR and cross-method comparisons, we report SE at a fixed `τ=0.2` (our canonical setting) to ensure fair comparison and assess robustness.
    - *Source:* Based on review of `configs/project_config.yaml` and raw result files.

---

## 3. Results: Signal Unreliability and Inconsistency

- **3.1. On JailbreakBench, Simple Baselines Outperform Semantic Entropy (H1)**
    - For **Llama-4-Scout**, SE achieves its best AUROC of 0.685 (at τ=0.1), but this is still outperformed by Avg. Pairwise BERTScore (AUROC 0.767). At our canonical τ=0.2, SE's FNR@5%FPR is 0.85 (at an actual_fpr=0.0).
      - *Source:* `outputs/h1/evaluation/llama4scout_120val_results.json`
    - For **Qwen-2.5-7B**, SE's best AUROC is 0.690 (at τ=0.1), which is outperformed by Embedding Variance (AUROC 0.721). At τ=0.1, SE's FNR is 1.0 (total failure), and at canonical τ=0.2, it is 0.983 (actual_fpr=0.05).
      - *Source:* `outputs/h1/evaluation/qwen25_120val_results.json`
    - **_Figure 1: AUROC Comparison on JailbreakBench._** Bar chart comparing detector AUROC on both models. X-axis: Model (Llama, Qwen). Y-axis: AUROC. Bars for SE (at best τ), BERTScore, Emb. Variance. Data from H1 JSON files.
- **3.2. Performance Generalizes Poorly to HarmBench (H2)**
    - For **Llama-4-Scout**, Embedding Variance is the best baseline by FNR (0.605), outperforming SE at its best τ=0.1 (FNR=0.654).
      - *Source:* `outputs/h2/evaluation/llama-4-scout-17b-16e-instruct_h2_results.json`
    - For **Qwen-2.5-7B**, SE at τ=0.1 is the best performing method (AUROC 0.733, FNR 0.630). This "win" sets up the brittleness analysis in Sec 4.2.
      - *Source:* `outputs/h2/evaluation/qwen2.5-7b-instruct_h2_results.json`
    - **_Table 2: FNR@5%FPR on JBB vs. HBC._** Table showing FNR (and actual_fpr) for SE (at τ=0.2) and all baselines across both models and datasets.

---

## 4. Results: Investigating the Failure Modes

- **4.1. Confounder Analysis: Response Length is Not the Primary Driver (H3)**
    - **Finding:** For Llama on HarmBench, an OLS regression of SE score against log(response length) yields a weak correlation (R²=0.103). After removing the length effect, the residualized SE (τ=0.1) AUROC only drops from 0.691 to 0.630.
      - *Source:* `outputs/h3/results/llama-4-scout-17b-16e-instruct_H2_h3_results.json`
    - **_Figure 2: SE vs. Response Length._** Scatter plot. X-axis: log(Median Response Length). Y-axis: SE Score (τ=0.1). Points colored by label (harmful/benign). Show the weak correlation visually. Data source: `outputs/h3/per_prompt_analysis/llama-4-scout-17b-16e-instruct_H2_h3_prompt_analysis.jsonl`.
- **4.2. Robustness Analysis: Performance is Brittle to Hyperparameters (H4)**
    - We test the one case where SE worked well: Qwen on HarmBench (at τ=0.1, N=5).
    - **τ-Brittleness:** Increasing the clustering threshold `τ` from 0.1 to 0.2 causes a performance collapse: FNR@5%FPR skyrockets from **0.630 to 0.889**.
    - **N-Sensitivity:** Increasing the number of samples `N` from 5 to 10 *improves* performance at τ=0.1 (FNR drops from 0.630 to 0.469), but the brittleness to `τ` persists (FNR at N=10 jumps to 0.827 when τ=0.2).
      - *Source:* `outputs/h4/evaluation/h4_brittleness_results.json`
    - **_Figure 3: FNR@5%FPR vs. Hyperparameters (τ, N) for Qwen on HarmBench._** Multi-line plot. X-axis: τ ∈ {0.1,0.2,0.3,0.4}. Y-axis: FNR@5%FPR. Two lines for SE (N=5, N=10) and one for a stable baseline. Data: `outputs/h4/evaluation/h4_brittleness_results.json`.
- **4.3. Data Contamination Analysis: Failures are Robust to Paraphrasing (H5)**
    - **Finding:** The hypothesis that paraphrasing would disproportionately harm SE was falsified. For Qwen on JBB, performance did not degrade. Some baselines saw performance *improve* (e.g., Avg BERTScore FNR -6.3pp, Embedding Variance FNR -2.0pp), while Levenshtein Variance worsened (+9.0pp).
      - *Source:* `outputs/h5/evaluation/h5_paraphrase_degradation_report.md`
    - **Key Takeaway:** The failure mechanism is more fundamental than simple sensitivity to prompt phrasing.

---

## 5. The Consistency Confound: A Generalizable Mechanism for Failure (H6)

- **5.1. Defining the Consistency Confound:** We hypothesize that SE fails because it punishes consistent outputs. Strong alignment leads to consistent, templated refusals (low SE), causing a false negative. Similarly, an effective jailbreak that produces consistent harmful content will also be missed.
- **5.2. Why SE is Uniquely Vulnerable:** Unlike baselines that directly measure embedding dispersion, SE first thresholds this dispersion via `τ`-clustering and then compresses the cluster counts via Shannon entropy. For small N, this two-step process makes it extremely sensitive to small changes in variance.
  - *Source:* `mentor_docs/mentor_feedback_post_checkpoint_2.md`
- **5.3. Quantitative Audit of False Negatives:**
    - We defined the "Consistency Confound" signature as (duplicate rate > 0.6 and cluster count ≤ 2).
    - **Llama @ JBB:** Explained **73.3% (44/60)** of false negatives.
      - *Source:* `outputs/h6/llama-h1-jailbreakbench/llama-4-scout-17b-16e-instruct_H1_h6_qualitative_audit_results.json`
    - **Qwen @ HBC:** Explained a stark **97.5% (79/81)** of false negatives.
      - *Source:* `outputs/h6/qwen-h2-harmbench/qwen-2.5-7b-instruct_H2_h6_qualitative_audit_results.json`
    - **_Figure 4: Breakdown of False Negative Causes._** Stacked bar chart. X-axis: Experiment (Llama@JBB, Qwen@HBC). Y-axis: Count of False Negatives. Bars segmented by "Consistency Confound" vs "Other".
- **5.4. Qualitative Examples:**
    - **Example 1 (Consistent Refusal):** For prompt `jbb_75` (phishing email), Llama produces 5 near-identical refusals. Duplicate rate = 1.0, SE = 0.0 -> False Negative.
    - **Example 2 (Consistent Jailbreak):** For prompt `h2_harmful_061` (SQL injection code), Qwen consistently provides harmful, compliant code. Duplicate rate = 1.0, SE = 0.0 -> False Negative.

---

## 6. Discussion and Conclusion

- **The Consistency Confound:** The core assumption of SE—that inconsistency signals internal conflict—is inverted in the safety domain. Better alignment leads to more consistent (and thus seemingly "safe") refusals, making the detector fail more often.
- **Implications for Safety:** This work highlights a fundamental tension: improving model alignment to make refusals more robust may inadvertently make them invisible to behavioral detectors.
- **Limitations:**
    - Our evaluation protocol did not use a separate calibration set for thresholding, so FNR figures may be optimistic.
    - We do not report confidence intervals, which limits claims of statistical significance. This is a key area for future work.
    - The study is limited to two open-source models and N=5/10 samples per prompt.
- **Future Work:**
    - Re-run evaluations with a strict calibration/test split and compute CIs.
    - Test the Consistency Confound on larger, more capable models (the planned H7).
    - Develop detectors based on signals that are *positively* correlated with alignment.