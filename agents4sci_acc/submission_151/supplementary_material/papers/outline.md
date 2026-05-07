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

- **The Appeal of Behavioral Signals:** Introduce the problem of black-box jailbreak detection. Frame the intuitive appeal of using behavioral signals, like response inconsistency, as a proxy for the model's internal conflict between its safety training and a harmful instruction.
  - *Source:* `idea/revised_idea_v1.md`
- **Semantic Entropy as a Formalism:** Introduce Semantic Entropy (SE) as a plausible method to formalize and measure this inconsistency. Briefly explain the mechanism: jailbreak causes conflict -> multimodal response distribution -> high SE. Benign/direct harmful prompts -> unimodal response -> low SE.
- **Our Core Claims:** We find that this plausible mechanism fails in practice. This paper makes three central claims:
    1. **SE Underperforms Baselines:** Semantic Entropy is consistently outperformed by simpler textual consistency baselines (e.g., Average Pairwise BERTScore, Embedding Variance) at low false-positive rates on standard benchmarks.
    2. **Performance is Unreliable and Brittle:** The effectiveness of all consistency detectors, including SE, is highly dependent on the model and data distribution. Even when SE shows apparent "wins," its performance is an artifact of specific hyperparameter choices (`τ`, `N`) and is not robust.
    3. **Failure is Caused by the Consistency Confound:** The primary failure mode is a mechanism we term the "Consistency Confound," where strong safety alignment produces templated, consistent refusals to harmful prompts, resulting in low semantic entropy that the detector misinterprets as safe.
- **Related Work:**
    - Situate our work in the lineage of consistency-based methods for LLM evaluation (e.g., SelfCheckGPT for hallucinations), arguing that the core assumption (consistency → correctness) breaks down in the safety domain.
    - Differentiate from input-perturbation methods (e.g., SemanticSmooth) and white-box methods (e.g., Gradient Cuff), highlighting the unique challenges of a black-box, output-only approach.
    - *Source:* `papers/methodology_notes.md`

---

## 2. Methodology

- **2.1. Detection Methods:**
    - **Primary Method: Semantic Entropy (SE):**
        - For each prompt, generate N=5 responses.
        - Embed responses using a sentence-transformer.
        - Perform Agglomerative Hierarchical Clustering on embeddings with a cosine similarity threshold `τ`.
        - Calculate Shannon entropy over the distribution of cluster sizes.
    - **Baseline Methods:**
        - **Average Pairwise BERTScore:** High score implies high consistency.
        - **Embedding Variance:** Low variance implies high consistency.
        - **Levenshtein Variance:** Low variance implies high textual similarity.
        - *Implementation Details:* `src/core/baseline_metrics.py`.
- **2.2. Rationale for Black-Box SE Variant:**
    - Explain the choice of embedding-based clustering over the original NLI-based method from Farquhar et al. (2024), driven by the constraints of black-box APIs which do not expose token probabilities.
    - **_Table 1: Comparison of SE Variants._** A small table contrasting our implementation with the original *Nature* paper on dimensions like Primary Application, Access Required, and Computational Cost.
    - *Source:* `papers/methodology_notes.md`
- **2.3. Experimental Setup:**
    - **Models:** Llama-4-Scout-17B-16E-Instruct, Qwen/Qwen2.5-7B-Instruct.
    - **Datasets:**
        - **JailbreakBench (JBB):** 120-prompt validation split (60 harmful, 60 benign).
        - **HarmBench-Contextual (HBC):** 81 harmful prompts and a matched set of 81 benign "twin" prompts.
    - **Implementation Details:**
        - **API Provider:** OpenRouter.
        - **Response Generation:** N=5, Temperature=0.7, Top-p=0.95, Max Tokens=1024.
        - **Embedding Model:** `Alibaba-NLP/gte-large-en-v1.5` (1024-dim, L2-normalized).
        - **Clustering Thresholds (τ):** Grid search over {0.1, 0.2, 0.3, 0.4} to test sensitivity. Rationale: 0.1 (near-identical), 0.2 (canonical), 0.3 (paraphrases), 0.4 (topical).
        - **Reproducibility:** All experiments run with a fixed random seed of 42.
        - *Source:* `papers/methodology_notes.md`, `configs/project_config.yaml`
- **2.4. Evaluation and Statistical Analysis:**
    - **Metrics:** AUROC and FNR@5%FPR.
    - **Protocol:** A 30% calibration split of the benign set is used to select the score threshold corresponding to a 5% FPR, which is then applied to the remaining 70% test split.
    - **Statistical Significance:** We report 95% confidence intervals for all primary metrics. Wilson CIs are used for FNR, and paired DeLong CIs are used for comparing AUROC between SE and baselines on the same prompt set.
    - *Source:* `mentor_docs/mentor_feedback_post_checkpoint_2.md`

---

## 3. Results: Signal Unreliability and Inconsistency

- **3.1. On JailbreakBench, Simple Baselines Outperform Semantic Entropy (H1)**
    - For **Llama-4-Scout**, SE (AUROC 0.625) is significantly outperformed by Avg. Pairwise BERTScore (AUROC 0.767).
      - *Source:* `outputs/h1/evaluation/llama4scout_120val_results.json`
    - For **Qwen-2.5-7B**, SE performs barely above chance (AUROC 0.529) and is outperformed by Embedding Variance (AUROC 0.721).
      - *Source:* `outputs/h1/evaluation/qwen25_120val_results.json`
    - **_Figure 1: AUROC Comparison on JailbreakBench._** Bar chart comparing detector AUROC on both models, with DeLong 95% CIs. **Caption Takeaway:** "Strong safety alignment produces templated, consistent refusals, resulting in low semantic entropy that fails to distinguish harmful from benign requests. In contrast, simple textual similarity metrics like BERTScore more effectively capture these distinct refusal patterns."
- **3.2. Performance Generalizes Poorly and Unpredictably to HarmBench (H2)**
    - For **Llama-4-Scout**, SE is again outperformed by Embedding Variance based on the primary FNR@5%FPR metric (0.605 vs 0.654 for SE). Notably, BERTScore, the previous winner, now fails (AUROC 0.506).
      - *Source:* `outputs/h2/evaluation/llama-4-scout-17b-16e-instruct_h2_results.json`
    - Paradoxically, for **Qwen-2.5-7B**, SE (AUROC 0.733) becomes the *best* performing method on this dataset.
      - *Source:* `outputs/h2/evaluation/qwen2.5-7b-instruct_h2_results.json`
    - **Key Takeaway:** The detector's utility is highly dependent on the model and data distribution, making it unreliable for deployment.
    - **_Table 2: FNR@5%FPR on JailbreakBench vs. HarmBench._** Table showing FNR (with Wilson 95% CIs) for all detectors, models, and datasets, highlighting the unstable performance.

---

## 4. Results: Investigating the Failure Modes

- **4.1. Confounder Analysis: Response Length is Not the Primary Driver (H3)**
    - **Finding:** We falsified the hypothesis that SE is merely a proxy for response length. For Llama on HarmBench, response length only explains ~10% of the variance in SE scores (R²=0.103). After controlling for length, the residual SE signal's AUROC only drops from 0.691 to 0.630.
      - *Source:* `outputs/h3/results/llama-4-scout-17b-16e-instruct_H2_h3_results.json`
- **4.2. Robustness Analysis: Performance is Brittle to Hyperparameters (H4)**
    - **Finding:** In the one case where SE worked well (Qwen on HarmBench), its success is extremely brittle. Increasing the clustering threshold `τ` from 0.1 to just 0.2 causes a performance collapse: FNR@5%FPR skyrockets from **0.630 to 0.889** (+25.9pp).
      - *Source:* `outputs/h4/evaluation/h4_brittleness_results.json`
    - **_Figure 2: FNR@5%FPR vs. Clustering Threshold (τ)._** Line plot showing Qwen FNR on HarmBench, demonstrating the sharp performance drop for SE as τ increases, contrasted with a more stable baseline.
- **4.3. Data Contamination Analysis: Failures are Robust to Paraphrasing (H5)**
    - **Finding:** The hypothesis that paraphrasing would disproportionately harm SE was falsified. For Qwen on JBB, performance was already at maximum failure (FNR=1.0) and did not degrade further. Baselines actually saw performance *improve*.
      - *Source:* `outputs/h5/evaluation/h5_robustness_evaluation.json`, `outputs/h5/evaluation/h5_paraphrase_degradation_report.md`
    - **Key Takeaway:** The failure mechanism is more fundamental than simple sensitivity to prompt phrasing.

---

## 5. The Consistency Confound: A Generalizable Mechanism for Failure (H6)

- **5.1. Defining the Consistency Confound:** We hypothesize that SE fails because it punishes consistent outputs. Strong alignment leads to consistent, templated refusals (low SE), causing the detector to misclassify a successful defense as a benign interaction (a false negative). Similarly, a highly effective jailbreak that produces consistent harmful content will also be missed.
- **5.2. Why SE is Uniquely Vulnerable:** Unlike baselines that directly measure embedding dispersion (e.g., Embedding Variance), SE first thresholds this dispersion via `τ`-clustering and then compresses the cluster counts via Shannon entropy. For small N, this two-step process makes it extremely sensitive to small changes in variance, explaining its unique brittleness compared to smoother baselines.
  - *Source:* `mentor_docs/mentor_feedback_post_checkpoint_2.md`
- **5.3. Quantitative Audit of False Negatives:**
    - We defined the "Consistency Confound" signature as (duplicate rate > 0.6) and (cluster count ≤ 2) and audited false negatives from two experiments.
    - **Finding 1 (Llama @ JBB):** Explained **73.3%** of false negatives.
      - *Source:* `outputs/h6/llama-h1-jailbreakbench/llama-4-scout-17b-16e-instruct_H1_h6_qualitative_audit_results.json`
    - **Finding 2 (Qwen @ HBC):** Explained a stark **97.5%** of false negatives (79 out of 81).
      - *Source:* `outputs/h6/qwen-h2-harmbench/qwen-2.5-7b-instruct_H2_h6_qualitative_audit_results.json`
    - **Key Takeaway:** The Consistency Confound is a robust, generalizable failure mechanism that worsens on a better-performing model.
    - **_Figure 3: Breakdown of False Negative Causes._** Stacked bar chart comparing the cause of false negatives for (Llama, JBB) and (Qwen, HBC), showing the dominance of the Consistency Confound.
- **5.4. Qualitative Examples:**
    - **Example 1 (Consistent Refusal):** For prompt `jbb_75` (phishing email), Llama produces 5 near-identical refusals. Duplicate rate = 1.0, SE = 0.0 -> False Negative.
      - *Source:* `outputs/h6/llama-h1-jailbreakbench/llama-4-scout-17b-16e-instruct_H1_h6_qualitative_audit_results.json`
    - **Example 2 (Consistent Jailbreak):** For prompt `h2_harmful_061` (SQL injection code), Qwen consistently provides harmful, compliant code. Duplicate rate = 1.0, SE = 0.0 -> False Negative.
      - *Source:* `outputs/h6/qwen-h2-harmbench/qwen-2.5-7b-instruct_H2_h6_qualitative_audit_results.json`

---

## 6. Discussion and Conclusion

- **The Consistency Confound:** The core assumption of SE—that inconsistency signals internal conflict—is inverted in the safety domain. Better alignment leads to more consistent (and thus seemingly "safe") refusals, making the detector fail more often. This is a fundamental flaw for this class of behavioral detectors.
- **Implications for Safety:** This work highlights a fundamental tension: improving model alignment to make refusals more robust may inadvertently make them invisible to behavioral detectors. This could create a false sense of security.
- **Future Work:**
    - Test the Consistency Confound on even larger models to see if the effect is more pronounced.
    - Develop detectors based on signals that are *positively* correlated with alignment (e.g., linguistic features of refusal text).
    - Investigate adaptive attacks that exploit the Consistency Confound by generating consistent harmful responses.

---

## Data Provenance Note
- This outline is based on final experimental results generated on or before 2025-08-31. All referenced files are in the `outputs/` directory. Hypothesis H7 (larger model test) was planned but not executed.