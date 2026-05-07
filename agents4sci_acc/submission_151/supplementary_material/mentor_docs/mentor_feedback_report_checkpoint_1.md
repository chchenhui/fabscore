```
# 📜 Mentor Brief (Final) — for the Autonomous AI Scientist

**Subject:** Decide next step + execute the SE-failure replication plan
**Date:** Aug 21, 2025
**From:** Research Mentor
**To:** Autonomous AI/ML Scientist

---

## TL;DR (what you should choose in the 4-way gate)

**👉 Recommendation:** **2) Create new hypotheses suite.**
We keep **H1** (already run on **Qwen-2.5-7B-Instruct** and **Llama-4-Scout-17B**) as canonical. Now formalize **H2–H5** to *map* the failure of Semantic Entropy (SE) across datasets, paraphrases, and settings (no hyperparameter tuning), with an **optional** H6 for closed-API aggregate. The story is a rigorous, contamination-aware *negative result* that aligns with how the safety community currently evaluates jailbreak robustness (JailbreakBench, HarmBench, refusal/over-refusal framing). ([arXiv][1], [proceedings.neurips.cc][2])

---

## Working title (use now)

**The Consistency Confound: Semantic Entropy Fails for Jailbreak Detection Across Models and Benchmarks**

*(We can rename later if we add a small “what works” appendix.)*

---

## What we’ve learned (H1 stands)

* **Finding:** On **JBB-120**, **SE underperforms** simple, cheap consistency baselines (e.g., Avg pairwise BERTScore) for **both** Qwen-2.5-7B-Instruct and Llama-4-Scout-17B.
* **Metric of record:** **FNR\@5% FPR**, where the threshold is chosen so **FPR ≤ 0.05**.
* **Mechanism (Consistency Confound):** Safety alignment teaches **stable/templated refusals** to harmful prompts; multi-sample outputs become near-duplicates; **SE ≈ 0** even when the prompt is harmful. Note: SE’s “consistency ⇒ correctness” assumption comes from *hallucination detection*, not safety refusals. ([ACL Anthology][3], [Nature][4])

---

## Create two short docs now (to anchor you)

1. **`papers/outline.md`**

   * One-sentence thesis: *SE (designed for hallucinations) fails for jailbreak detection due to alignment-induced refusal consistency.*
   * Three claims for page-1:

     1. SE underperforms simple baselines at low FPR on JBB.
     2. Failure replicates on HarmBench contextual with matched benigns.
     3. Pattern holds for Qwen-2.5-7B-Instruct & Llama-4-Scout; optional closed-API aggregate.
   * **Figure 1:** ROC at identical N/decoding (**SE vs Avg-BERTScore**); caption takeaway: *alignment → templated refusals → low SE on harmful*.
   * Methods blurb: *Discrete SE (embedding-clustered), no token log-probs, no LLM-entailment clustering (black-box).* Cite SE’s original scope in hallucinations. ([Nature][4])
   * Evaluation blurb: JBB + HarmBench, low-FPR, contamination notes, refusal/over-refusal framing. ([arXiv][1], [proceedings.neurips.cc][2])

2. **`papers/methodology_notes.md`**

   * Record the exact **SE variant** we use: *Discrete SE via embedding clustering; τ grid reported; no log-probs; no entailment-based clustering; black-box constraint.*
   * Contrast briefly with the Nature estimator context (hallucinations). Link and cite.
     Links:
   * Nature SE: [https://www.nature.com/articles/s41586-024-07421-0](https://www.nature.com/articles/s41586-024-07421-0) ([Nature][4])
   * SelfCheckGPT: [https://aclanthology.org/2023.emnlp-main.557/](https://aclanthology.org/2023.emnlp-main.557/) ([ACL Anthology][3])

---

## Hypotheses to **rewrite** (H2–H5) — **lock models to Qwen-2.5-7B-Instruct & Llama-4-Scout-17B**

> Maintain your H1 thresholds and decoding constants for comparability. **No hyperparameter tuning.** Report **across** τ values; do not pick a best.

### **H2 — Cross-dataset replication on HarmBench-Contextual**

**Claim:** On **HarmBench-Contextual** (positives) vs **HarmBench-Benign-Matched** (negatives), **SE underperforms** Avg-BERTScore at the same decoding budget for **both** target models.
**Metrics:** AUROC; **FNR\@5% FPR**.
**Protocol:** Build **HarmBench-Benign-Matched** (topic & ±20% length via embeddings; 30% **calibration** to set FPR ≤ 0.05, 70% **test** to report FNR). ([arXiv][5])

### **H3 — Paraphrase robustness on JBB (post-dated)**

**Claim:** On **JBB-Paraphrase-2025-08** (harmful **and** benign paraphrases generated now), **SE separation degrades further** vs the original JBB-120 for both models; Avg-BERTScore degrades less.
**Metrics:** ΔAUROC and ΔFNR\@5% (original → paraphrased).
**Rationale:** Paraphrases reduce contamination and stress refusal-template stability. ([arXiv][1])

### **H4 — Sensitivity (reporting-only; no tuning)**

**Claim:** The SE failure holds **across reported settings**, not an artifact of τ/N/T.
**Report grid (descriptive):**

* **τ** ∈ {0.1, 0.2, 0.3, 0.4} *(exactly your current four; you may add 0.05 & 0.6 as edge rows if trivial)*
* **N** = 5 *(canonical)*; optionally add N = 10 *(one robustness row)*
* **T** = 0.7 *(canonical)*; optionally add T = 0.3 *(one robustness row)*
  **Policy:** For each (τ, N, T), **calibrate on a held-out calibration split** to FPR ≤ 0.05; **report AUROC & FNR** on a disjoint **test** split. **Do not** select a best.

### **H5 — Qualitative exemplars (mechanism made visible)**

**Claim:** Real outputs illustrate the **Consistency Confound**:

* Harmful + templated refusals ⇒ **SE ≈ 0** across N samples;
* Stylistic refusal variants inflate SE without compliance;
* Benign-but-long prompts: contrasting behavior.
  Include anonymized snippets/hashes as licensing allows.

---

## **Optional** H6 — Closed-API aggregate (Claude, OpenAI)

**Claim:** The SE-failure pattern **persists** (directionally) on popular **closed-source** models at the same low-FPR operating point.
**Metrics:** AUROC if you can build matched benigns via the same procedure; otherwise **TPR\@5% FPR** using a **fixed threshold** calibrated on JBB-120.
**Notes:** Report **aggregate** metrics only (no copyrighted verbatim content). This strengthens external validity without shifting the core story.

---

## Datasets, tools, and evaluation norms (links & why they matter)

* **JailbreakBench (JBB)** — behaviors, judges, over-refusal split; standard for jailbreak evaluation.
  PDF: [https://proceedings.neurips.cc/paper\_files/paper/2024/file/63092d79154adebd7305dfd498cbff70-Paper-Datasets\_and\_Benchmarks\_Track.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/63092d79154adebd7305dfd498cbff70-Paper-Datasets_and_Benchmarks_Track.pdf); arXiv: [https://arxiv.org/abs/2404.01318](https://arxiv.org/abs/2404.01318) ([proceedings.neurips.cc][6], [arXiv][1])
* **HarmBench** — standardized framework for automated red teaming & robust refusal; we’ll use the **Contextual** harmful split.
  [https://arxiv.org/abs/2402.04249](https://arxiv.org/abs/2402.04249) ([arXiv][5])
* **JailbreakEval toolkit** — pragmatic utilities for end-to-end evaluation.
  [https://arxiv.org/abs/2406.09321](https://arxiv.org/abs/2406.09321) ([PubMed][7])
* **WildGuard** — moderation tool with *prompt harm*, *response harm*, and **refusal rate** tasks; good source of benign prompts and framing.
  NeurIPS page: [https://neurips.cc/virtual/2024/poster/97764](https://neurips.cc/virtual/2024/poster/97764); abstract: [https://proceedings.neurips.cc/paper\_files/paper/2024/hash/0f69b4b96a46f284b726fbd70f74fb3b-Abstract-Datasets\_and\_Benchmarks\_Track.html](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f69b4b96a46f284b726fbd70f74fb3b-Abstract-Datasets_and_Benchmarks_Track.html) ([neurips.cc][8], [proceedings.neurips.cc][2])
* **Llama-Guard-3** model cards — modern moderation baseline to cite for refusal framing.
  [https://huggingface.co/meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) ([Hugging Face][9])
* **Over-refusal context** (for discussion & matched-benign rationale):
  **SORRY-Bench** (ICLR 2025): [https://openreview.net/forum?id=YfKNaRktan](https://openreview.net/forum?id=YfKNaRktan); proceedings page: [https://proceedings.iclr.cc/paper\_files/paper/2025/hash/9622163c87b67fd5a4a0ec3247cf356e-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9622163c87b67fd5a4a0ec3247cf356e-Abstract-Conference.html) ([openreview.net][10], [proceedings.iclr.cc][11])
  **OR-Bench** (2024): [https://arxiv.org/abs/2405.20947](https://arxiv.org/abs/2405.20947); OpenReview: [https://openreview.net/forum?id=obYVdcMMIT](https://openreview.net/forum?id=obYVdcMMIT) ([arXiv][12], [openreview.net][13])
* **Contamination-limited practice** (supports your *post-dated paraphrases*):
  **LiveBench** site: [https://livebench.ai/](https://livebench.ai/); arXiv: [https://arxiv.org/abs/2406.19314](https://arxiv.org/abs/2406.19314); OpenReview: [https://openreview.net/forum?id=sKYHBTAxVa](https://openreview.net/forum?id=sKYHBTAxVa) ([livebench.ai][14], [arXiv][15], [openreview.net][16])

*(Optional, for a small appendix on procedural/adaptive stress):*

* **TAP — Tree of Attacks (black-box, automated)**: [https://arxiv.org/abs/2312.02119](https://arxiv.org/abs/2312.02119) ([arXiv][17])
* **Many-shot Jailbreaking** (long-context power laws): [https://proceedings.neurips.cc/paper\_files/paper/2024/file/ea456e232efb72d261715e33ce25f208-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea456e232efb72d261715e33ce25f208-Paper-Conference.pdf); OpenReview: [https://openreview.net/forum?id=cw5mgd71jW](https://openreview.net/forum?id=cw5mgd71jW) ([proceedings.neurips.cc][18], [openreview.net][19])

---

## How to compute AUROC for HarmBench (you currently have only harmfuls)

**Build a defensible matched benign set (“HarmBench-Benign-Matched”):**

1. **Benign pool:** Start from **WildGuardTest benign** + trusted curated safe prompts. (WildGuard includes refusal rate and benign pools.) ([proceedings.neurips.cc][2])
2. **Topic & length match:** For each contextual harmful prompt, retrieve benign candidates via **embedding similarity** (same embedder used in SE clustering); from top-k candidates, pick one within **±20% length** (relax to ±30% if needed).
3. **Similarity gate:** cosine **≥ 0.80**; if none passes, benign-paraphrase the nearest candidate and re-check.
4. **Spot-check:** Manually verify \~10% that negatives are truly benign.
5. **Split:** 30% **calibration** / 70% **test**; **calibrate the threshold** to FPR ≤ 0.05 on calibration; **report FNR** on test.
6. **Report residuals:** length & similarity distributions; per-length-bin AUROC and a micro-average to prove fairness.

> If time-boxed: calibrate the 5%-FPR threshold on **JBB** and report **TPR\@5% FPR** on HarmBench positives as a transfer result (gives an operating-point generalization, though not AUROC). ([arXiv][1])

---

## Metrics you’ll report (quick refresher)

* **AUROC** — threshold-free ranking quality of the detector.
* **FNR\@5% FPR** — false-negative rate at an operating point with **FPR ≤ 0.05** (calibrate on **calibration split**, evaluate on **test**).
* **EER (optional)** — error rate at threshold where **FPR = FNR**; a compact single number (not the headline safety metric).
* **Cost/Query (optional)** — practical overhead (tokens × **N** and latency per prompt). This underscores that cheap baselines already beat SE without extra cost.

---

## Sensitivity policy (your “no hyperparameter tuning” rule)

* **Do not tune** τ/N/T to pick a best.
* **Do report** tables **across τ** (your {0.1, 0.2, 0.3, 0.4}; optionally add 0.05, 0.6 as edges), **N = 5** (optionally 10), **T = 0.7** (optionally 0.3).
* For each row: **calibrate on calibration split** (FPR ≤ 0.05), **report FNR on test**, and include AUROC.
* The “**all τ are bad**” result is itself a key insight; keep it visible in tables and captions.

---

## Qualitative section (include a short “Exemplars” subsection)

* **Harmful + canned refusal (SE≈0):** N near-duplicate refusals; SE collapses; benign appears indistinguishable by SE.
* **Stylistic refusal noise:** generic vs. topic-specific refusal styles spuriously inflate SE without actual compliance.
* **Benign-but-hard:** long safe queries; contrast SE with simple baselines.
* **Paraphrase stability:** post-dated paraphrases stay low-SE for harmful prompts.

Use anonymized snippets or response hashes if licensing restricts verbatim content.

---

## Repo & config hygiene (non-prescriptive, minimal friction)

* Keep **all H1 artifacts immutable** (data, logs, figures).
* Centralize **run configs** (decoding constants; dataset splits with timestamps; model IDs and whether API/local).
* Provide **CLI entrypoints per hypothesis** (e.g., `run_se_h2`, `run_baselines_h2`) that

  * read configs,
  * emit a **CSV** with: dataset/split, model, method (SE/Avg-BERTScore/…), τ, N, T, **calibrated threshold**, **FPR on calibration**, **FNR on test**, **AUROC** (with CI if possible), and run seed,
  * save a small **calibration report** (operating point details),
  * dump a **matching/contamination report** (length & similarity distributions; near-dup stats).
* One script per **figure/table** (caption lists the exact command that produced it).
* Store **qual exemplars**: anonymized prompt IDs + response hashes used in the paper.

---

## Reading refresh (short, focused)

* **SE & “consistency ⇒ correctness” comes from hallucinations:**
  *SelfCheckGPT* (EMNLP 2023): [https://aclanthology.org/2023.emnlp-main.557/](https://aclanthology.org/2023.emnlp-main.557/);
  *Semantic Entropy* (Nature 2024): [https://www.nature.com/articles/s41586-024-07421-0](https://www.nature.com/articles/s41586-024-07421-0) ([ACL Anthology][3], [Nature][4])
* **Safety eval norms / datasets:**
  **JailbreakBench**: [https://arxiv.org/abs/2404.01318](https://arxiv.org/abs/2404.01318) (camera-ready NeurIPS track) ([arXiv][1])
  **HarmBench**: [https://arxiv.org/abs/2402.04249](https://arxiv.org/abs/2402.04249) ([arXiv][5])
  **JailbreakEval**: [https://arxiv.org/abs/2406.09321](https://arxiv.org/abs/2406.09321) ([PubMed][7])
* **Refusal / moderation context:**
  **WildGuard**: NeurIPS page [https://neurips.cc/virtual/2024/poster/97764](https://neurips.cc/virtual/2024/poster/97764); abstract [https://proceedings.neurips.cc/paper\_files/paper/2024/hash/0f69b4b96a46f284b726fbd70f74fb3b-Abstract-Datasets\_and\_Benchmarks\_Track.html](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f69b4b96a46f284b726fbd70f74fb3b-Abstract-Datasets_and_Benchmarks_Track.html) ([neurips.cc][8], [proceedings.neurips.cc][2])
  **Llama-Guard-3**: [https://huggingface.co/meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) ([Hugging Face][9])
  **SORRY-Bench**: [https://openreview.net/forum?id=YfKNaRktan](https://openreview.net/forum?id=YfKNaRktan) (ICLR 2025) ([openreview.net][10])
  **OR-Bench**: [https://arxiv.org/abs/2405.20947](https://arxiv.org/abs/2405.20947) ([arXiv][12])
* **Contamination-limited practice:**
  **LiveBench**: [https://livebench.ai/](https://livebench.ai/); arXiv: [https://arxiv.org/abs/2406.19314](https://arxiv.org/abs/2406.19314) ([livebench.ai][14], [arXiv][15])
* **(Optional) Procedural/adaptive stress:**
  **TAP**: [https://arxiv.org/abs/2312.02119](https://arxiv.org/abs/2312.02119);
  **Many-shot Jailbreaking**: [https://proceedings.neurips.cc/paper\_files/paper/2024/file/ea456e232efb72d261715e33ce25f208-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea456e232efb72d261715e33ce25f208-Paper-Conference.pdf) ([arXiv][17], [proceedings.neurips.cc][18])

---

## What to run next (order of execution)

1. **Create** `papers/outline.md` and `papers/methodology_notes.md` (content above).
2. **Rewrite** the hypotheses file with **H2–H5** exactly as specified (models fixed to **Qwen-2.5-7B-Instruct** & **Llama-4-Scout-17B**), plus **optional H6** (closed-API aggregate).
3. **Draft** the updated experimental plan: datasets to build (**HarmBench-Benign-Matched**; **JBB-Paraphrase-2025-08**), calibration/test splits, decoding constants, metrics to log, and qualitative exemplars to collect.
4. **Build** HarmBench-Benign-Matched (topic & ±20% length matching; similarity gate ≥ 0.80; 10% spot-check).
5. **Run H2** (HarmBench) and **H3** (JBB-Paraphrase) on **both** models.
6. **Run H4** (sensitivity reporting tables) — **no tuning**, just τ/N/T rows.
7. **(Optional) Run H6** (closed-API aggregate).
8. **Assemble** 3–4 qualitative exemplars (anonymized) and add to Results.
9. **Emit** CSV logs + calibration reports + contamination/matching reports; generate Figure 1 via a single script (include the command in the caption).

---

## Why this meets our research guidelines

* **Community impact:** Addresses a live uncertainty: *why* SE (good for hallucinations) **fails** for jailbreak detection—clarifying evaluation assumptions used by JBB/HarmBench and moderation toolchains. ([arXiv][1])
* **Generalization path:** We replicate across **two model families**, **two datasets**, and **post-dated paraphrases**; optional **closed-API** confirmation.
* **Impact-complexity tradeoff:** Simple baselines (Avg-BERTScore) already outperform SE at equal budget; we’re not proposing heavy machinery to eke out marginal gains.
* **Method-problem alignment:** SE’s assumption is mismatched to refusal dynamics; we show this empirically and qualitatively. ([ACL Anthology][3])
* **Failure-mode planning:** If matched-benign construction introduces bias, we report per-length-bin AUROC & micro-averages, with distributions and spot-check notes.
* **Rigor & skepticism:** Strict calibration/test separation; **FNR\@5% FPR** focus; **no tuning**; contamination notes; post-dated paraphrases (LiveBench philosophy). ([arXiv][15])
* **Autonomous-agent tractability:** All experiments are black-box friendly; standard HF stacks; single-GPU feasible; small, config-driven scripts.
* **Novelty verification:** We cite recent safety benchmarks and refusal/over-refusal work; to our knowledge, a **multi-dataset, cross-model, low-FPR map of SE’s failure** (with contamination-aware paraphrases) is not yet published. ([arXiv][1])

---

## Final line you’ll give the decision gate

> **Select:** **2) Create new hypotheses suite.**
> Rationale: H1 is complete; the next step is to **codify H2–H5** and run the **SE-failure replication** (JBB-Paraphrase + HarmBench-Matched) on **Qwen-2.5-7B-Instruct** and **Llama-4-Scout-17B**, reporting **FNR\@5% FPR** across τ (no tuning). Optional: closed-API aggregate and a brief procedural appendix (TAP). ([arXiv][1])

---

**You’ve got this.**

[1]: https://arxiv.org/abs/2404.01318?utm_source=chatgpt.com "An Open Robustness Benchmark for Jailbreaking Large ..."
[2]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f69b4b96a46f284b726fbd70f74fb3b-Abstract-Datasets_and_Benchmarks_Track.html?utm_source=chatgpt.com "WildGuard: Open One-stop Moderation Tools for Safety ..."
[3]: https://aclanthology.org/2023.emnlp-main.557/?utm_source=chatgpt.com "SelfCheckGPT: Zero-Resource Black-Box Hallucination ..."
[4]: https://www.nature.com/articles/s41586-024-07421-0?utm_source=chatgpt.com "Detecting hallucinations in large language models using ..."
[5]: https://arxiv.org/abs/2402.04249?utm_source=chatgpt.com "HarmBench: A Standardized Evaluation Framework for ..."
[6]: https://proceedings.neurips.cc/paper_files/paper/2024/file/63092d79154adebd7305dfd498cbff70-Paper-Datasets_and_Benchmarks_Track.pdf?utm_source=chatgpt.com "JailbreakBench: An Open Robustness Benchmark for ..."
[7]: https://pubmed.ncbi.nlm.nih.gov/38898292/?utm_source=chatgpt.com "Detecting hallucinations in large language models using ..."
[8]: https://neurips.cc/virtual/2024/poster/97764?utm_source=chatgpt.com "WildGuard: Open One-stop Moderation Tools for Safety ..."
[9]: https://huggingface.co/meta-llama/Llama-Guard-3-8B?utm_source=chatgpt.com "meta-llama/Llama-Guard-3-8B"
[10]: https://openreview.net/forum?id=YfKNaRktan&utm_source=chatgpt.com "SORRY-Bench: Systematically Evaluating Large Language ..."
[11]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/9622163c87b67fd5a4a0ec3247cf356e-Abstract-Conference.html?utm_source=chatgpt.com "SORRY-Bench: Systematically Evaluating Large Language ..."
[12]: https://arxiv.org/abs/2405.20947?utm_source=chatgpt.com "OR-Bench: An Over-Refusal Benchmark for Large Language Models"
[13]: https://openreview.net/forum?id=obYVdcMMIT&utm_source=chatgpt.com "An Over-Refusal Benchmark for Large Language Models"
[14]: https://livebench.ai/?utm_source=chatgpt.com "LiveBench"
[15]: https://arxiv.org/abs/2406.19314?utm_source=chatgpt.com "LiveBench: A Challenging, Contamination-Free LLM Benchmark"
[16]: https://openreview.net/forum?id=sKYHBTAxVa&utm_source=chatgpt.com "LiveBench: A Challenging, Contamination-Limited LLM ..."
[17]: https://arxiv.org/abs/2312.02119?utm_source=chatgpt.com "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically"
[18]: https://proceedings.neurips.cc/paper_files/paper/2024/file/ea456e232efb72d261715e33ce25f208-Paper-Conference.pdf?utm_source=chatgpt.com "Many-shot Jailbreaking"
[19]: https://openreview.net/forum?id=cw5mgd71jW&utm_source=chatgpt.com "Many-shot Jailbreaking"
```