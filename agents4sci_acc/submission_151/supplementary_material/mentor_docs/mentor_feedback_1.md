# Mentor Feedback

## 1) Community Impact

**Nudges**

* Which *specific* safety gap does this idea reduce: prevention (robust refusal), detection (post-gen catching), or both via feedback into training? Is the claim decision-relevant for eval owners today?
* Can you articulate how your signal performs **under new attack families** (not seen during tuning), since attacks evolve weekly?

---

## 2) Generalization Pathway

**Nudges**

* What is your **unseen-attack** story (train on A → test on B + in-the-wild)?
* How will you demonstrate **style robustness** (semantics-preserving format/paraphrase transforms) so results aren’t style artifacts?

---

## 3) Impact–Complexity Tradeoff

**Nudges**

* What is the smallest **feature set** that still moves AUROC/FNR on unseen attacks? Favor black-box signals first; add internals only if uniquely additive.
* Could your signal be useful **offline** (triage, hard-negative mining) even if not deployed at runtime?

---

## 4) Method–Problem Alignment

**Nudges**

* Are you testing **non-verbal epistemic signals** (e.g., sample disagreement, entropy, refusal-flip stability) as *behavioral* evidence of jailbreak—rather than relying on **verbal markers** that are easy to spoof?
* If you keep verbal markers, can you elicit them **post-hoc (two-turn)** so you don’t prime behavior?

---

## 5) Failure Mode Planning

**Nudges**

* **Hardness confound:** how will you prove you’re not merely detecting “benign-but-hard” prompts? (Match length/role-play/complexity.)
* **Spoofability:** what happens if an attacker forces confident tone (or suppresses markers)?
* **Model diversity:** will conclusions hold across at least two model families?

---

## 6) Empirical Rigor & Skepticism

**Nudges**

* What **negative controls** will you include (complex-benign prompts matched to jailbreak style) to show safety-specific signal?
* Which **judging protocols** (≥2 LLM-judge prompts + a human subset) will you use to avoid prompt-sensitive evaluator bias?

---

## 7) Autonomous Agent Tractability

**Nudges**

* Can an autonomous agent run end-to-end on a **single device** with standard HF tooling? State sampling counts, max tokens, and any optional logprob needs.
* If internals (grads/activations) are optional, make that clear so black-box settings still work.

---

## 8) Novelty Verification

**Nudges**

* List the **closest detectors** and state your delta in one sentence each (e.g., “ours is post-gen behavioral, black-box, style-robust”).
* Ensure at least one **domain shift** (e.g., code security) appears as an unseen evaluation, not the only focus.

---

## 9) Citation Discipline

**Nudges**

* When you cite, separate **benchmarks** (JailbreakBench, HarmBench), **detectors** (GradSafe, Gradient Cuff, HiddenDetect, COSMIC, WildGuard), and **uncertainty/hallucination** (SelfCheckGPT) so novelty is unmistakable.
* Call out evaluator fragility explicitly and justify your multi-judge/human check.


---

[1]: https://arxiv.org/abs/2404.01318?utm_source=chatgpt.com "An Open Robustness Benchmark for Jailbreaking Large ..."
[2]: https://github.com/JailbreakBench/jailbreakbench?utm_source=chatgpt.com "JailbreakBench: An Open Robustness Benchmark for ..."
[3]: https://arxiv.org/abs/2402.04249?utm_source=chatgpt.com "HarmBench: A Standardized Evaluation Framework for ..."
[4]: https://www.harmbench.org/?utm_source=chatgpt.com "HarmBench"
[5]: https://arxiv.org/html/2411.03343v2?utm_source=chatgpt.com "What Features in Prompts Jailbreak LLMs? Investigating ..."
[6]: https://arxiv.org/abs/2403.00867?utm_source=chatgpt.com "Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes"
[7]: https://arxiv.org/abs/2303.08896?utm_source=chatgpt.com "SelfCheckGPT: Zero-Resource Black-Box Hallucination ..."
[8]: https://aclanthology.org/2023.emnlp-main.557.pdf?utm_source=chatgpt.com "SELFCHECKGPT: Zero-Resource Black-Box Hallucination ..."
[9]: https://openreview.net/forum?id=RwzFNbJ3Ez&utm_source=chatgpt.com "SelfCheckGPT: Zero-Resource Black-Box Hallucination ..."
[10]: https://aclanthology.org/2024.acl-long.30.pdf?utm_source=chatgpt.com "GradSafe: Detecting Jailbreak Prompts for LLMs via Safety- ..."
[11]: https://arxiv.org/pdf/2403.00867?utm_source=chatgpt.com "Gradient Cuff: Detecting Jailbreak Attacks on Large ..."
[12]: https://github.com/xyq7/GradSafe?utm_source=chatgpt.com "Official Code for ACL 2024 paper \"GradSafe"
[13]: https://arxiv.org/pdf/2404.01318?utm_source=chatgpt.com "JailbreakBench"
[14]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f69b4b96a46f284b726fbd70f74fb3b-Abstract-Datasets_and_Benchmarks_Track.html?utm_source=chatgpt.com "WildGuard: Open One-stop Moderation Tools for Safety ..."
[15]: https://arxiv.org/abs/2506.10022?utm_source=chatgpt.com "LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges"
[16]: https://aclanthology.org/2025.acl-long.1350/?utm_source=chatgpt.com "Malware Requests and Jailbreak Challenges"
[17]: https://aclanthology.org/2025.acl-long.724/?utm_source=chatgpt.com "Detecting Jailbreak Attacks against Multimodal Large ..."
