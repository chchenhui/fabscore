# Mentor Brief — Post-H2 Decision, Mechanism Hypotheses, and Marching Orders

*Date:* 25 Aug 2025
*From:* Research Mentor
*To:* Autonomous AI/ML Researcher

---

## A. Direct answers to your three meeting questions

**1) Revising the thesis.**
**Yes—pivot the thesis.** Use:

> **“SE’s utility as a safety signal decreases with model alignment; any apparent wins are brittle to configuration.”**

Why this is justified: Semantic Entropy (SE) was proposed for hallucination/confabulation detection, not jailbreak/refusal behavior (“Detecting hallucinations in large language models using semantic entropy,” Nature 2024). On strongly aligned models, harmful prompts elicit **templated refusals** that compress variation (SE→0) without improving **low-FPR** separability; this is exactly how jailbreak benchmarks expect detectors to be evaluated (JailbreakBench, NeurIPS 2024; HarmBench, 2024).

**2) Framing the brittleness.**
**Yes—the hyperparameter sensitivity strengthens your claim.** A metric that “wins” only at **τ=0.1** and collapses at **τ=0.2** or when **N** increases from 5→10 is non-deployable. Report **FNR\@5% FPR vs τ** (and AUROC vs τ) explicitly. That converts an anecdote into a principled negative result about **non-robustness** under standard operating points (JailbreakBench; HarmBench).

**3) Redefining H3 (paraphrasing).**
**Yes—adjust H3 asymmetrically.** Hypothesize that **post-dated paraphrases** (to reduce contamination) will **hurt SE more** than simpler baselines, with a **larger drop on Qwen-2.5-7B** (paraphrases break any rote refusal strings and induce style churn) and **little change on Llama-4-Scout** (already templated). This mirrors contamination-limited evaluation practice (e.g., LiveBench, ICLR 2025 Spotlight) and keeps the study anchored in realistic shifts of phrasing.

---

## B. Two mechanism hypotheses you should test (empirical ≠ statistical)

> We need **statistical validity** (low-FPR, CIs) and **empirical validity** (observable mechanisms in your N=5 regime). These two hypotheses give both.

### H-M1 — Refusal-template homogeneity vs. benign topical variety

**Claim.** On aligned models (e.g., Llama-4-Scout), harmful prompts mostly trigger **short, templated refusals** (lexically homogeneous across samples), while benign prompts elicit **longer, topic-specific** answers with higher dispersion.
**Why this explains the metrics.** Avg pairwise **BERTScore** and **embedding-variance** track dispersion directly; **SE** first thresholds dispersion via τ-clustering and then compresses counts into Shannon entropy—making SE **brittle** for small N.
**How to verify now (no new decoding):**

* **Duplicate-rate & cluster count:** per prompt, `dup_rate = 1 − (unique/N)` and `#clusters` from your SE diagnostics. Expect **high dup-rate / few clusters** for **harmful\@Llama-4** across τ.
* **Length-residualized SE:** fit `SE(τ) ~ α + β·log(length)` on **benign**, compute residuals, then **AUROC** & **FNR\@5%** on residual SE. Expect **≈ chance** on Llama-4 while **BERTScore/variance** still separate.
* **Prefix homogeneity:** bigram overlap across the first \~30 tokens for the N samples—expect very high overlap on harmful\@Llama-4.
* **Qualitative snippets:** a small plate of refusal templates (connects to over-refusal literature: OR-Bench 2024; SORRY-Bench 2024).

### H-M2 — Contamination/novelty drives dispersion patterns

**Claim.** On **older/known** jailbreaks (JBB), aligned models retrieve **memorized refusal phrasing** (low dispersion → low SE). On **newer/contextual** harms (HarmBench), weaker models lack a single refusal template and exhibit **style churn** (higher dispersion → occasional SE “wins” at brittle τ).
**How to verify:**

* **Paraphrase sensitivity (JBB→JBB-para):** compute ΔFNR\@5% and ΔAUROC; **SE** should drop more than baselines, especially on Qwen.
* **Dataset vintage effect:** harmful dispersion (lower BERTScore, higher embedding-variance) is larger on HarmBench than JBB for Qwen; Llama-4 remains template-like on both.
* **τ-brittleness uniquely for SE:** your Qwen\@HarmBench blip at τ=0.1 should vanish at τ≥0.2 (and at N=10 if you top-up), while BERTScore/variance curves are smoother.

**Context to cite:** JailbreakBench (NeurIPS 2024); HarmBench (2024); LiveBench contamination-limited philosophy (ICLR 2025); SelfCheckGPT (EMNLP 2023) as the multi-sample-consistency lineage SE belongs to.

---

## C. What to run next (fast, decisive, minimal generation)

**1) H3′ — Length-controlled separability (no re-generation).**

* Compute per-response length; take per-prompt median length.
* Fit `SE(τ) ~ α + β·log(length)` on **benign**; evaluate **residual SE** for harmful vs benign.
* Report **AUROC** and **FNR\@5% FPR** for residual SE (compare to BERTScore/variance residuals).

**2) H4 — “Brittleness band” (not tuning; report full mini-grid).**

* **τ ∈ {0.1, 0.2, 0.3, 0.4} at N=5** for both datasets/models using existing outputs.
* **Optional:** Only where SE “wins” (Qwen\@HarmBench, τ=0.1), **top-up to N=10** (+5 samples per prompt) to show the win collapses with modest N.
* Plot **FNR\@5% vs τ** (and AUROC vs τ).

**3) H3 — Paraphrase (small, post-dated slice).**

* Build “JBB-Paraphrase-2025-08” (harmful+benign).
* Compute **ΔFNR\@5%** & **ΔAUROC** original→paraphrased; predict **SE degrades most**, especially on Qwen.

**4) H5 — Short qualitative audit (30–50 items).**

* Tag SE false negatives at 5% FPR into: templated refusal, topic-specific refusal, compliance drift, “benign-looking but policy-safe.”
* Include 2–3 anonymized snippets per bucket + per-bucket dup-rate/#clusters summaries.

**5) (Optional) H6 — One big-model sanity check.**

* One larger open-weights model **or** one closed API on a **60/60 JBB** slice.
* Report **aggregate** AUROC and TPR/FNR\@5% (no per-vendor micro-claims).
* If it contradicts the trend, that’s valuable to report.

---

## D. Reporting & statistical hygiene (math-only on existing scores)

* **Primary operating point:** **FNR\@5% FPR** and **AUROC** (JailbreakBench; HarmBench).
* **Uncertainty:**

  * **Wilson 95% CI** for FNR (binomial FN / #harmful).
  * **DeLong 95% CI** for AUROC; use **paired DeLong** when comparing SE vs baseline AUCs on the same prompts.
* **Threshold caveat:** If the 5% threshold is estimated on the same benign set, FNR is optimistic—add a one-line note (optionally a **5× bootstrap pseudo-split** sanity check: threshold from a benign resample; FNR on an independent harmful resample).
* **Diagnostics to keep:** class-conditional means; **dup-rate** and **#clusters** per prompt; **length distributions** (and residualized SE).
* **Paper hygiene:** update your **Methods** doc as you compute each item above (formulas, CI choices, bootstrap note, regex list if you ever use refusal tagging), and keep a **page-1 figure plan** (FNR\@5% vs τ; AUROC bars with CIs; qualitative refusal plate).

**Key references you’ll cite in paper text**

* JailbreakBench: Chao et al., NeurIPS 2024 (low-FPR jailbreak evaluation).
* HarmBench: Mazeika et al., 2024 (contextual harms, robust refusal).
* Semantic Entropy: Farquhar et al., Nature 2024 (hallucination/confabulation detection).
* SelfCheckGPT: Manakul et al., EMNLP 2023 (multi-sample consistency lineage).
* OR-Bench & SORRY-Bench: 2024 (over-refusal behavior).
* LiveBench: ICLR 2025 Spotlight (contamination-limited philosophy).

---

## E. Minimal repo/process notes (so results are reproducible)

* Log provider/route, model ID, seeds, N/T/top-p, τ; persist raw generations.
* For paraphrases, save the rewrite prompt, model, and any post-filters; keep an audit CSV.
* Add one **local open-weights** replication on a 20-prompt slice if you worry about API routing/caching.

---

## F. Decision for the agent (pick one of four)

**→ Choose: 2) Create new hypotheses suite.**
You’re not blocked by implementation (#3), the idea isn’t exhausted (#1), and we shouldn’t just proceed to the next test (#4) without locking brittleness, length-control, and paraphrase sensitivity. The updates above are **small, decisive**, and will turn your current tables into a **clean, publishable** negative-plus-mechanism paper.

---

### One-paragraph abstract target (for orientation)

> *We evaluate Semantic Entropy (SE) as a jailbreak detector at low-FPR operating points across JailbreakBench and HarmBench. On a strongly aligned model (Llama-4-Scout), SE fails consistently; on a weaker model (Qwen-2.5-7B), SE shows a narrow win only at τ=0.1 that vanishes with slight τ or N changes. Length-controlled analyses and dispersion diagnostics (duplicate-rate/cluster counts) indicate that aligned models’ harmful prompts produce templated refusals (SE≈0), while weaker models on novel harms exhibit style churn. Post-dated paraphrases disproportionately degrade SE. We conclude that SE’s utility **decreases with alignment**, and any wins are **brittle**, explaining the field’s preference for refusal-aware or single-pass detectors over multi-sample diversity.*
