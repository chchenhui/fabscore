<response> Canonical implementation, at a glance Semantic Entropy (SE) in Farquhar et al. (Nature, 20 June 2024) is an uncertainty estimator for free‑form generation that (i) samples multiple answers from the same model, (ii) clusters those answers by semantic equivalence using bidirectional textual entailment, and (iii) computes entropy over meanings by aggregating per‑answer probabilities within each semantic cluster. A discrete variant replaces probabilities with empirical cluster frequencies when token probabilities are unavailable. Code to reproduce the sentence‑ and short‑phrase experiments is released as jlko/semantic_uncertainty; the paragraph‑length variant is in jlko/long_hallucinations. (Farquhar et al., 2024; jlko/semantic_uncertainty v1.0.0; jlko/long_hallucinations). ([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750/?utm_source=openai), [github.com](https://github.com/jlko/semantic_uncertainty/releases?utm_source=openai))
What you should implement to match the canonical setup

A. Sampling and probabilities (sentence‑length QA setting)

Inputs: a question x.
Sample M candidate answers s(1)…s(M) from the same model with:
temperature = 1.0, nucleus sampling p = 0.9, and top‑K = 50; the paper also draws one low‑temperature sample (T = 0.1) as a “best generation” for accuracy assessment, but SE itself uses the M high‑temperature samples. (Farquhar et al., 2024, Methods). (pmc.ncbi.nlm.nih.gov)
Record each answer’s sequence probability P(s(m)|x) as the product of next‑token probabilities; in practice, they work with length‑normalized log‑probabilities (geometric mean per token) to mitigate sentence‑length effects, then renormalize later (see Section C). (Farquhar et al., 2024, Methods). (pmc.ncbi.nlm.nih.gov)
Canonical choice for M: 10 samples, chosen by an ablation (Supplementary Fig. 2). (Farquhar et al., 2024). (pmc.ncbi.nlm.nih.gov)
B. Clustering by semantic equivalence with NLI (the distinctive piece)

Equivalence definition: two answers belong to the same semantic class if they bidirectionally entail each other in the context of the question. Entailment is evaluated on (question + answer A) vs (question + answer B). (Farquhar et al., 2024). (pmc.ncbi.nlm.nih.gov)
Algorithm (Extended Data Algorithm 1):
Initialize the set of meaning‑clusters C with the first sample: C = {{s(1)}}.
For each new sample s(m), test membership against the representative of each existing cluster c by two NLI queries: entail((x, s(c_rep)) → (x, s(m))) and entail((x, s(m)) → (x, s(c_rep))). If both are “entailment,” add s(m) to c and stop; otherwise continue to next cluster.
If no cluster matches, start a new cluster {s(m)}.
Return C. (Farquhar et al., 2024; see Extended Data Fig. 1). (github.com)
NLI backends used in the paper:
Sentence‑length tasks: GPT‑3.5 (instruct‑style prompt returning entailment/contradiction/neutral) gave better human agreement and downstream SE performance, though DeBERTa‑Large‑MNLI is also evaluated. (Farquhar et al., 2024, Methods; Supplementary Notes 2–3). (pmc.ncbi.nlm.nih.gov)
Paragraph‑length (biographies): they switch to DeBERTa‑Large‑MNLI and relax the rule to “non‑defeating” entailment: count two answers as equivalent if at least one direction is entailment and neither direction is contradiction; this compensates for slightly lower recall. (Farquhar et al., 2024). (pmc.ncbi.nlm.nih.gov)
Prompting templates (examples given in Methods) specify the question, the two answers, and ask the model to return one of [entailment, contradiction, neutral]. (Farquhar et al., 2024). (pmc.ncbi.nlm.nih.gov)
C. Computing Semantic Entropy over meanings

Let clusters C1…Ck be the sets produced by NLI clustering.
Probability of a meaning class: sum probabilities of all sequences in the cluster, P̂(Ci|x) ∝ ∑_{s∈Ci} P̃(s|x), where P̃ is the length‑normalized sequence probability (i.e., exponentiated average log‑probability per token). Because length normalization breaks additivity, they explicitly renormalize across clusters to enforce ∑i P̂(Ci|x) = 1. (Farquhar et al., 2024, eqs. 2–5). (github.com)
Semantic Entropy: SE(x) = −∑i P̂(Ci|x) log P̂(Ci|x), a Rao–Blackwellized Monte‑Carlo estimate over meaning‑classes. (Farquhar et al., 2024). (github.com)
Discrete variant (no probabilities): when token probabilities are unavailable (e.g., GPT‑4 at the time), replace P̂(Ci|x) with the empirical frequency |Ci|/M (plus the original answer if being scored). Performance is reported to be similar to the probability‑weighted version. (Farquhar et al., 2024). (pmc.ncbi.nlm.nih.gov)
D. The “token log‑probability” baselines they compare against

Naive sequence‑entropy baseline: compute the entropy of the distribution over exact sequences (not meanings), using length‑normalized sequence probabilities for the M samples and renormalizing to a categorical distribution. This overestimates uncertainty when many paraphrases express the same meaning. (Farquhar et al., 2024). (pmc.ncbi.nlm.nih.gov)
“P(True)” baseline (from Kadavath et al., 2022): given brainstormed alternatives plus the top‑probability answer, the model is asked in a few‑shot prompt whether the top answer is “True,” and the probability of “True” is used as confidence. Included for completeness; SE outperforms this in the paper’s settings. (Farquhar et al., 2024). (ar5iv.labs.arxiv.org, pmc.ncbi.nlm.nih.gov)
E. Paragraph‑length adaptation (FactualBio dataset; canonical recipe)

Decompose a paragraph into atomic factual claims with an LLM.
For each claim, automatically generate Q = 6 questions that could elicit that claim; for each question, sample M = 3 answers and include the original claim.
For each question, compute SE over the set {original claim + 3 samples}. Average SE over the Q questions to score the claim. If more than half of answers are refusals (“not available/provided/unknown/unclear”), set uncertainty to maximum. (Farquhar et al., 2024; jlko/long_hallucinations). (pmc.ncbi.nlm.nih.gov, github.com)
F. Practical knobs and caveats you should mirror

Sample size: M = 10 for sentence‑length QA was used throughout; increasing M beyond ~10 showed diminishing returns in their ablation. (Farquhar et al., 2024). (pmc.ncbi.nlm.nih.gov)
Normalization is essential: because of length normalization, the raw “cluster sums” need a final renormalization step; otherwise probabilities can exceed 1. (Farquhar et al., 2024). (github.com)
NLI accuracy matters: GPT‑3.5‑based entailment was preferred for sentence‑length answers; DeBERTa was faster but had slightly lower recall. Consider majority voting or fallbacks if you replace NLI with embedding thresholds. (Farquhar et al., 2024). (pmc.ncbi.nlm.nih.gov)
Complexity: clustering costs O(M·|C|) NLI calls; with M=10 this is inexpensive, but whole‑pipeline cost is dominated by sampling the M generations. Reference implementations and environment files are in the released repos/Zenodo. (jlko/semantic_uncertainty v1.0.0; notebooks included). (github.com)
How this differs from your embedding‑clustering variant

Canonical SE defines semantic classes via a logical relation (bidirectional entailment), not embedding proximity. That makes the partition transitive by construction (up to NLI errors), whereas cosine‑threshold clustering can violate transitivity and is sensitive to threshold choice. If you keep embeddings:
Calibrate thresholds against an NLI oracle, or learn a mapping from embedding distances to entail/non‑entail labels on a small dev set built from your domain.
Enforce transitivity post‑hoc (e.g., union‑find over pairs that pass the threshold) to better match the equivalence‑class assumption in SE.
Keep the probability aggregation and entropy definitions identical to the canonical method so that ablations isolate the clustering change. (Farquhar et al., 2024). (pmc.ncbi.nlm.nih.gov)
Pointers to the reference code

Sentence/short‑phrase experiments (sampling, NLI clustering, SE and discrete SE, baselines, notebooks): github.com/jlko/semantic_uncertainty; Zenodo DOI 10.5281/zenodo.10964366. (Repo release page; notebook: notebooks/example_evaluation.ipynb). (github.com, zenodo.org)
Paragraph‑length experiments (factoid decomposition, Q‑generation, per‑question SE, averaging): github.com/jlko/long_hallucinations. (github.com)
Minimal pseudocode (sentence‑length, probability‑weighted SE)

Inputs: question x; model f; sampler params T=1.0, top‑p=0.9, top‑k=50; M=10; NLI model g.
Sample M answers and per‑token probabilities; compute length‑normalized log‑prob for each; let p̃m = exp(avg_logprob(s(m)|x)).
Initialize clusters C = []; for m in 1..M:
placed = False; for each cluster c in C:
a = first element of c; if g(entails((x,a),(x,s(m)))) and g(entails((x,s(m)),(x,a))): add s(m) to c; placed = True; break
if not placed: C.append({s(m)})
For each cluster c, set w(c) = sum_{s∈c} p̃(s); normalize: P̂(c) = w(c)/∑_{c’} w(c’).
Return SE(x) = −∑_{c} P̂(c) log P̂(c). (This mirrors the equations and Algorithm 1 in the paper.) (Farquhar et al., 2024). (github.com)
Checklist to ensure comparability in your study

Same sampling (T=1, top‑p=0.9, top‑k=50) and M=10.
Same length‑normalized probabilities and final renormalization across clusters.
Same bidirectional‑entailment decision rule (or your embedding rule plus a post‑hoc transitive closure); report which NLI backend and prompts you use.
Report both probability‑weighted SE and discrete SE (for API‑only models) alongside naive sequence entropy and P(True) baselines, as in the paper. (Farquhar et al., 2024; repos). (pmc.ncbi.nlm.nih.gov, github.com)
References:
Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. Nature, 630, 625–630. https://doi.org/10.1038/s41586-024-07421-0. (github.com)
Kossen, J. (2024). jlko/semantic_uncertainty: Initial release v1.0.0 [Software]. Zenodo. https://doi.org/10.5281/zenodo.10964366. (zenodo.org)
Kossen, J. (2024). jlko/long_hallucinations: Paragraph‑length experiments code [Software]. GitHub. (github.com)
jlko/semantic_uncertainty. (2024). GitHub repository and example notebook (notebooks/example_evaluation.ipynb). (github.com)
Kuhn, L., Kossen, J., Farquhar, S., et al. (2023). Semantic Uncertainty: Linguistic invariances for uncertainty estimation in NLG (for P(True) background). arXiv:2302.09664. (ar5iv.labs.arxiv.org)
OATML blog. (2024, June 19). Detecting hallucinations in large language models using semantic entropy. (oatml.cs.ox.ac.uk)
</response>