# Paper Outline Review

## What Works Well
- Memorable, accurate title: \"The Consistency Confound\" clearly captures the central mechanism and guides the narrative.
- Clear central hypothesis stated early: that high semantic entropy (SE) can serve as a reliable, zero-shot jailbreak detector; and a crisp refutation is presented.
- Strong, multi-dataset evidence: Results span two models (Llama-4-Scout-17B, Qwen-2.5-7B) and two datasets (JailbreakBench, HarmBench twins), with consistent logging and file references.
- Mechanism-oriented analysis: The Consistency Confound is articulated, tested, and quantitatively supported by H6 audits.
- Baselines are sensible and implemented: Average Pairwise BERTScore, Embedding Variance, Levenshtein Variance (implementation noted in src/core/baseline_metrics.py).
- Concrete numbers are provided for most claims in Results sections with filepaths to raw JSONs, which supports reproducibility.

## Critical Issues Found

### Missing Results
- H4 brittleness likely includes N-sensitivity (N ∈ {5,10}) per configs/project_config.yaml, but the outline only reports τ sensitivity. If N-results exist, they are not reported.
  - Potential source: outputs/h4/evaluation/h4_brittleness_results.json (check for N grid entries) and outputs/h4/logs/h4_brittleness_evaluation_run_detailed_logs.txt
- H3 per-prompt diagnostics exist but are not referenced in the outline (useful for visualization/examples):
  - outputs/h3/per_prompt_analysis/llama-4-scout-17b-16e-instruct_H2_h3_prompt_analysis.jsonl
  - outputs/h3/per_prompt_analysis/qwen2.5-7b-instruct_H2_h3_prompt_analysis.jsonl
- H2 includes detailed evaluation reports not cited explicitly in the outline narrative:
  - outputs/h2/evaluation/h2_llama-4-scout-17b-16e-instruct_evaluation_report.md
  - outputs/h2/evaluation/h2_qwen2.5-7b-instruct_evaluation_report.md### Accuracy Problems  
- Evaluation protocol mismatch: Outline Section 2.4 states a 30% benign calibration split and 95% CIs (Wilson for FNR, DeLong for AUROC). However:
  - configs/project_config.yaml: evaluation: \"Note: No calibration/test split - we report metrics across full tau grid\".
  - Raw results files do not contain CI fields and use \"actual_fpr\" values that differ from 0.05, indicating thresholds were selected on the full set rather than a held-out calibration split.
  - Action: Either revise the outline to reflect the actual protocol (no cal/test split, report exact actual_fpr), or re-run evaluations with the stated calibration protocol and compute/report CIs.
- Section 4.3 claim \"Baselines actually saw performance improve\" is only partially true for Qwen/JBB paraphrases: Avg Pairwise BERTScore and Embedding Variance improved in FNR, but Levenshtein Variance worsened (FNR +9pp). Source: outputs/h5/evaluation/h5_paraphrase_degradation_report.md. Clarify to \"some baselines improved\" and quantify.
- Ensure the H6 Llama/JBB figure (73.3%) is explicitly confirmed from the raw JSON:
  - outputs/h6/llama-h1-jailbreakbench/llama-4-scout-17b-16e-instruct_H1_h6_qualitative_audit_results.json (outline cites 73.3%; confirm percentage and counts in text).

### Narrative Gaps
- Section 2.2 Table 1 (SE variants comparison) is proposed but not specified. Provide concrete entries and citations, including precise dimensions compared and constraints.
- Figure/Table specifications lack axes and exact data sources. The guidelines require explicit plot specs (axes, ranges, data files). For Figures 1–3 and Tables 1–2, add:
  - Axes definitions, data source filepaths, and selection rules (e.g., which τ used for SE?).
- The method selection for τ is ambiguous across sections. Sometimes the outline discusses the \"best\" τ; elsewhere it references a particular τ (0.1) or reports AUROC at different τ without clarifying selection policy. Make τ selection criteria explicit (fixed vs grid vs best-at-test; consistent across H1–H2–H5) and state it at first use.
- Statistical significance: The outline claims CIs and DeLong comparisons, but no evidence is present in the results files. Either include the computed intervals with sources or remove the claim and add to Future Work.

## Specific Improvements Needed

### Section-by-Section Feedback
**Introduction:**
- Current: Strong framing of behavioral signals and the central refutation; introduces Consistency Confound well.
- Issue: Claims three central points; consider foreshadowing the evaluation-protocol nuance (thresholding, τ sensitivity) to manage reader expectations.
- Suggestion: Add 1–2 sentences clarifying that SE’s brittleness to τ and small N is a driver of unreliability; reference Sections 4.2 and H4 outputs/h4/evaluation/h4_brittleness_results.json.

**Methodology (2.1–2.4):**
- Current: Describes SE variant and baselines, datasets, and hyperparameters. Points to baseline implementation file.
- Issues:
  - Protocol mismatch: Outline describes calibration split + CIs, but config and outputs indicate otherwise.
  - Missing implementation breadcrumbs for SE: src/core/semantic_entropy.py and evaluation code src/core/evaluation.py are not referenced.
  - Embedding model details (dimension=1024, L2 norm) claimed in text; config lists model but not dimension. That detail is fine to include, but add a source reference (e.g., model card) or code reference where normalization is applied.
- Suggestions:
  - Align protocol: Either remove calibration/CIs or run the calibrated evaluation and add CI fields to results and reports. If running calibrated protocol, note the split seed and file list used for the 30% benign calibration set with a manifest.
  - Add explicit file references for SE implementation and evaluation scripts: src/core/semantic_entropy.py, src/experiments/h1/run_h1_scoring.py, src/core/evaluation.py.
  - Specify the τ selection rule per section (e.g., report both best τ AUROC and FNR@5%FPR at τ=0.1 for comparability across H2/H4), and justify.

**Results 3.1 (H1, JBB):**
- Current: States SE AUROC 0.625 (Llama), 0.529 (Qwen), baselines outperform; cites files.
- Issues: Make explicit which τ produced the SE AUROC in each case (H1 Llama optimal τ=0.3 per outputs/h1/evaluation/llama4scout_120val_results.json; Qwen likely optimal τ=0.2). Also report FNR@5%FPR values with actual_fpr.
- Suggestions:
  - Add: \"SE (τ=0.3) AUROC=0.625, FNR@5%FPR=0.733, actual_fpr=0.0167; BERTScore AUROC=0.767, FNR=0.600 (actual_fpr=0.05).\" with explicit file path.
  - For Qwen, add the corresponding numbers from outputs/h1/evaluation/qwen25_120val_results.json and note that τ=0.1 has AUROC≈0.690 but collapses at thresholding (FNR=1.0), as documented in outputs/h5/evaluation/h5_paraphrase_degradation_report.md.

**Results 3.2 (H2, HarmBench twins):**
- Current: Correctly notes instability and model/data dependence; cites Llama and Qwen JSONs.
- Issues: Clarify τ selection for SE and state the best baseline by both AUROC and FNR@5%FPR for completeness. For Llama, embedding variance is best by FNR (0.605 vs SE 0.654) with AUROC 0.684 (outputs/h2/evaluation/llama-4-scout-17b-16e-instruct_h2_results.json). For Qwen, explicitly report SE AUROC (0.733) and FNR, with the τ used.
- Suggestions:
  - Add actual_fpr and thresholds to the text to capture calibration realities.
  - Expand Table 2 to include SE results for all τ values and baselines for both datasets, with file references.

**Results 4.1 (H3, Length confound):**
- Current: Claims R²≈0.103 and residual AUROC drop from 0.691 to 0.630 for Llama/H2.
- Issues: Provide both the regression model spec (features, targets), residualization procedure, and the exact file references.
- Suggestions:
  - Specify: \"We regress SE score on response length using OLS across prompts (N=162); R²=0.103. Using residualized SE, AUROC drops from 0.691 (SE τ=0.1) to 0.630; source: outputs/h3/results/llama-4-scout-17b-16e-instruct_H2_h3_results.json.\" If additional per-prompt results exist, reference the JSONL files.

**Results 4.2 (H4, Brittleness):**- Current: Reports τ brittleness for Qwen/H2: FNR jumps 0.630→0.889 when τ increases 0.1→0.2; cites outputs/h4/evaluation/h4_brittleness_results.json.
- Issues: configs/project_config.yaml indicates N-grid testing (5,10). These results are not discussed.
- Suggestions:
  - If present in h4_brittleness_results.json, add N-sensitivity (e.g., \"At N=10, FNR@5%FPR at τ=0.1 improved by X pp, but brittleness persisted (τ=0.2 FNR=Y)\"). If not present, either run or explicitly state they were planned but not executed.
  - For Figure 2, add axes and data source: \"x-axis: τ ∈ {0.1,0.2,0.3,0.4}, y-axis: FNR@5%FPR, data: outputs/h4/evaluation/h4_brittleness_results.json (Qwen/H2, SE and baseline), include actual_fpr error bars if varying from 0.05.\"

**Results 4.3 (H5, Paraphrase robustness):**
- Current: States SE not uniquely harmed; for Qwen/JBB SE already FNR=1.0 and baselines improve.
- Issues: Clarify which baselines improved and by how much; include AUROC deltas.
- Suggestions:
  - Add explicit numbers from outputs/h5/evaluation/h5_robustness_evaluation.json and outputs/h5/evaluation/h5_paraphrase_degradation_report.md. E.g., \"Avg Pairwise BERTScore FNR: 0.867→0.804 (-6.3pp), Embedding Variance: 0.967→0.946 (-2.0pp), Levenshtein: 0.767→0.857 (+9.0pp).\"

**Section 5 (H6, Consistency Confound):**
- Current: Well-argued mechanism, quantitative audit with clear criteria, and strong examples; cites Llama/JBB and Qwen/H2.
- Issues: Qwen H2 audit JSON unavailable; markdown report confirms 97.5%. Llama/JBB JSON should be cited with counts and denominator in-text to contextualize 73.3%.
- Suggestions:
  - Add counts: \"Explained 73.3% (e.g., 44/60) of FNs\" if that is the true denominator; cite outputs/h6/llama-h1-jailbreakbench/...json and match with per_prompt_predictions.jsonl if needed.
  - For Figure 3, add axis labels, legend, and data source paths.

**Discussion/Conclusion:**
- Current: Strong implications and clear cautionary framing.
- Suggestions: Add a brief \"Limitations\" paragraph acknowledging black-box constraints (no token logprobs), small N for SE estimates, and dependency on embedding model and clustering threshold; point to Section 2.2 and configs.

### Missing Experiments to Include
- H4 N-sensitivity analysis
  - Location: outputs/h4/evaluation/h4_brittleness_results.json (if present), configs/project_config.yaml (brittleness_grid)
  - Suggested placement: Section 4.2
- H3 per-prompt residualization diagnostics
  - Location: outputs/h3/per_prompt_analysis/..._prompt_analysis.jsonl
  - Suggested placement: Section 4.1 as a supplementary figure/table.

### Reproducibility Gaps
- Calibration and CIs not implemented as described:
  - Missing config references for calibration split: none in configs/project_config.yaml; raw files lack CI fields.
  - Action: Either re-run with calibration split + Wilson/DeLong CIs, or change the text to match current non-calibrated evaluation and remove CI claims.
- τ selection policy is not standardized in text. Define explicitly and apply consistently.
- Exact seeds are partially specified (global_seed=42; H4 provides seed=42). Ensure response generation logs include seed and decode params; cross-reference outputs/h1/response_generation/*logs.md and outputs/h2/response_generation/*_generation_log.md.
- Embedding model specifics: Note the exact model version (Alibaba-NLP/gte-large-en-v1.5), dimensionality, and normalization steps; add reference to method code where normalization is applied.

## Priority Revisions (Ranked)
1. Resolve evaluation protocol mismatch: either implement the 30% calibration + CI computation or revise Section 2.4 and all results to reflect the actual protocol (include actual_fpr and thresholds).
2. Standardize τ selection/reporting across sections; specify and justify policy (e.g., report best-τ AUROC and FNR at τ=0.1 for robustness comparisons) and apply consistently.
3. Add precise figure/table specifications (axes, sources, ranges) and include any missing results (H4 N-sensitivity, H3 per-prompt diagnostics) with exact file paths.
4. Quantify all claims with exact numbers and file paths (e.g., Section 4.3 baseline improvements with deltas and AUROC changes), and add counts/denominators for H6 percentages.
5. Add implementation breadcrumbs for SE and evaluation code paths and decoding configs to strengthen reproducibility.

## Verification Checklist
- [ ] All completed experiments included: NO – H4 N-sensitivity likely missing; H3 prompt-level diagnostics not referenced.
- [ ] All metrics accurate: PARTIAL – Numerical values match raw files for H1 Llama and H2 Llama; Qwen H1/H2 values appear consistent but τ selection and thresholding nuances are not always stated. Section 4.3 needs nuanced phrasing on baseline changes.
- [ ] Narrative flow logical: YES – The story flows from hypothesis to refutation to mechanism; add explicit τ policy to tighten coherence.
- [ ] Reproducibility info complete: NO – Calibration/CIs mismatch; τ policy unspecified; add code paths and seeds in text; add actual_fpr and thresholds.
- [ ] Future work clearly separated: YES – H7 is marked as planned and unexecuted; consider expanding with prioritized experiments.
