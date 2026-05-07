### Research Plan v2.1: The Consistency Confound - Mechanism and Brittleness Analysis

Working Thesis: SE’s utility as a safety signal decreases with model alignment; any apparent wins are brittle to configuration.

--- 

Priority 1: Codify the New Research Direction

* Task 1: Create Paper Outline
 * Description: Create the papers/outline.md file to anchor the paper's narrative, claims, and key figures.
 * Implementation: Use the Write_file tool to create papers/outline.md.
 * Reference/Content: The content will be taken directly from the "papers/outline.md" section of the mentor feedback note (mentor_feedback_report_checkpoint_1).

* Task 2: Create Methodology Notes
 * Description: Create the papers/methodology_notes.md file to precisely define the SE variant used and contrast it with its original context in hallucination detection.
 * Implementation: Use the Write_file tool to create papers/methodology_notes.md.
 * Reference/Content: Content from mentor feedback, section "papers/methodology_notes.md", including links to the Nature and SelfCheckGPT papers.

* Task 3: Create v3 Hypotheses Suite
 * Description: Create a new, definitive hypothesis file (hypotheses_suites/hypotheses_suite_v3_final.md) that includes the original H1, the completed H2, the new mechanism hypotheses (H-M1, H-M2), and the revised experimental hypotheses (H3', H4, H3, H5, H6).
 * Implementation: Write_file to create the new suite.
 * Reference/Content: Synthesize all claims from the mentor feedback note into a single, comprehensive document.

--- 

Priority 2: Mechanism Verification (Analysis on Existing Data)

* Task 4: Execute H-M1 Analysis (Refusal Homogeneity)
 * Description: Verify that Llama-4's refusals are short and templated while its benign answers are long and varied, using existing H1 and H2 results.
 * Subtasks:
 1. Compute Dispersion Diagnostics: For each prompt in the JBB and HarmBench datasets, calculate duplicate-rate and #clusters from existing diagnostics.
 2. Compute Length-Residualized SE (H3'): Fit a simple linear model SE ~ log(length) on benign prompts and calculate the residuals. Re-run the evaluation (AUROC, FNR@5%FPR) using these residuals as the detector score. 3. Compute Prefix Homogeneity: For each prompt, calculate the mean pairwise bigram overlap across the first 30 tokens of the N=5 responses.
 * Reference Files: outputs/h1/llama4scout_120val_results.json, outputs/h2/evaluation/llama-4-scout-17b-16e-instruct_h2_results.json, outputs/h1/Llama4_jbb120val_responses.jsonl, outputs/h2/response_generation/llama-4-scout-17b-16e-instruct_h2_responses.jsonl.
 * Artifacts: A summary report (reports/h_m1_analysis_report.md) with tables and figures showing the results of these analyses.

* Task 5: Execute H4 (Brittleness Band Analysis)
 * Description: Demonstrate that SE's performance is brittle to hyperparameter changes, and its one "win" (Qwen on HarmBench) is not robust.
 * Subtasks:
 1. Plot Brittleness Curves: Using existing H1 and H2 results, generate plots of FNR@5%FPR vs. τ and AUROC vs. τ for both models on both datasets.
 2. Top-up N=10 Data (Optional but Recommended): For the specific case of Qwen on HarmBench, generate 5 additional responses for each prompt to create an N=10 dataset.
 3. Evaluate N=10 Case: Re-run the scoring and evaluation for the Qwen/HarmBench/N=10 case to show that the SE win at τ=0.1 collapses.
 * Reference Files: All H1 and H2 result files (*.json).
 * Artifacts: papers/figures/figure_brittleness_curves.png, reports/h4_brittleness_report.md.

--- 

Priority 3: New Data Generation and Evaluation

* Task 6: Execute H3 (Paraphrase Robustness)
 * Description: Test the hypothesis that paraphrasing disproportionately degrades SE's performance, especially on the weaker Qwen model.
 * Subtasks:
 1. Build Paraphrase Dataset: Create the JBB-Paraphrase-2025-08 dataset by processing data/processed/jbb_validation.jsonl with the R2J protocol.
 2. Run Full Pipeline: Execute the response generation, scoring, and evaluation pipeline on this new dataset for both Qwen and Llama-4.
 3. Report Performance Degradation: Calculate and report the ΔFNR@5%FPR and ΔAUROC from the original JBB results (H1) to the paraphrased results.
 * Reference Files: data/processed/jbb_validation.jsonl, H1 result files.
 * Artifacts: data/processed/jbb_paraphrase_test.jsonl, outputs/h3/h3_results.json, reports/h3_paraphrase_report.md.

--- 

Priority 4: Final Qualitative Analysis and Reporting

* Task 7: Execute H5 (Qualitative Audit)
 * Description: Collect and document clear examples of the Consistency Confound from the experimental runs.
 * Implementation: A manual or semi-automated review of the response logs (e.g., Llama4_jbb120val_responses.jsonl and the new logs from H2/H3).
 * Reference Files: All H1/H2 response logs and scoring files.
 * Artifacts: reports/h5_qualitative_audit.md with anonymized examples and summary statistics per category.

* Task 8: Generate Final Paper Assets
 * Description: Consolidate all experimental results into the final figures and tables for the paper.
 * Implementation: Create scripts to generate all figures and tables specified in papers/outline.md, ensuring all metrics include 95% Confidence Intervals (Wilson for FNR, DeLong for AUROC).
 * Reference Files: All H1-H5 result files, papers/outline.md.
 * Artifacts: papers/figures/figure_1.png, papers/tables/h4_sensitivity.md, etc.