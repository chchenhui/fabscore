## Reasoning

"The mentor's feedback provides a clear and decisive directive to pivot the research project. The previous research idea has been invalidated, and the new goal is to formalize and publish a rigorous negative result. The mentor has explicitly recommended selecting 'Create new hypotheses suite' as the next step and has provided the exact structure for the new hypotheses (H2-H5/H6). This plan systematically maps the failure of Semantic Entropy across different models, datasets, and conditions, aligning the project with community standards for robustness and rigor. All other options are inappropriate: the idea is now fixed, there is no code to debug for the new direction yet, and continuing with the old hypotheses is not an option. Executing the mentor's plan is the only logical and productive path forward."

## Plan
### Research Plan: The Consistency Confound (v2)

Working Title: The Consistency Confound: Semantic Entropy Fails for Jailbreak Detection Across Models and Benchmarks

Priority 1: Foundational Scaffolding

* Task 1: Create Paper Outline Document
 * Owner: Claude Code
 * Description: Create the papers/outline.md file to anchor the paper's narrative and structure.
 * Implementation: Use the Write_file tool to create papers/outline.md.
 * Reference/Content: The content will be taken directly from the "papers/outline.md" section of the mentor feedback note.

* Task 2: Create Methodology Notes Document
 * Owner: Claude Code
 * Description: Create the papers/methodology_notes.md file to formally define the specific variant of Semantic Entropy being investigated.
 * Implementation: Use the Write_file tool to create papers/methodology_notes.md.
 * Reference/Content: The content will be taken directly from the "papers/methodology_notes.md" section of the mentor feedback note, including citations to the Nature and SelfCheckGPT papers.

**Priority 2: Codify New Hypotheses STATUS: COMPLETE - File : /idea_14/revised_hypotheses_20250821_160000.json**

Owner: Hypotheses_Suite_Generation_Prompt.md
* Task 3: Rewrite the Hypotheses Suite
 * Description: Update the hypotheses_suite.md file to reflect the new H2-H5 (and optional H6) focused on systematically testing the failure of Semantic Entropy. H1 will be preserved as the completed foundational experiment.
 * Implementation: Use Read_file on the existing hypotheses_suite.md to retrieve H1, then use Write_file to create a new version (hypotheses_suite_v2.md) that appends the new H2-H6 as specified by the mentor.
 * Reference/Content: The claims, metrics, and protocols for H2, H3, H4, H5, and H6 will be taken directly from the "Hypotheses to rewrite" section of the mentor feedback note.

Priority 3: Experimental Design & Setup

* **Task 4: Draft New Master Experimental Plan. COMPLETE: Owner: Experiment Planning Code.md - COMPLETE**
 * Description: Create a new experimentation_master_plan_v2.md that provides a detailed, step-by-step roadmap for executing the new H2-H6.
 * Implementation: Use Write_file to create the new plan document.
 * Reference/Content: This plan must detail:
 * Dataset Creation: The protocol for building HarmBench-Benign-Matched and JBB-Paraphrase-2025-08, referencing the procedures in the mentor note.
 * Execution Scripts: Define the required CLI entrypoints (e.g., run_se_h2, run_baselines_h2) and the exact structure of their CSV outputs.
 * Analysis Scripts: Plan for scripts to generate Figure 1 (ROC curve) and the sensitivity tables for H4.

* Task 5: Build the HarmBench Matched Benign Dataset
 * Owner: Claude Code
 * Description: Execute the plan to create the HarmBench-Benign-Matched dataset needed for H2.
 * Subtasks:
 1. Pool Benign Prompts: Load benign prompts from data/processed/wildguard_test.jsonl.
 2. Generate Embeddings: Use the Alibaba-NLP/gte-large-en-v1.5 model to generate embeddings for the benign pool and the data/processed/harmbench_contextual_test.jsonl prompts.
 3. Perform Matching: For each harmful prompt, find a benign prompt that is top-k in cosine similarity (≥ 0.80) and within ±20% length.
 4. Create Splits: Split the resulting matched dataset into a 30% calibration set (harmbench_matched_calibration.jsonl) and a 70% test set (harmbench_matched_test.jsonl).
 5. Generate Report: Create a harmbench_matching_report.md with similarity and length distributions.
 * Reference/Content: Follow the detailed protocol in the "How to compute AUROC for HarmBench" section of the mentor note.

Priority 4: Execution of New Hypotheses
 * Owner: Claude Code
* Task 6: Execute H2 (Cross-Dataset Replication)
 * Description: Run the SE vs. Avg-BERTScore comparison on the newly created HarmBench matched dataset for both Qwen-2.5-7B-Instruct and Llama-4-Scout-17B.
 * Implementation: Develop and run scripts (run_h2_v2.py) that perform response generation, scoring, calibration on the 30% split, and final evaluation on the 70% split.
 * Artifacts: Generate CSV logs and a summary markdown file (h2_v2_summary.md) reporting AUROC and FNR@5% FPR.

* Task 7: Execute H3 (Paraphrase Robustness)
 * Description: Generate paraphrases for the JBB-120 validation set and re-run the H1 experiment to measure performance degradation.
 * Implementation: First, create a script to generate paraphrases for prompts in data/processed/jbb_validation.jsonl. Then, run the H1 evaluation pipeline on this new dataset (jbb_paraphrase_test.jsonl).
 * Artifacts: Generate CSV logs and a summary markdown file (h3_v2_summary.md) reporting the ΔAUROC and ΔFNR@5% FPR compared to the original H1 results.

* Task 8: Execute H4 (Sensitivity Analysis)
 * Description: Run the experiment across a grid of τ, N, and T values as a reporting exercise.
 * Implementation: Create a script (run_h4_v2.py) that iterates through the specified grid, performing the calibration/test split evaluation for each combination.
 * Artifacts: A comprehensive CSV file and a summary table (h4_v2_sensitivity_table.md) showing that SE's failure is not an artifact of a specific parameter choice.

* Task 9: Execute H5 (Qualitative Exemplars)
 * Description: Collect and document clear examples of the Consistency Confound from the experimental runs.
 * Implementation: A manual or semi-automated review of the response logs (e.g., Llama4_jbb120val_responses.jsonl and the new logs from H2/H3).
 * Artifacts: A markdown file (h5_v2_exemplars.md) containing anonymized prompts and response snippets/hashes that illustrate the key failure mechanisms, as described in the mentor's note."