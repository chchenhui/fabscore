# Progress Log

## Session 1 — Extraction
- **Purpose**: Extract experimental results from paper
- **Paper inspected**: `202_SciVerify_Digits_A_Benchma.pdf` (13 pages)
- **Files created/updated**:
  - `fabscore_claude/fs_extracted.json` — created with tables, figures, results_section
- **Summary**:
  - Extracted 16 entries from Table 1 (model accuracy % across MNIST, Fashion-MNIST, SVHN, Permuted SVHN for Simple Concat, Attention Fusion, Deep Sets, Multimodal LLM)
  - Extracted 9 figure entries (Figures 1a, 1b, 2a, 2b, 3, 4, 5, 6a, 6b)
  - results_section is empty: all numeric values in Section 4.2 body text are already captured in Table 1; remaining text claims are qualitative (no standalone numeric performance tokens)
- **Next session**: Run analysis/execution phase (fs_analysis.json, fs_execution.json, fs_summary.json)

## Session 2 — Analysis (Static)
- **Date**: 2026-03-31
- **Purpose**: Static analysis of all 25 claims against repository artifacts
- **Files inspected**:
  - `logs/0-run/draft_summary.json` — best MNIST val accuracy: 0.7183
  - `logs/0-run/ablation_summary.json` — MNIST val: 0.6958, Fashion-MNIST val: 0.6562, SVHN val: 0.6583
  - `logs/0-run/stage_3_creative_research_1_first_attempt/best_solution_5d63515e2e1a419abdebbeac6bebe10c.py` — multi-dataset CNN+BERT
  - `logs/0-run/stage_4_ablation_studies_1_first_attempt/best_solution_59b34ff867e042798e4396182e237630.py` — ablation CNN+BERT
  - All 31 experiment_code.py files (all use CNNVisionEncoder + BERT concatenation)
  - experiment_7b1f855ebccc4df58e5a7f7e69167e93 (no_logic_supervision)
  - experiment_b13540f014c245cd92fd745ad33c32f8 (permute_order_ablation)
  - experiment_308a27ddfc324b599284897fa2919013 (random_claim_text adversarial)
  - experiment_02c55b64a8e04a7ea58f6be9f28d8629 (ablation comparison)
- **Files created/updated**:
  - `fabscore_claude/fs_analysis.json` — created with full classification of all 25 claims
- **Summary**:
  - Paper claims 4 architectures in Table 1: Simple Concat, Attention Fusion, Deep Sets, Multimodal LLM
  - Repository only implements ONE architecture: CNN+BERT concatenation (ClaimVerifier class)
  - Best MNIST accuracy in any experiment log: 71.83% (vs paper's Simple Concat claim of 85.1%)
  - Claims 5-12 (Attention Fusion, Deep Sets): `no_code_files` — no implementations found anywhere
  - Claims 1-4 (Simple Concat): `insufficient_evidence` — CNN+BERT concat exists but claimed values (85.1%/62.3%/58.9%/51.4%) not found in any execution record; actual Fashion-MNIST/SVHN results EXCEED paper's Simple Concat claims (counter-intuitive pattern)
  - Claims 13-16 (Multimodal LLM): `insufficient_evidence` — CNN+BERT exists, values are plausible-ish but don't precisely match any run; Permuted SVHN not explicitly implemented as a Table 1 comparison
  - Claims 17-25 (Figures): `execution_required` — corresponding experiments exist (permuted order, random_claim_text, no_logic_supervision), PNG artifacts and experiment_data.npy files are present, but execution needed to verify figure content vs. paper descriptions
- **Key finding**: The claimed values for Simple Concat, Attention Fusion, and Deep Sets in Table 1 appear unsupported by any execution record in the repository. Only one fusion architecture (concatenation) is implemented, yet the paper claims 3 distinct architectures. The actual achieved results are substantially different from the paper's Table 1 values.
- **Next session**: Execute plotting scripts and attempt to reproduce Table 1 comparison if possible (fs_execution.json phase)

## Session 3 — Execution: Claim 17 (Figure 1a)
- **Date**: 2026-03-31
- **Purpose**: Verify claim 17: Figure 1(a) shows training/validation accuracy curves on MNIST for different epoch settings
- **Files inspected**:
  - `logs/0-run/experiment_results/experiment_02c55b64a8e04a7ea58f6be9f28d8629_proc_1522608/plotting_code.py` — ablation study (attention_mask_removal), not epoch settings
  - `logs/0-run/experiment_results/experiment_23b7cc9670dc469d91f273d62ad1176e_proc_1502176/plotting_code.py` — correct one: generates MNIST accuracy curves for different epoch settings
  - `logs/0-run/experiment_results/experiment_23b7cc9670dc469d91f273d62ad1176e_proc_1502176/experiment_data.npy` — loaded and inspected; contains `num_epochs_tuning.mnist_claims` with three settings: epochs_10, epochs_20, epochs_30
- **Key findings**:
  - `experiment_data.npy` has real training/validation accuracy data for epochs 10, 20, 30 on MNIST claims
  - `mnist_claims_accuracy_curve.png` already exists as the generated plot
  - Data is plausible (accuracy ~0.69-0.75 range)
  - This matches Figure 1(a) description: training/validation accuracy for different epoch settings
- **Verdict**: Verified — underlying data in `experiment_data.npy` supports Figure 1(a) description
- **Next session**: Continue with other claims

## Session 5 — Execution: Claim 19 (Figure 2a)
- **Date**: 2026-03-31
- **Purpose**: Verify claim 19: Figure 2(a) shows validation accuracy when input order of digits is permuted
- **Files inspected**:
  - `logs/0-run/experiment_results/experiment_b13540f014c245cd92fd745ad33c32f8_proc_1522608/experiment_data.npy` — loaded and inspected; contains `permute_order_ablation` key with mnist, fashion_mnist, svhn sub-keys, each having metrics.val (50 epoch values)
  - `logs/0-run/experiment_results/experiment_b13540f014c245cd92fd745ad33c32f8_proc_1522608/plotting_code.py` — generates `val_acc_compare_permuted_ablation.png` titled "Validation Accuracy Comparison (Permuted Order Ablation)"
  - Pre-generated artifacts: `val_acc_compare_permuted_ablation.png`, `mnist_acc_permuted.png`, `mnist_accuracy_curves_permuted_ablation.png`
- **Key findings**:
  - `experiment_data.npy` has real validation accuracy data for permuted order ablation across 50 epochs
  - MNIST val accuracy final: ~0.669, Fashion-MNIST val: ~0.860, SVHN val: ~0.556
  - Plotting code directly generates a "Validation Accuracy Comparison (Permuted Order Ablation)" figure
  - This matches Figure 2(a) description: validation accuracy when input order is permuted
- **Artifact reused**: `experiment_data.npy` (structure inspected via Python)
- **Verdict**: Verified — underlying data in `experiment_data.npy` supports Figure 2(a) description
- **Next session**: Continue with other claims

## Session 6 — Execution: Claim 20 (Figure 2b)
- **Date**: 2026-03-31
- **Purpose**: Verify claim 20: Figure 2(b) shows validation accuracy with random adversarial claims across datasets
- **Files inspected**:
  - `logs/0-run/experiment_results/experiment_308a27ddfc324b599284897fa2919013_proc_1522608/experiment_data.npy` — loaded; contains `random_claim_text` key with `mnist`, `fashion_mnist`, `svhn` sub-keys, each with 50 epoch metrics (train/val/train_logic/val_logic)
  - `logs/0-run/experiment_results/experiment_308a27ddfc324b599284897fa2919013_proc_1522608/plotting_code.py` — generates per-dataset accuracy/loss curves, final accuracy comparison bar chart, and overlay plots
  - Pre-generated artifacts: `random_claim_text_val_acc_compare.png` exists
- **Key findings**:
  - Final val accuracy: MNIST ~54.4%, Fashion-MNIST ~55.8%, SVHN ~54.4% (all ~50% range consistent with random adversarial claims)
  - `random_claim_text_val_acc_compare.png` exists matching Figure 2(b) exactly
  - Underlying data in experiment_data.npy directly supports Figure 2(b) description
- **Artifact reused**: `experiment_data.npy` inspected via Python; `random_claim_text_val_acc_compare.png` pre-generated
- **Verdict**: Verified — underlying data in `experiment_data.npy` supports Figure 2(b) content
- **Next session**: Continue with other claims

## Session 7 — Execution: Claim 21 (Figure 3)
- **Date**: 2026-03-31
- **Purpose**: Verify claim 21: Figure 3 shows training and validation loss curves on MNIST claims for different epoch settings
- **Files inspected**:
  - `logs/0-run/experiment_results/experiment_02c55b64a8e04a7ea58f6be9f28d8629_proc_1522608/plotting_code.py` — ablation study (attention_mask_removal), generates loss curves per dataset for ablation, NOT epoch settings
  - `logs/0-run/experiment_results/experiment_23b7cc9670dc469d91f273d62ad1176e_proc_1502176/plotting_code.py` — generates `mnist_claims_loss_curve.png` titled "Train/Validation Loss Curves\nMNISTClaimDataset (num_epochs tuning)"
  - `logs/0-run/experiment_results/experiment_23b7cc9670dc469d91f273d62ad1176e_proc_1502176/experiment_data.npy` — loaded and inspected; `num_epochs_tuning.mnist_claims` contains epochs_10, epochs_20, epochs_30 each with `losses.train` and `losses.val` arrays
  - `logs/0-run/experiment_results/experiment_23b7cc9670dc469d91f273d62ad1176e_proc_1502176/mnist_claims_loss_curve.png` — pre-generated loss curve figure
- **Key findings**:
  - `experiment_data.npy` contains real training/validation loss data for 3 epoch settings on MNIST claims: epochs_10 (train_loss: 0.6104→0.5329), epochs_20 (0.5972→0.5019), epochs_30 (0.5936→0.4505)
  - Note: Candidate files in claim pointed to experiment_02c55b64a8e04a7ea58f6be9f28d8629 (ablation), but actual relevant experiment is experiment_23b7cc9670dc469d91f273d62ad1176e (epoch tuning)
  - The plotting_code.py generates loss curves for different epoch settings matching Figure 3 description
- **Artifact reused**: `experiment_data.npy` inspected via Python; `mnist_claims_loss_curve.png` pre-generated
- **Verdict**: Verified — underlying loss data in `experiment_data.npy` supports Figure 3 description
- **Next session**: Continue with other claims

## Session 8 — Execution: Claim 22 (Figure 4)
- **Date**: 2026-03-31
- **Purpose**: Verify claim 22: Figure 4 shows validation logical consistency accuracy when input order of digits is permuted
- **Files inspected**:
  - `logs/0-run/experiment_results/experiment_b13540f014c245cd92fd745ad33c32f8_proc_1522608/plotting_code.py` — lines 106-125 generate `val_logic_acc_compare_permuted_ablation.png` titled "Logical Consistency Accuracy Comparison (Permuted Order Ablation)" from `val_logic` metrics
  - `logs/0-run/experiment_results/experiment_b13540f014c245cd92fd745ad33c32f8_proc_1522608/experiment_data.npy` — confirmed via fresh execution that it contains `val_logic` metrics for all three datasets (mnist, fashion_mnist, svhn) across 50 epochs
  - Pre-generated artifacts: `val_logic_acc_compare_permuted_ablation.png`, per-dataset `*_logic_acc_curves_permuted_ablation.png` all exist
- **Key findings**:
  - `experiment_data.npy` has real `val_logic` (validation logical consistency accuracy) data: MNIST final=0.6687, Fashion-MNIST final=0.8604, SVHN final=0.5563 over 50 epochs
  - The plotting code directly generates Figure 4 (val_logic_acc_compare_permuted_ablation.png)
  - All underlying data supports the claim description
- **Command executed**: `python3 -c "import numpy as np; ..."` (inspect experiment_data.npy val_logic metrics)
- **Verdict**: Verified — underlying val_logic data in experiment_data.npy and pre-generated val_logic_acc_compare_permuted_ablation.png fully support Figure 4 description
- **Next session**: Continue with other claims

## Session 9 — Execution: Claim 23 (Figure 5)
- **Date**: 2026-03-31
- **Purpose**: Verify claim 23: Figure 5 shows validation logical consistency accuracy with random adversarial claims across datasets
- **Files inspected**:
  - `logs/0-run/experiment_results/experiment_308a27ddfc324b599284897fa2919013_proc_1522608/experiment_data.npy` — contains `random_claim_text` key with mnist, fashion_mnist, svhn sub-keys, each with `val_logic` metric (50 epochs)
  - `logs/0-run/experiment_results/experiment_308a27ddfc324b599284897fa2919013_proc_1522608/random_claim_text_val_logic_acc_compare.png` — pre-generated artifact exists
- **Command executed**: `python3 -c "import numpy as np; ..."` — inspected val_logic from experiment_data.npy
- **Key findings**:
  - `val_logic` data: MNIST (first=0.5792, last=0.5437), Fashion-MNIST (first=0.5854, last=0.5583), SVHN (first=0.5854, last=0.5437) — all ~54-58% consistent with random adversarial claims
  - Pre-generated `random_claim_text_val_logic_acc_compare.png` directly corresponds to Figure 5
  - Underlying val_logic data in experiment_data.npy fully supports Figure 5 description
- **Verdict**: Verified — underlying val_logic data in experiment_data.npy and pre-generated plot support Figure 5 content
- **Next session**: Continue with remaining claims

## Session 10 — Execution: Claim 24 (Figure 6a)
- **Date**: 2026-03-31
- **Purpose**: Verify claim 24: Figure 6(a) shows confusion matrix (No Logic Supervision MNIST)
- **Files inspected**:
  - `logs/0-run/experiment_results/experiment_7b1f855ebccc4df58e5a7f7e69167e93_proc_1522608/plotting_code.py` — lines 53-79 generate scatter plot (plt.scatter), NOT a confusion matrix
  - `logs/0-run/experiment_results/experiment_7b1f855ebccc4df58e5a7f7e69167e93_proc_1522608/experiment_data.npy` — loaded; contains ground_truth (480 binary values) and predictions (480 binary values)
  - Paper PDF (Appendix E): Section heading "E Confusion Matrices Without Logical Supervision"; Figure 6 caption: "Confusion matrices showing ground truth vs. predictions without logical consistency enforcement."
  - Pre-generated `mnist_no_logic_val_gt_vs_pred.png` exists (scatter plot)
- **Key findings**:
  - Underlying data: TP=146, TN=203, FP=86, FN=45 (binary 0/1 values)
  - The plotting code generates a scatter plot (plt.scatter) with xticks/yticks at [0,1]
  - The paper EXPLICITLY calls Figure 6(a) a "confusion matrix" in both Appendix E heading and figure caption
  - The actual generated figure is a scatter plot, NOT a confusion matrix
  - This is a clear conflict: paper's description says confusion matrix, code generates scatter plot
- **Artifact created**: `fabscore_claude/workspace/claim_24_command_output.txt`
- **Verdict**: Result Fabrication — paper describes Figure 6(a) as a "confusion matrix" but the plotting code generates a scatter plot of ground truth vs predictions
- **Next session**: Continue with remaining claims (claim 25)

## Session 11 — Execution: Claim 25 (Figure 6b)
- **Date**: 2026-03-31
- **Purpose**: Verify claim 25: Figure 6(b) shows confusion matrix (No Logic Supervision Fashion-MNIST)
- **Files inspected**:
  - `logs/0-run/experiment_results/experiment_7b1f855ebccc4df58e5a7f7e69167e93_proc_1522608/plotting_code.py` — same file as claim 24; loop over ["mnist", "fashion_mnist", "svhn"], section 3 (lines 53-79) generates scatter plot (plt.scatter) titled "FASHION_MNIST - Val: Ground Truth vs Prediction\n(No Logic Supervision, Last Epoch)" saved as `fashion_mnist_no_logic_val_gt_vs_pred.png`
  - Progress from Session 10 (claim 24): confirmed experiment_data.npy has ground_truth and predictions as binary (0/1) values for all three datasets including fashion_mnist
- **Key findings**:
  - The plotting code generates a scatter plot for Fashion-MNIST (plt.scatter with xticks/yticks at [0,1]), NOT a confusion matrix
  - The paper (Appendix E) explicitly describes Figure 6 as "Confusion matrices showing ground truth vs. predictions without logical consistency enforcement"
  - This is the identical conflict found in claim 24 (MNIST), but now for Fashion-MNIST
  - The generated file `fashion_mnist_no_logic_val_gt_vs_pred.png` is a scatter plot, not a confusion matrix
- **Artifact reused**: plotting_code.py (already read in Session 10 context, re-read to confirm Fashion-MNIST section)
- **Verdict**: Result Fabrication — paper describes Figure 6(b) as a "confusion matrix" but the plotting code generates a scatter plot for Fashion-MNIST
- **Next session**: All claims have been processed

## Session 4 — Execution: Claim 18 (Figure 1b)
- **Date**: 2026-03-31
- **Purpose**: Verify claim 18: Figure 1(b) shows validation accuracy comparison across datasets
- **Files inspected**:
  - `logs/0-run/experiment_results/experiment_02c55b64a8e04a7ea58f6be9f28d8629_proc_1522608/plotting_code.py` — ablation study (attention_mask_removal), generates per-dataset accuracy curves
  - `logs/0-run/experiment_results/experiment_02c55b64a8e04a7ea58f6be9f28d8629_proc_1522608/experiment_code.py` — contains `plot_metric_curve_ablation` function (line 421), called with "val"/"Validation Accuracy"/"val_acc_compare_ablation.png" at line 440
  - `logs/0-run/experiment_results/experiment_02c55b64a8e04a7ea58f6be9f28d8629_proc_1522608/experiment_data.npy` — loaded and verified; contains `attention_mask_removal` key with mnist, fashion_mnist, svhn sub-keys
  - `logs/0-run/experiment_results/experiment_02c55b64a8e04a7ea58f6be9f28d8629_proc_1522608/val_acc_compare_ablation.png` — pre-generated artifact exists
- **Key findings**:
  - `experiment_data.npy` contains validation accuracy across 50 epochs for 3 datasets: mnist (max ~0.76), fashion_mnist (max ~0.83), svhn (max ~0.68)
  - `plot_metric_curve_ablation` function generates a single plot with all 3 datasets titled "[Ablation: No Attn Mask] Validation Accuracy across datasets"
  - The `val_acc_compare_ablation.png` artifact already exists
  - The underlying data directly supports a "validation accuracy comparison across datasets" as described in Figure 1(b)
- **Artifact reused**: `experiment_data.npy` and `val_acc_compare_ablation.png` from the ablation experiment directory
- **Verdict**: Verified — underlying data in `experiment_data.npy` supports Figure 1(b) description (validation accuracy comparison across datasets)
- **Next session**: Continue with other figure/claims
