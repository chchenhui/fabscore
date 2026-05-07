# Progress Log

## Session 1
- **Purpose**: extraction
- **Paper inspected**: 126_MT_ViT_CCHA_Multi_Task_Lea.pdf (MT-ViT-CCHA: Multi-Task Learning for Canine Cardiomegaly Classification and VHS Keypoint Detection)
- **Files created/updated**: `fabscore_claude/fs_extracted.json`
- **Summary**:
  - Extracted 54 table entries from Table 1 (performance comparison of 14 models, validation and test accuracy) and Table 2 (ablation study with 13 experiment configurations, validation and test accuracy).
  - Extracted 3 figures: Figure 1 (model architecture), Figure 2 (training/validation loss curves), Figure 3 (confusion matrix).
  - Extracted 1 results_section claim: the standard deviation (1.38%) of MT-ViT-CCHA test accuracy, which was not present in any table.
- **Next session**: Run analysis (fs_analysis.json) to evaluate the quality of the paper's claims and reproducibility.

## Session 3 (2026-04-24)
- **Purpose**: execution (claim 55 - Figure 1 model architecture)
- **Files inspected**:
  - `Supplementary files/code/create_figures.py` - contains `create_model_architecture_figure()` function
  - `Supplementary files/code/run_experiments.py` - ProposedModel implementation
- **Execution**:
  - Created workspace script `fabscore_claude/workspace/run_arch_fig.py` (replica of `create_model_architecture_figure()` with placeholder image instead of proprietary dataset image)
  - Ran script; generated `fabscore_claude/workspace/model_architecture_claim55.png` successfully
  - `claim_55_command_output.txt` records output
- **Findings**:
  - `create_model_architecture_figure()` generates a diagram titled "Proposed Model (ViT Base)" containing: ViT Backbone, Cross-Attention, Keypoint Head (HRNet), Classification Head, VHS Head, and outputs (Keypoint Heatmaps, Class Logits, VHS Prediction, Diagnosis)
  - This matches the paper's claim of Figure 1 showing the "Proposed model (ViT base) architecture for MT-ViT-CCHA"
- **Verdict**: Verified - the figure script successfully generates the model architecture figure matching the claim
- **Next session**: No further action needed for this claim

## Session 4 (2026-04-24)
- **Purpose**: execution (claim 56 - Figure 2: Training and validation loss curves)
- **Files inspected**:
  - `Supplementary files/Results/results.json` - Contains `history.train_loss`, `history.val_loss`, `history.train_accuracy`, `history.val_accuracy` arrays (20 epochs each)
  - `Supplementary files/code/create_figures.py` - Script that reads results.json and plots curves
- **Execution**:
  - Ran inline Python script (equivalent to create_figures.py main body) to generate curves figure
  - Saved output to `fabscore_claude/workspace/claim_56_curves.png`
  - `claim_56_command_output.txt` records output
- **Findings**:
  - results.json contains 20 epochs of train_loss (range 0.0017–4.4603) and val_loss (range 0.3756–1.0638)
  - Script successfully generates two-subplot figure: "Training and Validation Loss" and "Training and Validation Accuracy"
  - Figure 2 caption in paper: "Training and validation loss curves for MT-ViT-CCHA" - matches what the script produces
- **Verdict**: Verified - underlying data in results.json supports the figure, and script successfully generates loss/accuracy curves
- **Next session**: No further action needed for this claim

## Session 5 (2026-04-24)
- **Purpose**: execution (claim 57 - Figure 3: Confusion matrix for MT-ViT-CCHA)
- **Files inspected**:
  - `Supplementary files/code/plot_confusion_matrix.py` - requires model weights and DogHeart dataset
  - `Supplementary files/Results/results.json` - has accuracy/precision/recall/f1 but NO raw prediction arrays or confusion matrix data
  - Searched repository for `.pth` model weights - none found
  - Searched for `.png` confusion matrix artifacts - none found in repo (only workspace files from prior sessions)
- **Execution**: No commands run (prerequisites missing: no model weights, no dataset)
- **Findings**:
  - `plot_confusion_matrix.py` requires `../../Results/best_model.pth` (model checkpoint) and `./Data` (DogHeart dataset)
  - Neither the model weights nor the dataset exists in the repository
  - `results.json` contains aggregate metrics (accuracy=0.81 val, 0.845 test) but no per-class prediction arrays that could reconstruct the confusion matrix
  - No pre-generated confusion matrix image exists in the repository
  - The code path is plausible but both critical prerequisites are missing, making reproduction impossible
- **Verdict**: Insufficient Evidence - plot_confusion_matrix.py exists but requires missing model weights and proprietary dataset; no saved prediction artifacts exist to reconstruct the confusion matrix
- **Next session**: No further action needed for this claim

## Session 2 (2026-04-24)
- **Purpose**: analysis
- **Files inspected**:
  - `Supplementary files/Results/performance_comparison.json` - Table 1 source (14 baselines + Our model)
  - `Supplementary files/Results/results.json` - Main single-run result (val: 0.81, test: 0.845)
  - `Supplementary files/Results/multi_task_ablation.json` - Table 2 multi-task ablation test accuracies
  - `Supplementary files/Results/cross_attention_kp_head_ablation.json` - Table 2 architecture ablation
  - `Supplementary files/Results/loss_ablation.json` - Table 2 loss weighting ablation
  - `Supplementary files/Results/finetune_summary.json` - Multi-seed mean (81.8%) and std (1.38%)
  - `Supplementary files/Results/multiseed_finetune/results_10..50.json` - 5 seed results
  - `Supplementary files/ablation_results/*/results.json` - Individual ablation run results (val + test)
  - `Supplementary files/code/create_tables.py` - Table generation pipeline
  - `Supplementary files/code/create_figures.py` - Figure generation pipeline
- **Files created**:
  - `fabscore_claude/fs_analysis.json`
- **Summary of classifications**:
  - **static_verifiable (53 claims)**: Claims 1-27, 29-53, 58
    - All 13 baseline models (claims 1-26) exactly match performance_comparison.json
    - MT-ViT-CCHA validation accuracy 81.0% (claim 27, 53) matches results.json and performance_comparison.json
    - All 12 ablation studies (claims 29-52) exactly match their ablation_results JSON files (including rounding)
    - Results section claim (58) exactly matches finetune_summary.json (mean=0.818, std=0.0138)
  - **obvious_hallucination / result_fabrication (2 claims)**: Claims 28 and 54
    - Claim 28 (Table 1, MT-ViT-CCHA, Test 81.8%): performance_comparison.json shows 84.5% (from results.json), not 81.8%. The 81.8% is the multi-seed mean from finetune_summary.json - a different experiment. The create_tables.py code explicitly derives Table 1 from results.json (test: 0.845).
    - Claim 54 (Table 2, Full Model, Test 81.8%): All ablation aggregation JSONs show "Our model" = 84.5%. The paper reports 81.8% from the multi-seed finetune, not the single-run used by the table generation script.
  - **execution_required (3 claims)**: Claims 55, 56, 57 (three figures)
    - No pre-generated figure artifacts (.png files) found in repository
    - create_figures.py and plot_confusion_matrix.py exist as generation scripts
- **Next session**: No further action needed - analysis complete. The key finding is a result mis-mapping for MT-ViT-CCHA in Tables 1 and 2: the paper reports multi-seed mean (81.8%) where the table-generation pipeline produces 84.5% from a single run.
