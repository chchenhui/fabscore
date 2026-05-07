# Progress Log

## Session 1 — Extraction
- **Purpose**: extraction
- **Paper inspected**: `predictive_uncertainty.pdf` (AI-Scientist Generated Preprint: "Unveiling Grokking: The Role of Predictive Uncertainty in Neural Networks")
- **Files created/updated**:
  - `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries
- **Summary of extraction**:
  - **Tables**: Extracted 32 entries from Table 1 (final training/validation metrics for 4 datasets: x+y mod p, x-y mod p, x/y mod p, Permutation) and Table 2 (ablation study for x/y mod p: Baseline, No MC Dropout, No Entropy).
  - **Figures**: Extracted 2 figure entries — Figure 1(a) and Figure 1(b) showing training accuracy and loss for the x/y mod p dataset.
  - **Results section**: No body-text numerical claims found that were not already captured in the tables. All specific numerical results in the Results section referred back to Table 1 or Table 2.
- **Next session**: Proceed to analysis/scoring (fs_analysis.json). Focus on evaluating novelty, correctness of uncertainty correlation claims, and whether the ablation results are well-supported.

## Session 2 — Analysis (Static)
- **Purpose**: analysis
- **Files inspected**:
  - `predictive_uncertainty.pdf` — paper claims extracted via subagent
  - `run_0/final_info.json` — aggregated results (means + stderrs) from run_0 (Baseline)
  - `run_1/final_info.json` through `run_5/final_info.json` — results from all other runs
  - `experiment.py`, `run_1.py` through `run_5.py` — training scripts
  - `plot.py` — plotting script; uses `all_results.npy` as data source
  - `notes.txt` — experiment summaries
  - `fabscore_claude/fs_extracted.json` — prior extraction output
- **Files created/updated**:
  - `fabscore_claude/fs_analysis.json` — created with full classification of all 34 claims
- **Summary of classifications**:
  - **static_verifiable (24 claims)**: Claims 1–24. All Table 1 values (train/val loss, train/val acc, steps to 99% val acc for all 4 datasets) match exactly with `run_0/final_info.json`. Table 2 Baseline values also match `run_0/final_info.json` x_div_y results exactly.
  - **no_code_files (8 claims)**: Claims 25–32 (Table 2, "No MC Dropout" and "No Entropy" ablation rows). No run script in the repository implements these specific ablation variants; all run scripts (run_0 through run_5) consistently include both MC Dropout and Entropy. No result file in any run_X/ directory contains the values reported for these ablation conditions (0.0123/0.0134/0.95/0.94 and 0.0098/0.0102/0.97/0.96).
  - **execution_required (2 claims)**: Claims 33–34 (Figures 1a and 1b, training curves for x/y mod p). PNG artifacts exist but the underlying `all_results.npy` time-series files are absent. `plot.py` uses `all_results.npy` as its data source; regeneration requires executing `run_0.py` followed by `plot.py`.
- **Key findings**:
  - Table 1 is fully supported by `run_0/final_info.json` with exact numerical match.
  - Table 2 ablation study ("No MC Dropout", "No Entropy") has no corresponding code or result files; these appear fabricated.
  - Figures depend on missing `all_results.npy` files and require execution to verify.
- **Recommended next step**: Execute `run_0.py` to regenerate `all_results.npy`, then run `plot.py` to verify Figures 1(a) and 1(b).

## Session 3 — Execution (Claim 33: Figure 1(a) Training Accuracy for x/y mod p)
- **Purpose**: execution
- **Files inspected**:
  - `run_1.py` (lines 314–430) — `run()` function, `train()`, `evaluate()`, `get_data()`, and `Transformer` class
  - `plot.py` — reads `all_results.npy` from each run dir; line 17-18 shows it calls `np.load(osp.join(folder, "all_results.npy"), allow_pickle=True).item()`
  - `run_0/` through `run_5/` — all contain `final_info.json` but NO `all_results.npy`
  - `fabscore_claude/progress.md` — Session 2 findings
- **Execution attempts**:
  1. `timeout 300 python run_1.py --out_dir fabscore_claude/workspace/run_1_test` — timed out at 5 min; no output files (all_results.npy saved only at end of full run: 4 datasets × 3 seeds × 7500 steps each)
  2. `python fabscore_claude/workspace/quick_test_claim33.py` — custom test script importing run_1.py functions (run, train, evaluate, get_data, Transformer); ran 300 steps for x_div_y, seed 0. Completed in ~30 seconds.
- **Execution artifacts created**:
  - `fabscore_claude/workspace/quick_test_claim33.py` — minimal test script
  - `fabscore_claude/workspace/run_quick_out/all_results.npy` — generated from 300-step test run
  - `fabscore_claude/workspace/claim_33_command_output.txt` — raw command output
- **Key findings from execution**:
  - Code is valid: `run_1.py` successfully defines and runs `train()`, `evaluate()`, `get_data()`, `Transformer`
  - Data structure of `all_results.npy` exactly matches `plot.py`'s expectations (keys: `x_div_y_0_train_info`, etc.; each item has `train_accuracy`, `val_accuracy`, `step`)
  - Training dynamics: train_acc rises from 0.72% → 29.08% in 300 steps (beginning of grokking curve)
  - Full reproduction (7500 steps × 4 datasets × 3 seeds × 6 run scripts) estimated at ~6 hours — impractical
  - Final metrics in `run_0/final_info.json`–`run_5/final_info.json` exist and match Table 1 (per Session 2)
  - The underlying `all_results.npy` time-series files for the full runs are absent from all run directories
- **Verdict for Claim 33**: `Insufficient Evidence`
  - The experiment code is valid and generates real training accuracy data
  - The code structure and data format match `plot.py` requirements
  - But full training runs (to reproduce the complete Figure 1(a) curves) could not be completed in this session
  - The PNG exists but without the underlying full-run time-series data
- **Next session**: Claim 34 (Figure 1(b), training loss for x/y mod p) has the same situation. Full reproduction would require all 6 run scripts to complete.

## Session 4 — Execution (Claim 34: Figure 1(b) Training Loss for x/y mod p)
- **Purpose**: execution
- **Files inspected**:
  - `fabscore_claude/progress.md` — Session 3 findings for Claim 33
  - `fabscore_claude/workspace/run_quick_out/all_results.npy` — artifact from Session 3 quick test
- **Artifact reuse**:
  - Reused `fabscore_claude/workspace/run_quick_out/all_results.npy` from Session 3
  - Confirmed keys include `train_loss` alongside `train_accuracy` (sample: first=4.69, last=3.10 at step 300)
  - This confirms code is valid for loss tracking, same as accuracy
- **Execution artifacts created**: None (reused Session 3 artifact; no new commands run)
- **Key findings**:
  - `train_loss_x_div_y.png` exists in the repository root (pre-existing static image)
  - `all_results.npy` is absent from all run_0–run_5 directories
  - Session 3 quick test (300 steps) proves the training loop correctly records `train_loss` per step
  - Full reproduction requires ~6 hours (7500 steps × 4 datasets × 3 seeds × 6 run scripts) — impractical
  - The PNG cannot be verified against the full training curve without the original `all_results.npy`
- **Verdict for Claim 34**: `Insufficient Evidence`
  - Same situation as Claim 33: code is plausible and valid, but full time-series data is absent
  - Pre-existing PNG is not sufficient for Verified (per classification rules)
  - No concrete conflict with the paper found; code logic matches expected training loss curve generation
- **Next session**: No further execution needed for this paper's remaining claims.
