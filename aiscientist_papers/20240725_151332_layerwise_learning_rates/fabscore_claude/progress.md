# Progress Log

## Session: 2026-04-22
- **Purpose**: extraction
- **Paper inspected**: `layerwise_learning_rates.pdf` — "Layer-wise Learning Rate Decay in Transformers: Enhancing Training Dynamics" (AI Scientist generated preprint)
- **JSON files created/updated**: `fabscore_claude/fs_extracted.json`
- **Summary**:
  - Extracted 60 table entries from Tables 1–5 (baseline, linear decay, exponential decay, step decay, polynomial decay results across shakespeare_char, enwik8, and text8 datasets).
  - Extracted 2 figures (Figure 1: training loss curves, Figure 2: validation loss curves).
  - No results_section entries: all numerical results in the Results section body text are already captured in Tables 1–5; remaining text is qualitative without standalone numerical performance metrics.
- **Next session**: No further extraction needed. Could verify or score against ground truth if available.

## Session: 2026-04-22 (analysis)
- **Purpose**: analysis
- **Files inspected**:
  - `layerwise_learning_rates.pdf` (paper)
  - `run_0/final_info.json` through `run_5/final_info.json` (all result files)
  - `plot.py` (plotting script)
  - `experiment.py`, `run_0.py`–`run_5.py` (training scripts)
  - `notes.txt` (experiment plan)
  - Repository structure (no .npy or .csv files found)
- **JSON files created/updated**: `fabscore_claude/fs_analysis.json`
- **Summary of classifications**:
  - **static_verifiable (60 claims)**: All 60 table claims (Tables 1–5, claims 1–60) verified by direct comparison against `run_*/final_info.json`. Every paper value matched the corresponding JSON mean to 4 decimal places. Mapping: Table 1 → run_0 (Baseline), Table 2 → run_1 (Linear Decay), Table 3 → run_2 (Exponential Decay), Table 4 → run_3 (Step Decay), Table 5 → run_4 (Polynomial Decay).
  - **execution_required (2 claims)**: Figures 1 and 2 (training and validation loss curves over iterations). PNG files exist but `all_results.npy` files needed by `plot.py` are absent from the repository. Training scripts (run_*.py) can regenerate them.
  - No obvious_hallucination, no_code_files, insufficient_evidence, or error cases found.
- **Next session**: Run training scripts (run_0.py through run_5.py) to regenerate `all_results.npy`, then execute `plot.py` to verify Figures 1 and 2 match the paper.

## Session: 2026-04-22 (execution - claim 61)
- **Purpose**: execution verification of claim 61 (Figure 1: training loss over iterations)
- **Files inspected**:
  - `plot.py` — requires `all_results.npy` from each run directory
  - `run_1.py` — training script; generates all_results.npy at completion
  - `run_0/final_info_enwik8_0.json` — enwik8 training took 819s per seed
  - `run_0/final_info_shakespeare_char_0.json` — shakespeare_char took 78s per seed
  - Confirmed no `all_results.npy` files exist in any run directory
- **Execution**:
  - Ran `timeout 90 python run_1.py --out_dir fabscore_claude/workspace/run_1_test` — script started training successfully with decreasing loss (iter 0: 4.27, iter 250: val 2.39, iter 750: val 1.82), confirming code pipeline is functional.
  - Process timed out after 90s with ~850 iterations of shakespeare_char completed.
  - No `all_results.npy` generated (only saved at end of complete training).
  - Full run would require ~31 min (3 seeds shakespeare + 1 enwik8 + 1 text8 = ~1872s).
- **Artifacts**: `fabscore_claude/workspace/claim_61_command_output.txt`
- **Verdict summary**: `Insufficient Evidence` — Training code is plausible and runs correctly. Pre-existing PNG files (train_loss_shakespeare_char.png etc.) exist in repo. But `all_results.npy` intermediate files are missing, and full regeneration exceeds feasible timeout (~31 min per run × 5 runs). Per classification rules: "A plotting script is plausible, but the required generated input such as `all_results.npy` is missing and reasonable repo-native attempts still do not produce enough evidence to verify the plotted claim => Insufficient Evidence."
- **Next session**: If more time is available, run a full training script (e.g., `python run_1.py --out_dir run_1`) and then `python plot.py` to verify Figure 1 reproduction. This would take ~31 min per run.

## Session: 2026-04-22 (execution - claim 62)
- **Purpose**: execution verification of claim 62 (Figure 2: validation loss over iterations)
- **Files inspected**:
  - `plot.py` — Plot 2 (lines 80–97) generates val_loss_{dataset}.png from `all_results.npy`, extracting `val/loss` keys from each run directory
  - `fabscore_claude/progress.md` — Prior session (claim 61) already confirmed no `.npy` files exist and training takes ~31 min per run
  - Confirmed no `.npy` files exist anywhere in the repository (Glob search returned empty)
- **Execution**: No new commands executed — situation is identical to claim 61. The same `all_results.npy` files are required by `plot.py` for both Figure 1 (train_loss) and Figure 2 (val_loss). Prior session already established the training pipeline is functional but full regeneration is infeasible within reasonable timeout.
- **Artifacts**: No new artifacts created (reusing findings from claim 61 session)
- **Verdict summary**: `Insufficient Evidence` — identical reasoning to claim 61. `plot.py` has plausible validation loss plotting code (lines 80–97). Pre-existing PNG files (val_loss_shakespeare_char.png, val_loss_enwik8.png, val_loss_text8.png) exist but are insufficient per classification rules. `all_results.npy` files are missing; full regeneration would take ~31 min × 5 runs ≈ 2.5+ hours.
- **Next session**: No further action needed for claims 61 and 62 unless extended compute time is available.
