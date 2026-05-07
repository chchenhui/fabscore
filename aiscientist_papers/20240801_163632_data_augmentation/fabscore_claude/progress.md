# Progress Log

## Session 1 — 2026-04-22
**Purpose:** extraction

**Paper inspected:** data_augmentation.pdf ("Augmenting Grokking: Unveiling the Effects of Data Augmentation on Neural Network Generalization")

**Context:** AI-Scientist generated paper studying the impact of data augmentation (random noise, shuffling) on the grokking phenomenon using modular arithmetic and permutation group datasets.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries

**Extraction summary:**
- **Tables:** 80 entries from Tables 1–5 (baseline + noise levels 0.1/0.2/0.3/0.4 across x_div_y, x_minus_y, x_plus_y, permutation datasets; columns: Final Train Loss, Final Val Loss, Final Train Acc, Final Val Acc)
- **Figures:** 1 entry — Figure 1 (training and validation loss curves for x_div_y across noise levels; subfigures 1a and 1b merged into one entry)
- **results_section:** Empty — all numerical results in the Results section body text were already captured in tables; remaining text claims were qualitative with no additional performance metrics

**Next session should:** Run analysis/scoring on the extracted results (fs_analysis.json or similar).

---

## Session 2 — 2026-04-22
**Purpose:** analysis (static verification of extracted claims)

**Files inspected:**
- `run_0/final_info.json` — baseline (no augmentation) results for all 4 datasets
- `run_1/final_info.json` — noise_level=0.1 results
- `run_2/final_info.json` — noise_level=0.2 results
- `run_3/final_info.json` — noise_level=0.3 results
- `run_4/final_info.json` — noise_level=0.4 results
- `experiment.py` — confirmed it saves `all_results.npy` per run (lines 426-427)
- `plot.py` — confirmed it requires `all_results.npy` for training curve generation
- Checked for `.npy` files — none found in repository

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with full classification of all 81 claims

**Classification summary:**
- **static_verifiable:** 80 (all table claims, indices 1–80)
  - Table 1 (claims 1–16) → `run_0/final_info.json` (baseline, no augmentation)
  - Table 2 (claims 17–32) → `run_1/final_info.json` (noise_level=0.1)
  - Table 3 (claims 33–48) → `run_2/final_info.json` (noise_level=0.2)
  - Table 4 (claims 49–64) → `run_3/final_info.json` (noise_level=0.3)
  - Table 5 (claims 65–80) → `run_4/final_info.json` (noise_level=0.4)
  - All 80 values match exactly (to 4 decimal rounding) their corresponding `final_*_mean` fields
- **execution_required:** 1 (Figure 1, index 81)
  - PNG artifacts exist but `all_results.npy` (needed for training trajectory curves) is absent; experiment.py can regenerate it

**Next session should:** Execute experiment runs to regenerate `all_results.npy` files and verify Figure 1 using `plot.py`. Entrypoints: `python experiment.py` (baseline) and `python run_1.py` through `run_4.py` for noise-level runs, then `python plot.py`.

---

## Session 3 — 2026-04-22
**Purpose:** execution (Claim 81: Figure 1 training/validation loss curves for x_div_y)

**Files/context inspected:**
- `run_0/` through `run_5/`: all have `final_info.json` but NO `all_results.npy`
- `experiment.py`: current version has `augment=True, noise_level=0.5` in __main__ (appears to be run_5 code)
- `run_1.py`: __main__ calls `run()` WITHOUT augment/noise_level args (baseline, no noise) — despite being labeled "Noise Level 0.1" in plot.py
- `run_2.py` through `run_5.py`: correctly pass `augment=True, noise_level=0.N`
- All run_*.py default to `--out_dir run_0` (would overwrite run_0 if run without --out_dir arg)
- `run_1/final_info_x_div_y_2.json`: seed 2 has catastrophic val_loss=2.87 (anomalous vs seeds 0&1 at ~0.008)
- `plot.py`: requires all_results.npy from each run dir to generate Figure 1

**Execution artifacts created:**
- `fabscore_claude/workspace/claim_81_command_output.txt`: stdout/stderr of runs
- `fabscore_claude/workspace/verify_run0/`, `verify_run1/`: partial training results
- `fabscore_claude/workspace/verify_trajectories.npy`: 750-step trajectories for x_div_y (baseline vs noise=0.1, 1 seed each)
  - Baseline final val_loss: 0.0085, val_acc: 1.0
  - Noise_level=0.1 final val_loss: 0.0508, val_acc: 0.999

**Key findings:**
1. Experiment framework runs correctly; different noise levels produce different trajectories (verified)
2. all_results.npy is missing everywhere — cannot reproduce Figure 1 exactly
3. run_1.py has a code bug: doesn't pass noise_level=0.1, so running it would produce baseline results, not noise=0.1
4. run_1/final_info shows anomalous results (seed 2 catastrophically fails), inconsistent with both baseline and noise_level=0.1 from fresh experiments
5. Timing: ~110s per training run; full Figure 1 reproduction would need 18+ runs (~33 min)

**Verdict summary for claim 81:** `Insufficient Evidence`
- Code infrastructure is plausible and functional; different noise levels produce distinct loss curves
- But all_results.npy (the training trajectory data for Figure 1) is absent from all run directories
- run_1.py has a bug (no noise level passed) and run_1/ shows anomalous results
- Cannot fully verify Figure 1's exact curves without complete trajectory data

**Next session:** No further action needed for this claim; verdict is Insufficient Evidence.
