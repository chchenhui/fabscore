# Progress Log

## Session: 2026-04-22

**Purpose:** extraction

**Paper inspected:** targeted_regularization.pdf
- Title: "Precision Regularization: Tailored Strategies for Optimized Transformer Training"
- 8 pages, AI Scientist generated preprint
- Evaluated targeted vs. uniform regularization for Transformer models on shakespeare_char, enwik8, text8

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created fresh; contains:
  - 60 table entries from Tables 1–5 (baseline, differential regularization, hyperparameter search for regularization strengths, learning rate, and dropout rate)
  - 6 figure entries (Figures 1a, 1b, 2a, 2b, 3a, 3b — training and validation loss curves for each dataset)
  - 0 results_section entries — the body text of the Results section contains no standalone numerical performance claims beyond what is already captured in tables

**Notes:**
- The Results section body text consists almost entirely of references to tables and figures with no additional numerical claims.
- Section 6.5 (Discussion) and 6.6 (Limitations) are qualitative only.
- Table 5 (dropout hyperparameter search) appears on page 6; its caption is on the same page as Figure 1.

**Next session should:** No further extraction needed. If scoring/comparison is required, compare fs_extracted.json against a reference or run fabscore evaluation.

---

## Session: 2026-04-22 (analysis)

**Purpose:** analysis

**Files inspected:**
- `targeted_regularization.pdf` (via extraction session notes)
- `run_0/final_info.json` — baseline results for Tables 1
- `run_1/final_info.json` — differential regularization results for Table 2
- `run_2/final_info.json` — hyperparameter search (reg strengths) results for Table 3
- `run_3/final_info.json` — hyperparameter search (LR) results for Table 4
- `run_4/final_info.json` — hyperparameter search (dropout) results for Table 5
- `plot.py` — plotting script that loads all_results.npy and final_info.json
- Glob for all_results.npy (confirmed ABSENT from all run directories)

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created fresh; contains full classification of all 66 claims

**Classification summary:**
- `static_verifiable`: 60 (all table claims, indices 1–60)
  - All values in Tables 1–5 match exactly (to 4 decimal places after rounding) the `final_train_loss_mean`, `best_val_loss_mean`, `total_train_time_mean`, and `avg_inference_tokens_per_second_mean` fields from the corresponding `run_*/final_info.json` files.
  - Table 1 → run_0, Table 2 → run_1, Table 3 → run_2, Table 4 → run_3, Table 5 → run_4.
- `execution_required`: 6 (all figure claims, indices 61–66)
  - PNG files for all 6 figures exist but `all_results.npy` (required by `plot.py` to draw loss curves) is absent from all run directories.
  - Training scripts (`experiment.py`, `run_1.py`–`run_4.py`) can regenerate `all_results.npy`, after which `plot.py` can verify the figures.
- `no_code_files`: 0
- `obvious_hallucination`: 0
- `insufficient_evidence`: 0
- `error`: 0

**Next step for next session:** Execute training to regenerate `all_results.npy` and run `plot.py` to verify figure claims (indices 61–66). Note: training is very long (800+ min/dataset), so execution may need to be abbreviated or a cached checkpoint used.

---

## Session: 2026-04-22 (execution - claim 61)

**Purpose:** execution — verify Claim 61: Figure 1(a) Training loss for shakespeare_char dataset

**Files inspected:**
- `fabscore_claude/progress.md` — prior session notes
- All `run_*/` directories — confirmed `all_results.npy` absent from all runs
- `experiment.py` lines 540–739 — training loop, how val_log_info/train_log_info are collected and written to all_results.npy at the very end of all 3 datasets
- `plot.py` — requires `all_results.npy` from run_0 through run_4 for all 5 curves
- `run_0/final_info_shakespeare_char_*.json` — only summary stats (no trajectories)
- `../../../data/shakespeare_char/` — confirmed train.bin, val.bin, meta.pkl exist

**Execution artifacts created:**
- `fabscore_claude/workspace/run_0/final_info_shakespeare_char_{0,1,2}.json` — generated fresh
- `fabscore_claude/workspace/claim_61_command_output.txt` — stdout from `python experiment.py --out_dir fabscore_claude/workspace/run_0` (killed after shakespeare_char completed, before enwik8)

**Key findings:**
- Training code is functional; shakespeare_char (3 seeds × 5000 iters) completed successfully in ~5 min on H200
- Training loss trajectory for baseline (run_0): step 0 → ~4.2, step 5000 → ~0.64–0.95 (across 3 seeds), clearly decreasing
- `all_results.npy` was NOT written because it is only saved after ALL 3 datasets complete; enwik8 (100K iters) was killed
- Only run_0 was attempted; Figure 1(a) requires all 5 runs (run_0–run_4), each with 3 datasets
- Full reproduction requires ~5× full experiments with 100K iters each for enwik8/text8 = impractical

**Verdict for claim 61:** `Insufficient Evidence`
- Partial execution confirmed code path is valid and training loss curves decrease as expected
- Cannot fully reproduce Figure 1(a) without all 5 runs completing across all 3 datasets
- all_results.npy absent; plot.py cannot be run to regenerate the figure

**Next session should:** For remaining figure claims (62–66), same situation applies — training code is valid, trajectories are plausible, but full reproduction of all 5 run configs is infeasible. Apply `Insufficient Evidence` to all figure claims (61–66) unless a faster reproduction path is found.

---

## Session: 2026-04-22 (execution - claim 62)

**Purpose:** execution — verify Claim 62: Figure 1(b) Validation loss for shakespeare_char dataset

**Files inspected:**
- `fabscore_claude/progress.md` — reviewed prior session notes (especially claim 61 session)
- `fabscore_claude/workspace/claim_61_command_output.txt` — reused; contains validation loss trajectory data for shakespeare_char (run_0/baseline) from prior run
- `fabscore_claude/workspace/run_0/final_info_shakespeare_char_{0,1,2}.json` — only summary stats, no trajectories
- All run_*/all_results.npy — confirmed ABSENT (checked via glob)
- `plot.py` line 17 — confirmed it loads `all_results.npy` to generate val_loss_shakespeare_char.png
- `val_loss_shakespeare_char.png` — pre-existing PNG exists but is a manually uploaded artifact

**Execution artifacts created/reused:**
- Reused `fabscore_claude/workspace/claim_61_command_output.txt` — contains valid validation loss trajectory data from run_0 shakespeare_char training (3 seeds)
- No new command run; prior session data is sufficient to establish the code path is valid

**Key findings:**
- Validation loss trajectory data from claim_61 session shows: step 0 → ~4.2, decreasing to best ~1.45 around steps 2000-3000, then increasing slightly (overfitting pattern)
- This is consistent with what Figure 1(b) would show for the baseline run
- `all_results.npy` still absent from all run directories; `plot.py` cannot regenerate the figure
- Full reproduction requires all 5 run configs × all 3 datasets (including 100K-iter enwik8/text8) — impractical
- The pre-existing PNG is not sufficient for `Verified` per classification rules

**Verdict for claim 62:** `Insufficient Evidence`
- Training code is functional and produces valid val loss trajectories consistent with the figure's expected content
- Cannot fully reproduce Figure 1(b) without all 5 runs completing all datasets
- all_results.npy absent; plot.py cannot be run to regenerate the figure
- Same situation as claim 61

**Next session should:** Apply `Insufficient Evidence` to remaining figure claims (63–66) for the same reasons. All figure claims share the same blocker: all_results.npy missing from all run directories and full reproduction is impractical.

---

## Session: 2026-04-22 (execution - claim 63)

**Purpose:** execution — verify Claim 63: Figure 2(a) Training loss for enwik8 dataset

**Files inspected:**
- `fabscore_claude/progress.md` — reviewed prior session notes (claims 61 and 62)
- `run_*/all_results.npy` — confirmed ABSENT from all run directories
- `run_*/final_info_enwik8_0.json` — only final summary stats (single seed), no training trajectories
- `plot.py` lines 1–77 — confirmed it requires `all_results.npy` to generate `train_loss_enwik8.png`
- `train_loss_enwik8.png` — pre-existing PNG exists but is a manually uploaded artifact

**Execution artifacts created/reused:**
- No new command run; prior session data is sufficient to establish the same blocker
- Reused prior session findings (claims 61–62) confirming code path is valid but enwik8 requires 100K iterations

**Key findings:**
- Figure 2(a) is generated by plot.py (lines 61–77) which loads training loss trajectories from `all_results.npy`
- `all_results.npy` is absent from all run directories (run_0 through run_5)
- Only `final_info_enwik8_0.json` exists per run dir (single seed, summary stats only — no step-by-step trajectories)
- Training enwik8 to completion requires 100K iterations × 5 run configs — impractical
- Pre-existing PNG `train_loss_enwik8.png` is not sufficient for Verified per classification rules

**Verdict for claim 63:** `Insufficient Evidence`
- Code path is plausible and experiment.py is functional (confirmed in claim 61 session)
- Cannot fully reproduce Figure 2(a) without all 5 runs completing enwik8 to 100K iterations
- `all_results.npy` absent; plot.py cannot be run to regenerate the figure
- Same situation as claims 61 and 62

**Next session should:** Apply `Insufficient Evidence` to remaining figure claims (64–66) for the same reasons.

---

## Session: 2026-04-22 (execution - claim 64)

**Purpose:** execution — verify Claim 64: Figure 2(b) Validation loss for enwik8 dataset

**Files inspected:**
- `fabscore_claude/progress.md` — reviewed prior session notes (claims 61–63)
- `run_*/all_results.npy` — confirmed ABSENT from all run directories (ls check)
- `fabscore_claude/workspace/` — contains claim_61_command_output.txt, claim_61_run0_output.txt, run_0/ (from prior sessions)
- `val_loss_enwik8.png` — pre-existing PNG exists but is a manually uploaded artifact

**Execution artifacts created/reused:**
- No new command run; prior session data is sufficient to establish the same blocker
- Reused prior session findings confirming code path is valid but enwik8 requires 100K iterations

**Key findings:**
- Figure 2(b) (validation loss for enwik8) is generated by plot.py which loads trajectories from `all_results.npy`
- `all_results.npy` is absent from all run directories (run_0 through run_5)
- Training enwik8 to completion requires 100K iterations × 5 run configs — impractical
- Pre-existing PNG `val_loss_enwik8.png` is not sufficient for Verified per classification rules

**Verdict for claim 64:** `Insufficient Evidence`
- Code path is plausible and experiment.py is functional (confirmed in claim 61 session)
- Cannot fully reproduce Figure 2(b) without all 5 runs completing enwik8 to 100K iterations
- `all_results.npy` absent; plot.py cannot be run to regenerate the figure
- Same situation as claims 61, 62, and 63

**Next session should:** Apply `Insufficient Evidence` to remaining figure claims (65–66) for the same reasons.

---

## Session: 2026-04-22 (execution - claim 65)

**Purpose:** execution — verify Claim 65: Figure 3(a) Training loss for text8 dataset

**Files inspected:**
- `fabscore_claude/progress.md` — reviewed prior session notes (claims 61–64)
- `run_*/all_results.npy` — confirmed ABSENT from all run directories (established in prior sessions)
- `train_loss_text8.png` — pre-existing PNG exists but is a manually uploaded artifact
- `plot.py` — confirmed it requires `all_results.npy` to generate `train_loss_text8.png`

**Execution artifacts created/reused:**
- No new command run; prior session data is sufficient to establish the same blocker
- Reused prior session findings confirming code path is valid but text8 requires 100K iterations (same as enwik8)

**Key findings:**
- Figure 3(a) (training loss for text8) is generated by plot.py which loads trajectories from `all_results.npy`
- `all_results.npy` is absent from all run directories (run_0 through run_5)
- Training text8 to completion requires 100K iterations × 5 run configs — impractical
- Pre-existing PNG `train_loss_text8.png` is not sufficient for Verified per classification rules

**Verdict for claim 65:** `Insufficient Evidence`
- Code path is plausible and experiment.py is functional (confirmed in claim 61 session)
- Cannot fully reproduce Figure 3(a) without all 5 runs completing text8 to 100K iterations
- `all_results.npy` absent; plot.py cannot be run to regenerate the figure
- Same situation as claims 61, 62, 63, and 64

**Next session should:** Apply `Insufficient Evidence` to remaining figure claim (66) for the same reasons.

---

## Session: 2026-04-22 (execution - claim 66)

**Purpose:** execution — verify Claim 66: Figure 3(b) Validation loss for text8 dataset

**Files inspected:**
- `fabscore_claude/progress.md` — reviewed prior session notes (claims 61–65)
- `run_*/all_results.npy` — confirmed ABSENT from all run directories (established in prior sessions)
- `val_loss_text8.png` — pre-existing PNG exists but is a manually uploaded artifact
- `plot.py` — confirmed it requires `all_results.npy` to generate `val_loss_text8.png`

**Execution artifacts created/reused:**
- No new command run; prior session data is sufficient to establish the same blocker
- Reused prior session findings confirming code path is valid but text8 requires 100K iterations (same as enwik8)

**Key findings:**
- Figure 3(b) (validation loss for text8) is generated by plot.py which loads trajectories from `all_results.npy`
- `all_results.npy` is absent from all run directories (run_0 through run_5)
- Training text8 to completion requires 100K iterations × 5 run configs — impractical
- Pre-existing PNG `val_loss_text8.png` is not sufficient for Verified per classification rules

**Verdict for claim 66:** `Insufficient Evidence`
- Code path is plausible and experiment.py is functional (confirmed in claim 61 session)
- Cannot fully reproduce Figure 3(b) without all 5 runs completing text8 to 100K iterations
- `all_results.npy` absent; plot.py cannot be run to regenerate the figure
- Same situation as claims 61, 62, 63, 64, and 65

**Next session should:** No further figure claims remain (all 6 done). All figure claims (61–66) were classified as `Insufficient Evidence` due to missing `all_results.npy` and impractical full reproduction.
