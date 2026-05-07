# Progress Log

---

## Session: 2026-04-22

**Purpose:** extraction

**Paper inspected:** adaptive_computation_reduction.pdf
- Title: "Efficient Transformers: Skipping Redundant Tokens for Faster Computation"
- AI Scientist autonomously generated preprint
- 8 pages, evaluated on shakespeare_char, enwik8, text8 datasets

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created new file with tables, figures, and results_section entries

**Extraction summary:**
- Tables: 24 entries extracted from Table 1 (Baseline Results) and Table 2 (Results with Dynamic Redundancy Threshold), covering Final Train Loss, Best Val Loss, Total Train Time (min), and Avg Inference Tokens/s for all 3 datasets
- Figures: 3 figures extracted (Figure 1: enwik8 loss curves, Figure 2: text8 loss curves, Figure 3: shakespeare_char loss curves)
- Results section: 0 entries — all numerical claims in Sections 6.1 and 6.2 body text are direct restatements of Table 1 and Table 2 values; Section 6.3 and 6.4 contain only qualitative comparisons with no new numerical values

**Next session should:**
- No further extraction needed; paper has been fully processed.
- If scoring or comparison tasks are requested, refer to fs_extracted.json for table values.

---

## Session: 2026-04-22 (Analysis)

**Purpose:** analysis

**Files inspected:**
- `latex/template.tex` — full LaTeX source confirming Table 1 = Baseline Results (run_0), Table 2 = Dynamic Redundancy Threshold (run_4); Figures 1-3 = loss curve PNGs per dataset
- `run_0/final_info.json` — baseline run metrics
- `run_1/final_info.json` — Low Redundancy Threshold (0.1)
- `run_2/final_info.json` — Medium Redundancy Threshold (0.5)
- `run_3/final_info.json` — High Redundancy Threshold (0.9)
- `run_4/final_info.json` — Dynamic Redundancy Threshold
- `run_5/final_info.json` — Control (No Redundancy Detection)
- `plot.py` — plotting script; requires `all_results.npy` per run directory (absent)

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created new file with full analysis of all 27 claims

**Classification summary:**
- `static_verifiable`: 24 (claims 1–24, all table claims)
  - Table 1 (claims 1–12): All 12 values match `run_0/final_info.json` means when rounded to 4 decimal places
  - Table 2 (claims 13–24): All 12 values match `run_4/final_info.json` means when rounded to 4 decimal places
  - plot.py labels confirm run_0="Baseline" and run_4="Dynamic Redundancy Threshold"
- `execution_required`: 3 (claims 25–27, all figure claims)
  - PNG artifacts exist (train_loss_*.png, val_loss_*.png) but underlying `all_results.npy` files are absent from all run directories
  - plot.py requires these .npy files to generate figures; run_*.py scripts exist as potential regeneration path

**Next session should:**
- Run training scripts (run_0.py through run_5.py) to regenerate all_results.npy, then run plot.py to regenerate figures and verify claims 25–27.

---

## Session: 2026-04-22 (Execution — Claim 25)

**Purpose:** execution

**Claim:** Figure 1: Training and validation loss curves for the enwik8 dataset.

**Files/context inspected:**
- `plot.py` — requires `all_results.npy` from each run directory; loads `val_info` and `train_info` per-iteration entries (eval_interval=1000 → 100 points for enwik8 at 100k iters)
- `experiment.py`, `run_1.py`–`run_5.py` — training scripts; produce `all_results.npy` + `final_info_enwik8_0.json`
- `run_{0–5}/final_info_enwik8_0.json` — exist for all 6 runs; contain final_train_loss, best_val_loss, total_train_time, avg_inference_tokens_per_second
- `run_0.py` — absent (no baseline script; experiment.py is effectively the baseline)
- `all_results.npy` — absent from ALL run directories
- `train_loss_enwik8.png`, `val_loss_enwik8.png` — pre-existing PNGs; insufficient alone for Verified
- GPU check: H200 GPUs available; enwik8 training ~820s per run × 6 runs × 3 datasets ≈ multi-hour job
- enwik8 dataset at `/home/chenhui/data/enwik8/` confirmed present

**Execution artifacts created:** None (no commands run; regeneration infeasible in session due to multi-hour total training time)

**Verdict summary (claim 25):** `Insufficient Evidence`
- All 6 `final_info_enwik8_0.json` files confirm training happened with realistic loss values and ~820–850 s train times, supporting that the original training was genuine
- `all_results.npy` (per-iteration curve data) absent from all run directories; plot.py cannot run without it
- Full reproduction requires hours of GPU training (6 runs × ~40 min each for all 3 datasets); too difficult to reproduce in this session
- Pre-existing PNGs insufficient for Verified per rules

**Next session should:**
- If regeneration is attempted: run `python run_1.py --out_dir workspace/test_run_1` etc. from repo root (GPU required, ~40 min/run)
- Otherwise accept `Insufficient Evidence` for all three figure claims (25–27)

---

## Session: 2026-04-22 (Execution — Claim 26)

**Purpose:** execution

**Claim:** Figure 2: Training and validation loss curves for the text8 dataset.

**Files/context inspected:**
- `progress.md` — prior sessions confirmed all_results.npy absent from all run directories; final_info_text8_0.json present for all 6 runs
- `run_*/final_info_text8_0.json` — confirmed present for runs 0–5; contain final_train_loss, best_val_loss, total_train_time, avg_inference_tokens_per_second
- `all_results.npy` — absent from ALL run directories (confirmed by ls exit code 2)
- `train_loss_text8.png`, `val_loss_text8.png` — pre-existing PNGs; insufficient alone for Verified
- Situation identical to claim 25 (enwik8); full regeneration requires ~40 min/run × 6 runs GPU training

**Execution artifacts created:** None (no commands run; regeneration infeasible due to multi-hour training time)

**Verdict summary (claim 26):** `Insufficient Evidence`
- All 6 `final_info_text8_0.json` files confirm training ran with realistic loss values; original training was genuine
- `all_results.npy` (per-iteration curve data) absent from all run directories; plot.py cannot generate figures without it
- Pre-existing PNGs insufficient for Verified per rules
- Identical situation to claim 25 (enwik8)

**Next session should:**
- Accept `Insufficient Evidence` for claim 27 (shakespeare_char) — same situation applies

---

## Session: 2026-04-22 (Execution — Claim 27)

**Purpose:** execution

**Claim:** Figure 3: Training and validation loss curves for the shakespeare_char dataset.

**Files/context inspected:**
- `progress.md` — prior sessions confirmed all_results.npy absent from all run directories
- `run_*/final_info_shakespeare_char_*.json` — present for runs 0–5; contain only summary metrics (final_train_loss, best_val_loss, total_train_time, avg_inference_tokens_per_second), not per-iteration curves
- `all_results.npy` — absent from ALL run directories (confirmed by prior sessions + find command returning no results)
- `train_loss_shakespeare_char.png`, `val_loss_shakespeare_char.png` — pre-existing PNGs; insufficient alone for Verified
- Situation identical to claims 25 (enwik8) and 26 (text8)

**Execution artifacts created:** None (no commands run; regeneration would require ~40 min/run × 6 runs GPU training)

**Verdict summary (claim 27):** `Insufficient Evidence`
- All 6 run directories have `final_info_shakespeare_char_*.json` files confirming training ran with realistic loss values
- `all_results.npy` (per-iteration curve data) absent from all run directories; plot.py cannot generate figures without it
- Pre-existing PNGs insufficient for Verified per rules; no raw data files or logs with per-iteration loss values found
- Full regeneration feasible in principle (GPU available) but requires multi-hour training; no concrete conflict with paper established

**Next session should:**
- No further action needed for figure claims 25–27; all three are `Insufficient Evidence`.
