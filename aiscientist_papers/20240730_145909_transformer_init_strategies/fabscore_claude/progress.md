# Progress Log

## Session: 2026-04-22 — extraction

**Purpose:** Extract experimental results from the paper.

**Paper inspected:** `transformer_init_strategies.pdf` (5 pages, AI-Scientist generated preprint on novel initialization strategies for transformer-based language models)

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries

**Summary of extraction:**
- **Tables:** 24 entries from Table 1 (Comparison of Initialization Methods: Xavier/Glorot, Kaiming/He, Depth-Adaptive Scaling, Task-Specific Priors, Full Strategy — perplexity, F1 score, convergence speed) and Table 2 (Ablation Study Results: No Depth-Adaptive Scaling, No Task-Specific Priors, Full Strategy).
- **Figures:** 2 entries — Figure 1(a) Validation Loss on enwik8, Figure 1(b) Training Loss on enwik8.
- **results_section:** Empty — all numerical claims in Section 6 body text refer directly to Table 1 and Table 2 values; no standalone numerical results found in the text body that were not already captured in the tables.

**Next session should:** No further extraction needed. Could perform scoring/evaluation against ground truth if available.

---

## Session: 2026-04-22 — analysis

**Purpose:** Static analysis of all 26 claims against the repository.

**Files inspected:**
- `transformer_init_strategies.pdf` (all 5 pages)
- `run_0/final_info.json` through `run_4/final_info.json`
- `run_1.py` (representative run script; experiment.py is similar)
- `plot.py`
- All run directories (confirmed only `final_info*.json` files present; no `all_results.npy`)

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with full classification of all 26 claims

**Key findings:**
1. **Critical metric mismatch (experiment_fabrication):** The paper claims to report Perplexity, F1 Score, and Convergence Speed (in epochs). The repository code tracks ONLY: `final_train_loss`, `best_val_loss`, `total_train_time` (seconds), and `avg_inference_tokens_per_second`. No F1 score, no perplexity computation, and no epoch-based convergence tracking exist anywhere in any Python script.
2. **Missing ablation/combined runs:** The paper references a "Full Strategy" (runs 0-4 are Baseline, Xavier/Glorot, Kaiming/He, Depth-Adaptive, Task-Specific). No combined run exists. Similarly, Table 2 ablation runs ("No Depth-Adaptive Scaling", "No Task-Specific Priors") are not implemented.
3. **Perplexity value mismatch:** Even if derived as exp(best_val_loss), actual enwik8 perplexities would be ~2.73 for all methods (val_loss ~1.005). The paper reports a range from 3.45 to 2.75 — a spread that does not match the actual near-identical values.
4. **Actual results contradict paper narrative:** On enwik8, run_3 (Depth-Adaptive) and run_4 (Task-Specific) have slightly *higher* val_loss than run_1 (Xavier/Glorot), contradicting the paper's claim of improvements.
5. **`all_results.npy` missing:** Required by `plot.py` to regenerate the figures. Existing PNG files (val_loss_enwik8.png, train_loss_enwik8.png) cannot be statically verified without this data.

**Classification summary:**
- `obvious_hallucination` (experiment_fabrication): 24 claims (all Table 1 and Table 2 entries — indices 1-24)
- `execution_required`: 2 claims (Figure 1a and 1b — indices 25-26)

**Recommended next step:** For claims 25-26 (figures), execution would need to re-run training scripts to regenerate `all_results.npy`, then run `plot.py`. However, given the metric fabrication found in all table claims, the figures are also expected to show the same near-identical loss curves across methods (contradicting the paper's narrative of significant improvements).

---

## Session: 2026-04-22 — execution (claim 25)

**Purpose:** Verify claim 25 — Figure 1(a): Validation Loss on 'enwik8'.

**Files inspected:**
- `run_0/` through `run_4/final_info_enwik8_0.json` — actual best_val_loss metrics from training
- `run_1.py` (representative run script) — confirmed no `--dataset` flag; runs all three datasets sequentially
- `plot.py` — confirmed it requires `all_results.npy` with per-iteration data
- `val_loss_enwik8.png` (pre-existing image) — visually inspected; shows all 5 methods with completely overlapping curves

**Key findings:**
1. `all_results.npy` is MISSING from all run directories (run_0 through run_4)
2. `val_loss_enwik8.png` EXISTS as a pre-existing image
3. The image shows all 5 methods (Baseline, Xavier/Glorot, Kaiming/He, Depth-Adaptive, Task-Specific) with nearly identical validation loss curves — all lines overlap completely
4. Actual final best_val_loss from `final_info_enwik8_0.json`: run_0=1.0055, run_1=1.0048, run_2=1.0048, run_3=1.0053, run_4=1.0054 — essentially identical
5. Suggested entrypoint `python run_0.py --dataset enwik8` is INVALID — run scripts have no `--dataset` argument
6. Re-running full training requires 100,000 iterations for enwik8 × 3 datasets × ~800s per dataset per run — infeasible

**No commands were executed** (pre-existing artifacts inspected only).

**Verdict summary:** `Insufficient Evidence`
- Pre-existing PNG cannot satisfy `Verified` per rules (need underlying data)
- `all_results.npy` (per-iteration data for line plots) is missing
- Full re-training is infeasible
- The actual final_info data and the pre-existing image both show near-identical curves across all methods

**Next session:** Claim 26 (Figure 1(b): Training Loss on enwik8) can be assessed similarly — `train_loss_enwik8.png` exists but `all_results.npy` is missing, so same verdict applies.

---

## Session: 2026-04-22 — execution (claim 26)

**Purpose:** Verify claim 26 — Figure 1(b): Training Loss on 'enwik8'.

**Files inspected:**
- `run_0/` through `run_4/final_info_enwik8_0.json` — actual final_train_loss metrics from training
- `train_loss_enwik8.png` (pre-existing image) — visually inspected; shows all 5 methods with completely overlapping curves
- `latex/template.tex` — confirmed Figure 1(b) caption is "Training Loss on `enwik8`", included as part of "Loss curves comparing different initialization methods"

**Commands executed:**
- `cat run_*/final_info_enwik8_0.json` — to get actual final_train_loss values

**Key findings:**
1. Actual final_train_loss from `final_info_enwik8_0.json`:
   - run_0 (Baseline): 0.9302
   - run_1 (Xavier/Glorot): 0.9284
   - run_2 (Kaiming/He): 0.9256
   - run_3 (Depth-Adaptive): 0.9367 — WORSE than baseline
   - run_4 (Task-Specific): 0.9404 — WORSE than baseline
2. Pre-existing `train_loss_enwik8.png` shows all curves completely overlapping — consistent with near-identical actual values
3. `all_results.npy` is MISSING (per-iteration training data for the line plot)
4. Paper claims "significant improvements" for Depth-Adaptive and Task-Specific methods (Table 1: better perplexity, F1 score, convergence speed)
5. Actual training loss data CONTRADICTS the paper's narrative — novel methods perform worse than baseline

**Verdict summary:** `Result Fabrication`
- Final_info JSON files confirm run_3 and run_4 have HIGHER training loss than baseline (worse)
- This directly contradicts the paper's narrative of improvements
- The pre-existing image confirms overlapping curves (actual data plotted, just doesn't show improvements claimed)
- The paper presents this figure as supporting "significant improvements" which is unsupported by the actual experimental results

**No further sessions needed** for this claim.
