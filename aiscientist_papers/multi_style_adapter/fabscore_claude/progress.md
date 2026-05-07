# Progress Log

## Session: extraction (2026-04-04)

**Purpose:** Extract experimental results from the paper.

**Paper inspected:** `multi_style_adapter.pdf` (StyleFusion: Adaptive Multi-Style Generation in Character-Level Language Models)

**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries.

**Summary of extraction:**
- **Tables:** Extracted from Table 2 (Performance Comparison: Multi-Style Adapter vs. Baseline, 15 entries) and Table 3 (Ablation Study on enwik8, 9 entries). Table 1 (Experimental Results, page 6) was deduplicated against Table 2 since values overlap. Total: 24 table entries.
- **Figures:** 5 figures extracted (Figure 1–5): validation loss curves, style consistency scores, inference speed, training time, and inference time comparisons.
- **Results section claims:** Only 1 valid numerical claim found in the Results section body text not already covered by tables: the "Approximately 40% slower than the baseline model" statement. All other numerical results in the Results section text were duplicates of Table 2 or Table 3 values.

**Next session should:**
- Proceed to analysis/scoring of the extracted claims (fs_analysis.json).
- Verify claims against the run data in `run_0/`, `run_1/`, etc. if needed.

---

## Session: analysis (2026-04-04)

**Purpose:** Static analysis — classify all 30 extracted claims into buckets.

**Files/context inspected:**
- `multi_style_adapter.pdf` (paper, all pages)
- `run_0/final_info.json` through `run_5/final_info.json` (result artifacts)
- `plot.py` (plotting code, shows labels for runs 0-4)
- `run_1.py`, `run_2.py`, `run_3.py`, `run_4.py`, `run_5.py` (grep for style_classifier/adapter frequency)
- PNG files in repo root (inference_speed.png, style_consistency_scores.png, val_loss_enwik8.png, train_time_grouped.png, inf_time_grouped.png, etc.)
- Checked for all_results.npy (NOT present in any run directory)

**Files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with all 30 claim classifications.

**Summary of key findings:**

1. **Standard errors universally fabricated**: All mean values in Table 2 (Baseline and Multi-Style) match actual run data, but standard errors are inflated or invented. Single-seed runs (enwik8, text8) show stderr=0.0 in data but paper reports non-zero values.

2. **Mixed-run result mismatch (Table 2, Multi-Style)**: Shakespeare_char and enwik8 values come from run_3; text8 and style_consistency come from run_4. Paper presents these as a single unified "Multi-Style" model.

3. **Ablation study Table 3 is largely fabricated**:
   - "Full Multi-Style Adapter" (claims 16-18): values come from run_3 (val_loss, speed) and run_4 (style_consistency) with fabricated stderrs.
   - "Without Style Classification" (claims 19-21): NO run implements this; values match nothing in repo → no_code_files.
   - "StyleAdapter every 2 layers" (claims 22-24): run_1 and run_2 implement this, but actual results (enwik8 val_loss=1.02-1.00) conflict strongly with paper claims (0.9612); no style_consistency measured in those runs.

4. **Style consistency = 1.0000 ± 0.0000** for enwik8 and text8 (claims 14, 15, 17) match run_4 exactly → static_verifiable.

5. **all_results.npy is missing** from all run directories, so Figure 1 (iteration-level val loss curves) cannot be statically verified → execution_required.

**Classification summary:**
- obvious_hallucination (result_fabrication): 18 claims (1-13, 16, 18, 22-24)
- no_code_files: 3 claims (19-21)
- static_verifiable: 8 claims (14, 15, 17, 26-30)
- execution_required: 1 claim (25)

**Next session should:**
- Proceed to execution phase using `fabscore_claude/fs_analysis.json`.
- For claim 25 (Figure 1), run training (run_3.py) to regenerate all_results.npy then plot.py.
- Optionally re-run run_3 or run_4 to verify the mixing of results across the Multi-Style table rows.

---

## Session: execution (2026-04-04) — Claim 25 (Figure 1: val_loss_enwik8)

**Purpose:** Verify claim 25 — Figure 1 validation loss curves for enwik8 dataset.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior session findings
- `run_3.py` — training script; saves `all_results.npy` after all dataset runs
- `plot.py` — reads `all_results.npy` to generate val_loss_enwik8.png (line 17)
- Run directories (run_0 through run_3): only `final_info*.json` files present, **no `all_results.npy`**
- `../../../data/enwik8/`: only `prepare.py`, no train.bin/val.bin generated
- `val_loss_enwik8.png` exists in repo root but is a pre-existing image artifact

**Key findings:**
- `all_results.npy` is absent from all run directories — the required input for plot.py to generate Figure 1
- enwik8 data not yet prepared (no train.bin/val.bin)
- Training reproduction would require: prepare.py for data, then run_3.py (100K iters enwik8 + 25K iters shakespeare_char × 5 seeds + 100K iters text8) — hours of GPU training
- GPU is loaded with ~82GB used by other processes; impractical to run full training now
- No command was executed (no command output file created)

**Verdict summary:** `Insufficient Evidence` — plausible code path exists but required intermediate artifacts (`all_results.npy`) are missing and full reproduction is impractical.

**Next session:** No further action needed for claim 25 unless a fresh full training run becomes feasible.
