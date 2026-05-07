# Progress Log

## Session 1 — 2026-04-22
**Purpose:** extraction

**Paper inspected:** `input_encoding_grokking.pdf`
- Title: "Decoding Grokking: How Input Representations Shape Mathematical Learning in Transformers"
- 11 pages, AI Scientist generated preprint
- No tables; results presented via 3 figures and body text in Section 6

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (empty), figures (3 entries), and results_section (7 entries)

**Summary of extraction:**
- Tables: None found in the paper
- Figures: Figure 1 (validation accuracy trajectories), Figure 2 (grokking points comparison), Figure 3 (final validation accuracies)
- Results section: 7 numerical claims extracted from Section 6 body text covering grokking steps for modular addition, subtraction, division, and permutation composition under different encoding schemes, as well as final validation accuracies

**Next session should:**
- Run scoring/evaluation on `fs_extracted.json` if needed
- No additional extraction needed; the paper has no supplementary materials or appendix with additional numerical results

---

## Session 2 — 2026-04-22
**Purpose:** analysis (static code audit)

**Files inspected:**
- `input_encoding_grokking.pdf` (paper)
- `run_0/final_info.json`, `run_1/final_info.json`, `run_2/final_info.json` (verified all numerical claims)
- `plot.py` (plotting logic; requires `all_results.npy` for Figure 1)
- `experiment.py`, `run_1.py`, `run_2.py` (confirmed `all_results.npy` is produced by training scripts but absent from repo)
- Directory listings of `run_0/` confirmed: only 13 JSON files, no `.npy` files in any run directory

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with full classification of all 10 claims

**Summary of classifications:**
| Bucket | Claims |
|---|---|
| static_verifiable | 2, 3, 4, 5, 6, 7, 9, 10 (8 claims) |
| obvious_hallucination (result_fabrication) | 8 |
| execution_required | 1 |

**Key finding — Claim 8 (result_fabrication):**
Paper states "All encoding schemes achieve high final accuracy (>99%)" for modular addition AND subtraction.
Actual JSON data:
- One-hot (run_1) x_minus_y: `final_val_acc_mean = 0.8519` (85.19%) — **not >99%**
- Binary (run_2) x_minus_y: `final_val_acc_mean = 0.8879` (88.79%) — **not >99%**
This is a clear result_fabrication: the paper overstates final accuracy for subtraction under one-hot and binary encodings.

**Key finding — Claim 1 (execution_required):**
`all_results.npy` (needed for per-step time-series plots in Figure 1) is missing from all run directories.
Training scripts must be re-run to regenerate it. Entrypoints: `python experiment.py`, `python run_1.py`, `python run_2.py`, then `python plot.py`.

**Claims 4–7, 9–10 (static_verifiable):**
All grokking step counts and final accuracy percentages in the paper match the `step_val_acc_99_mean` and `final_val_acc_mean` values in run_0, run_1, run_2 final_info.json files to within rounding (e.g., 2363.33→2363, 1173.33→1173, 876.67→877, etc.).

**Claims 2–3 (static_verifiable):**
Figures 2 and 3 are bar charts reading from `final_info.json` (via plot.py lines 114–154). The underlying JSON data files exist and contain the values used to generate these charts.

**Next session should:**
- Execute `experiment.py`, `run_1.py`, `run_2.py` to regenerate `all_results.npy` for Figure 1 verification (Claim 1)
- No further static analysis required; all claims have been classified

---

## Session 3 — 2026-04-22
**Purpose:** execution (Claim 1 — Figure 1: validation accuracy time-series)

**Files inspected:**
- `plot.py` — confirmed 2x2 subplot structure (lines 92-112) requiring all_results.npy from run_0, run_1, run_2, run_5
- `experiment.py` — confirmed all_results.npy saved at end of all 12 training runs (4 datasets x 3 seeds), 5000 steps each
- All run directories (run_0 through run_5) — confirmed: only final_info*.json files, NO all_results.npy
- `fabscore_claude/workspace/run_0_fresh/` — partial execution artifact (2 of 12 seeds)

**Commands run:**
- `python -c "import torch; print('CUDA:', torch.cuda.is_available())"` → CUDA available, 1 GPU
- `timeout 600 python experiment.py --out_dir fabscore_claude/workspace/run_0_fresh` → timed out after 10 min; completed only x_div_y seeds 0 and 1 of 12 total seeds
  - Partial output: `final_info_x_div_y_0.json` (val_acc=0.86%, step_val_acc_99=5000 — used hybrid encoding)
  - vs existing `run_0/final_info_x_div_y_0.json` (val_acc=100%, step_val_acc_99=3910)

**Key findings:**
1. `all_results.npy` is absent from ALL run directories (run_0 through run_5). Confirmed by `find` command.
2. GPU is available (CUDA). Code runs without errors.
3. Full regeneration would require ~4+ hours (4 scripts x ~60 min each), beyond practical session limits.
4. `experiment.py` uses `encoding_type='hybrid'` but is labeled "run_0" output directory — mismatch with run_0 original which uses default encoding. This suggests `experiment.py` in the repo may correspond to run_5 (hybrid encoding), not run_0.
5. The `combined_val_acc.png` exists in repo root but is insufficient alone for `Verified`.
6. Code structure (plot.py lines 92-112) exactly matches Figure 1 description: 2x2 subplots, val_accuracy over update steps, lines per encoding scheme.

**Artifacts created:**
- `fabscore_claude/workspace/run_0_fresh/` — partial run (2 seeds)
- `fabscore_claude/workspace/claim_1_command_output.txt` — created during this session

**Verdict summary:** `Insufficient Evidence` — all_results.npy absent from all run directories; training scripts confirmed runnable on GPU but full reproduction requires ~4+ hours; no concrete conflict with the paper found (code structure matches Figure 1 claim).

**Next session should:**
- If more time/compute available: run full `run_1.py --out_dir workspace/run_1_fresh` and related scripts to regenerate all_results.npy
- Otherwise, classification stands as `Insufficient Evidence`
