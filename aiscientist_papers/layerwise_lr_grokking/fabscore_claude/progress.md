# Progress Log

## Session 1 — extraction

**Purpose:** Extract experimental results from the paper.

**Paper inspected:** `layerwise_lr_grokking.pdf` — "Grokking Accelerated: Layer-wise Learning Rates for Transformer Generalization"

**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries

**Summary of extraction:**
- **Tables:** 36 entries from Table 1 (comparison of baseline vs. layer-wise LR across 4 tasks: Mod Division, Mod Subtraction, Mod Addition, Permutation — metrics: Final Val Acc, Steps to 99% Val Acc, Final Val Loss) and Table 2 (ablation study on permutation task with 4 method variants).
- **Figures:** 1 entry — Figure 1 with 4 subfigures (a–d) showing validation accuracy curves for all tasks.
- **Results section:** 3 entries — percentage reduction/improvement values (54.2%, 56.3%, 54.6%) that appear in text but not in any table.

**Next session should:**
- Run analysis/execution steps (fs_analysis.json, fs_execution.json) to verify claims.
- Cross-check extracted table values against the actual paper tables for correctness.

---

## Session 2 — analysis

**Purpose:** Static analysis to classify all 40 extracted claims.

**Files inspected:**
- `layerwise_lr_grokking/latex/template.tex` — confirmed paper structure, Table 1 and Table 2 values, method description (LR params), figure paths
- `run_0/final_info.json` — baseline results across 4 tasks
- `run_1/final_info.json` — initial layerwise LR results
- `run_2/final_info.json` — adjusted LR results
- `run_3/final_info.json` — optimized LR results (paper's "Ours")
- `run_4/final_info.json` — fine-tuned LR results
- `run_3.py` — confirmed code structure
- `fabscore_claude/progress.md` — prior session notes

**Files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with full classification of all 40 claims

**Key findings:**

1. **Claims 1–21, 23, 25 (Table 1 baseline and non-permutation Ours, + permutation Ours val_acc):** All match run_0 and run_3 final_info.json exactly. `static_verifiable`.

2. **Claims 22, 24 (Permutation Ours steps=5270.0, val_loss=0.0106):** The paper states "Ours = Run 3" (template.tex:459, 474) but run_3 permutation has step=5050.0 and val_loss=0.01962. The reported values (5270.0, 0.0106) match run_4 permutation instead. Since val_acc=0.9995 (claim 20) matches run_3 but steps/loss match run_4, the paper mixes results from different runs → `obvious_hallucination/result_fabrication`.

3. **Claims 26–27 (Table 2 Full Method steps=5270.0, val_loss=0.0106):** Same fabrication as claims 22/24.

4. **Claims 28–36 (Table 2 ablation conditions):** No ablation scripts exist (no eta_e=eta_l, eta_e=eta_h, eta_l=eta_h variants). The step value 7176.7 matches run_2 permutation but the other metrics don't; val_acc≈0.9625 and val_loss=0.1648 match run_4 x_div_y (wrong task). All three ablation conditions show identical steps and val_loss — implausible for independent experiments → `obvious_hallucination/result_fabrication`.

5. **Claims 38–40 (results_section percentages):** Mathematically correct given verified step counts → `static_verifiable`.

6. **Claim 37 (Figure 1):** Underlying PNG files exist; plot.py can regenerate from all_results.npy → `execution_required`.

**Classification summary:**
- static_verifiable: 26 (claims 1–21, 23, 25, 38, 39, 40)
- obvious_hallucination (result_fabrication): 13 (claims 22, 24, 26, 27, 28–36)
- execution_required: 1 (claim 37)
- no_code_files: 0 | insufficient_evidence: 0 | error: 0

**Next session should:**
- Execute `python plot.py` to verify Figure 1 (claim 37).
- Optionally verify the paper's hyperparameter description inconsistency (method section says eta_e=8e-4=run_4 but results section says Ours=Run 3 with eta_e=7e-4).

---

## Session 3 — execution (claim 37)

**Purpose:** Verify Figure 1 (claim 37) — validation accuracy curves for all tasks, comparing baseline and layer-wise LR.

**Files inspected:**
- `plot.py` — requires `all_results.npy` in each run_* directory; plots all 5 runs' val_acc curves per dataset
- `run_0/` through `run_4/` — only contain `final_info*.json` files, no `all_results.npy`
- Pre-existing PNG files: `val_acc_x_div_y.png`, `val_acc_x_minus_y.png`, `val_acc_x_plus_y.png`, `val_acc_permutation.png` exist in repo root

**Execution:**
- Ran `python plot.py` from repo root
- **Failed:** `FileNotFoundError: run_4/all_results.npy` — all_results.npy files are missing from all run directories
- Full training reproduction is not feasible (would require hours/days across 5 runs × 4 datasets × 3 seeds)

**Verdict:** `Insufficient Evidence`
- Pre-existing PNG images exist but cannot be used alone per policy
- all_results.npy files (required input to plot.py) are absent
- Running the training to regenerate them is not feasible in this session

**Artifacts created:**
- `fabscore_claude/workspace/claim_37_command_output.txt` — plot.py failure output

**Next session should:**
- No further action needed for claim 37; verdict is Insufficient Evidence.

