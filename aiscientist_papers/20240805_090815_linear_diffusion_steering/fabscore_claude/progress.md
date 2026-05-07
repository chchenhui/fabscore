# Progress Log

## Session 1 — Extraction
- **Purpose**: extraction
- **Paper inspected**: linear_diffusion_steering.pdf ("Guided Paths in Low-D: Controlled Generation via Trajectory Guidance in Low-Dimensional Diffusion Models")
- **Files created/updated**:
  - `fabscore_claude/fs_extracted.json` (created)
- **Summary**:
  - Extracted 40 table entries from Table 1 (KL divergence and MSE for linear guidance across 4 datasets × 4 conditions), Table 2 (circular guidance results), and Table 3 (ablation study).
  - Extracted 2 figures: Figure 1 (generated samples grid) and Figure 2 (trade-off scatter plot).
  - No results_section entries: all quantitative claims in the Results section either duplicate table values or are purely qualitative (no independent numeric performance metrics in body text beyond what tables capture).
- **Next session**: Run analysis/scoring on fs_extracted.json; verify completeness against paper claims.

## Session 2 — Analysis (2026-04-22)
- **Purpose**: analysis (static code audit)
- **Files inspected**:
  - `linear_diffusion_steering.pdf` (full paper)
  - `run_0/final_info.json` through `run_5/final_info.json` (all result files)
  - `run_1.py`, `run_3.py`, `run_4.py`, `run_5.py` (run scripts)
  - `experiment.py`, `plot.py`, `notes.txt`
  - Glob for `**/all_results.pkl` → returned empty (no pkl files exist anywhere)
- **Files created/updated**:
  - `fabscore_claude/fs_analysis.json` (created)
  - `fabscore_claude/progress.md` (updated)
- **Summary of classifications**:
  - **35 static_verifiable**: Claims 1–32 (all KL and MSE values for baseline, linear 0.1/0.5/1.0, and circular guidance KL) directly match corresponding `run_*/final_info.json` values. Claims 37, 39, 40 (Table 3 Line dataset ablation KL and MSE) also match run_0 and run_1 JSONs.
  - **4 obvious_hallucination (experiment_fabrication)**: Claims 33–36 (Table 2, circular guidance path adherence, all reported as 1.667). `run_4.py` line 289 computes path_adherence against the **linear** path `np.linspace([-1,-1],[1,1],...)` even though the guidance is circular. The paper (Section 5.5) states path adherence is measured against "the nearest points on the guidance path" (i.e., the circular path). The 1.667 ≈ 5/3 value is explained analytically: with guidance_strength=1.0, all samples collapse to the circular endpoint (1,0), and MSE of (1,0) against the linear path = 5/3. Clear evaluation protocol conflict.
  - **1 insufficient_evidence**: Claim 38 (Table 3, Without Guidance, Path Adherence = 3.842). `run_0/final_info.json` has no `path_adherence` key. No `all_results.pkl` files exist. No repo-native code computes path adherence for the baseline. Value is completely untraced.
  - **2 execution_required**: Claims 41–42 (Figures 1 and 2). Figure PNG files exist, but `all_results.pkl` files are missing and `plot.py` requires them at load time. Cannot regenerate figures without retraining all models.
- **Key finding**: `run_4.py` (circular guidance) uses the wrong reference path for path adherence computation (linear path instead of circular path), making the Table 2 Path Adherence values `experiment_fabrication`. All KL divergence values and all linear guidance path adherence values are consistent with stored artifacts.
- **Next step**: Execution phase — retrain models (run_1–run_4) to regenerate all_results.pkl, then run plot.py to verify figures. Also consider adding path_adherence computation to baseline to check claim 38.

## Session 3 — Execution (Claim 41) (2026-04-22)
- **Purpose**: execution — verify Claim 41 (Figure 1 content: row ordering and red dashed lines)
- **Files inspected**: plot.py, run_1.py through run_4.py, fabscore_claude/workspace/ (prior artifacts)
- **Execution artifacts created**:
  - `fabscore_claude/workspace/gen_baseline.py` — new baseline generation script (guidance_strength=0.0)
  - `fabscore_claude/workspace/run_0/all_results.pkl` + `final_info.json` — baseline (no guidance)
  - `fabscore_claude/workspace/run_1/all_results.pkl` + `final_info.json` — linear guidance 0.5
  - `fabscore_claude/workspace/run_2/all_results.pkl` + `final_info.json` — linear guidance 0.1
  - `fabscore_claude/workspace/run_3/all_results.pkl` + `final_info.json` — linear guidance 1.0
  - `fabscore_claude/workspace/run_4/all_results.pkl` + `final_info.json` — circular guidance 1.0
  - `fabscore_claude/workspace/figure1_verification.png` — regenerated Figure 1
  - `fabscore_claude/workspace/claim_41_command_output.txt` — command outputs
- **Verdict summary (Claim 41)**: **Verified**. Successfully regenerated all 5 all_results.pkl files and ran plot.py equivalent. Row order (top to bottom): Baseline, Linear(0.5), Linear(0.1), Linear(1.0), Circular(1.0) — exactly matches the claim. Red dashed lines (r--) indicate linear path for rows 1-4 and circular path for row 5 — exactly matches the claim.
- **Note**: No run_0.py exists in the repo; baseline was regenerated using a custom script with guidance_strength=0.0 (equivalent to no guidance). The figure structure is fully determined by plot.py code.
- **Next session**: Claim 42 (Figure 2: KL divergence vs path adherence trade-off scatter plot). The same workspace pkl files can be reused for this figure.

## Session 4 — Execution (Claim 42) (2026-04-22)
- **Purpose**: execution — verify Claim 42 (Figure 2: KL vs path adherence trade-off scatter, error bars over 5 runs)
- **Files inspected**: plot.py (lines 113-129), workspace/run_*/final_info.json, workspace/kl_path_tradeoff.png (newly generated), repo/kl_path_tradeoff.png (original)
- **Execution artifacts created**:
  - `fabscore_claude/workspace/kl_path_tradeoff.png` — freshly generated Figure 2 using workspace pkl files
  - `fabscore_claude/workspace/claim_42_command_output.txt` — command outputs
- **Key findings**:
  1. plot.py lines 116-128 use `ax.scatter()` with NO error bar parameters (no xerr/yerr, no plt.errorbar)
  2. No 5 replications exist in repo — only single runs (run_1 through run_4)
  3. final_info.json files have no 'stds' key, only 'means'
  4. Paper claims "Error bars represent standard deviation over 5 runs" — completely absent from both code and generated figure
  5. Original repo kl_path_tradeoff.png also has no error bars (confirmed visually)
  6. The basic structure (trade-off scatter for different guidance methods) IS present and correct
- **Verdict summary (Claim 42)**: **Result Fabrication**. The trade-off scatter plot exists and shows KL vs path adherence for different guidance methods as described. However, the paper's claim of "error bars represent standard deviation over 5 runs" is false — the code uses ax.scatter() without error bars, there are no 5 replications, and no standard deviation data exists anywhere in the repository.
- **Next session**: No further sessions needed for this directory's execution claims (41 and 42 both handled).
