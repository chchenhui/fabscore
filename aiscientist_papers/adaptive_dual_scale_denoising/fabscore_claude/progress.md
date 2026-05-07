# Progress Log

## Session: 2026-04-04
**Purpose:** extraction
**Paper inspected:** adaptive_dual_scale_denoising.pdf
**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries

**Summary:**
- Extracted 60 table entries from Table 1 (KL Divergence, Training Time, Inference Time across 5 model variants: Baseline, Fixed Weighting, Adaptive Weighting, Weight Analysis, Improved Weight Network; for 4 datasets: Circle, Dino, Line, Moons)
- Extracted 2 figures (Figure 1: generated samples grid, Figure 2: weight evolution across timesteps)
- Extracted 6 results_section claims including percentage improvements and training time comparisons

**Next session should:**
- Perform analysis/verification of the extracted results against paper claims
- Check if any results were missed or misattributed

---

## Session: 2026-04-04 (analysis)
**Purpose:** analysis
**Files inspected:**
- `adaptive_dual_scale_denoising.pdf` (read attempt; PDF renderer unavailable)
- `run_0/final_info.json`, `run_1/final_info.json`, `run_2/final_info.json`, `run_3/final_info.json`, `run_4/final_info.json`, `run_5/final_info.json`
- `plot.py` (read fully)
- `fabscore_claude/progress.md`
- Repository structure via Explore agent

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with full classification of all 68 claims

**Summary of classifications:**

**Key finding — result_fabrication for "Improved Weight Network" (Run 5) claims:**
- `plot.py` labels `run_4` as "Weight Visualization" and `run_5` as "Improved Weight Network"
- The paper's "Improved Weight Network (Run 5)" reported values EXACTLY match `run_4/final_info.json` (e.g., Dino KL: 0.862 = run_4's 0.86225)
- `run_5/final_info.json` has substantially different values (e.g., Dino KL: 1.019 vs 0.862 reported)
- This is a clear mis-mapping: paper presents run_4's results as "Run 5 (Improved Weight Network)"
- The claimed 12.8% Dino improvement doesn't hold for run_5 (KL actually worsens: 0.989→1.019)

**Claims 1–48 (Baseline, Fixed Weighting, Adaptive Weighting, Weight Analysis):** `static_verifiable`
- All 48 values exactly match run_0, run_1, run_2, run_3 final_info.json respectively (to 3 decimal places)

**Claims 49–60 (Improved Weight Network table values):** `obvious_hallucination` / `result_fabrication`
- All 12 values match run_4, not run_5 (which is labeled "Improved Weight Network" in plot.py)

**Claims 61–62 (Figures 1 and 2):** `execution_required`
- PNG artifacts exist but all_results.pkl (needed by plot.py) is absent from main run directories
- Entrypoint: `python plot.py`

**Claims 63–68 (Results section):** `obvious_hallucination` / `result_fabrication`
- All calculations (percentages, training time averages) match run_4 data, not run_5

**Totals:** 48 static_verifiable, 18 obvious_hallucination (all result_fabrication), 2 execution_required

**Next session should:**
- Run execution step for claims 61–62 (figures): execute `python plot.py` to regenerate and verify figure data
- Optionally re-run run_5.py to confirm run_5 results are as stored in final_info.json

---

## Session: 2026-04-04 (execution — claim 61)
**Purpose:** execution
**Claim:** 61 — Figure 1: Generated samples from adaptive dual-scale diffusion model, rows=runs, columns=circle/dino/line/moons datasets

**Files inspected:**
- `plot.py` (read previously, re-confirmed structure: `plt.subplots(num_runs, 4)`, datasets = [circle, dino, line, moons])
- `run_0/ through run_5/` — contain only `final_info.json`, no `all_results.pkl`
- `fabscore_codex/workspace/claim_62_run_3/all_results.pkl` — real pkl with images (10000, 2) for all 4 datasets
- `fabscore_codex/workspace/claim_61_run_1/all_results.pkl` — real pkl (1 training step, images (10000, 2) for all 4 datasets)

**Execution artifacts created:**
- `fabscore_claude/workspace/verify_fig1.py` — verification script
- `fabscore_claude/workspace/claim_61_command_output.txt` — command output
- `fabscore_claude/workspace/claim_61_generated_images_verification.png` — generated figure (970KB, 6 rows × 4 cols)
- `fabscore_claude/workspace/temp_runs/run_*/all_results.pkl` — temporary pkl files created for all 6 runs

**Key findings:**
- `plot.py` line 81: `plt.subplots(num_runs, 4)` → 6 rows × 4 columns
- Line 10: `datasets = ["circle", "dino", "line", "moons"]` — exactly the 4 datasets in the claim
- Lines 83–95: row `i` = run_i label, columns = dataset names
- Existing pkl from `fabscore_codex/workspace/claim_62_run_3/` has real model-generated images of shape (10000, 2) for all 4 datasets
- Figure successfully regenerated at `claim_61_generated_images_verification.png` (970,257 bytes)
- Figure structure: 6 rows (Baseline, Fixed Weighting, Learnable Weighting, Weight Analysis, Weight Visualization, Improved Weight Network) × 4 columns (circle, dino, line, moons) ✓

**Verdict for claim 61:** VERIFIED
- The plot.py code explicitly creates a grid with num_runs rows and 4 columns (circle, dino, line, moons)
- Underlying data (images arrays shape (10000, 2) for all 4 datasets) confirmed from existing pkl
- Figure successfully regenerated with described structure

**Next session should:**
- Process claim 62 (Figure 2: weight evolution) if not yet done

---

## Session: 2026-04-04 (execution — claim 62)
**Purpose:** execution
**Claim:** 62 — Figure 2: Evolution of global and local feature weights across timesteps for different datasets

**Files inspected:**
- `plot.py` lines 101-121: weight_evolution plot code — 2×2 subplots for 4 datasets, solid=global, dashed=local
- `fabscore_codex/workspace/claim_62_run_3/all_results.pkl` — real pkl with weight_evolution shape (100, 2) for all 4 datasets
- `fabscore_codex/workspace/claim_62_plot/weight_evolution.png` — previously generated figure (119,231 bytes)

**Execution artifacts reused (no new commands run):**
- Reused `fabscore_codex/workspace/claim_62_run_3/all_results.pkl` (weight_evolution shape=(100,2) for circle, dino, line, moons)
- Reused `fabscore_codex/workspace/claim_62_plot/weight_evolution.png` (119,231 bytes) — generated by plot.py in prior codex session

**Key findings:**
- `weight_evolution` data confirmed: shape (100, 2) for all 4 datasets → 100 diffusion timesteps, column 0=global, column 1=local
- plot.py generates 2×2 grid with x-axis="Timestep", y-axis="Weight", solid line for global ([:, 0]), dashed for local ([:, 1])
- Figure was successfully re-generated in prior codex session using these underlying data

**Verdict for claim 62:** VERIFIED
- Underlying weight_evolution data exists with shape (100, 2) for all 4 datasets
- plot.py code structure matches all aspects of the claim: x=timesteps, y=weights, solid=global, dashed=local
- weight_evolution.png was re-generated from real data in a prior session

**Next session should:**
- No further action needed for claims 61–62
