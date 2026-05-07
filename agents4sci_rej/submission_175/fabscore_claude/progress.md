# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** 175_Hierarchical_Meta_Learning.pdf (Hierarchical Meta-Learning for Cancer Pathway Signatures)
**Files created:**
- `fabscore_claude/fs_extracted.json` — extracted tables, figures, and results_section claims

**Summary of extraction:**
- 1 table extracted (Table 1: ablation study, 10 entries)
- 4 figures extracted (Figures 1–4)
- 8 results_section claims extracted (few-shot accuracy comparisons, transferability thresholds, Pearson correlation, ablation improvement)

**Next session should:**
- Run scoring/evaluation against ground truth claims (fs_analysis or verdict step)

---

## Session 3 — 2026-04-24
**Purpose:** execution (verify claim 1: Full Model 5-shot accuracy 85.7%)

**Files/context inspected:**
- `SupplementaryMaterials/results/hierarchical_meta_learning_analysis.pkl` — confirmed few_shot_results are from nearest-neighbor classifier
- `SupplementaryMaterials/code/simple_meta_learning_analysis.py` — confirmed nearest-neighbor approach (not MAML)
- `SupplementaryMaterials/code/hierarchical_meta_learning_pipeline.py` — main MAML pipeline
- `SupplementaryMaterials/code/src/data/preprocessing.py` — requires RDS files

**Execution artifacts created:**
- `fabscore_claude/workspace/claim_1_command_output.txt` — command outputs

**Commands run:**
1. Inspected pkl few_shot_results: SKCM 1.0, SARC 0.633, TGCT 0.867, LGG 0.967, MESO 0.8 (avg ~85.3%) from nearest-neighbor classifier
2. Attempted pipeline: `/home/chenhui/miniconda3/envs/mlrbench/bin/python3 hierarchical_meta_learning_pipeline.py --data_dir ../data ...`
   - FAILED: `FileNotFoundError: No RDS files found in ../data`

**Key findings:**
- No TCGA RDS data files exist anywhere in the repository
- The MAML pipeline cannot be run at all (missing data)
- No MAML checkpoints or evaluation results exist
- The only few_shot_results (from pkl) are from a nearest-neighbor classifier (~85.3% average), NOT the hierarchical MAML
- The nearest-neighbor results don't exactly match 85.7%, but they're close

**Verdict for claim 1:** `Insufficient Evidence`
- Code path for MAML exists but was never executed (no data, no checkpoints)
- Cannot verify or refute the 85.7% claim without running the MAML pipeline on actual TCGA data
- The only existing result (~85.3% from nearest-neighbor) is from a different model

**Next session should:**
- No further work needed on claim 1 — classified as Insufficient Evidence

---

## Session 2 — 2026-04-24
**Purpose:** analysis (static code audit)

**Files/context inspected:**
- `175_Hierarchical_Meta_Learning.pdf` — full paper (10 pages + appendix)
- `SupplementaryMaterials/results/hierarchical_meta_learning_analysis.pkl` — only execution artifact
- `SupplementaryMaterials/code/simple_meta_learning_analysis.py` — script that actually produced the pkl
- `SupplementaryMaterials/code/create_publication_figures.py` — figure generation script (674 lines)
- `SupplementaryMaterials/code/hierarchical_meta_learning_pipeline.py` — full MAML pipeline (never executed)
- `SupplementaryMaterials/code/src/models/hierarchical_maml.py` — MAML model code
- `SupplementaryMaterials/code/src/analysis/evaluation.py` — evaluation framework
- `SupplementaryMaterials/CLAUDE.md` — project guidance

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — full analysis with 22 claim classifications

**Key findings:**
1. The pkl results come from `simple_meta_learning_analysis.py` which uses a **nearest-neighbor classifier**, NOT the hierarchical MAML claimed in the paper.
2. The full MAML pipeline code exists but was NEVER executed: no checkpoints (.pt), no evaluation logs, no result JSONs from the MAML pipeline.
3. Figure generation (`create_publication_figures.py`) fabricates several panels: confusion matrix (np.random), learning curves (synthetic exponential), transfer learning performance matrix (np.random.rand), clinical relevance scores (np.random.beta), and "literature evidence" scores (hardcoded fake values).
4. Pathway importance claimed as "derived from attention weights" is actually computed via F-ratio statistic on raw data.
5. Pearson r=0.78 reported in paper but figure code produces R²≈0.27 (r≈0.52); moreover the "independent cohort" data is hardcoded synthetic.

**Classification summary (22 claims):**
- `no_code_files`: 10 (claims 3, 4, 7, 8, 9, 10, 18, 19, 20, 22) — ablation variants without code, pathway-level transferability formula unimplemented
- `obvious_hallucination`: 5 (claims 11, 12, 13, 14, 21) — figures using synthetic/random data or wrong experimental method
  - experiment_fabrication: 3 (claims 11, 12, 13)
  - data_fabrication: 2 (claims 14, 21)
- `execution_required`: 7 (claims 1, 2, 5, 6, 15, 16, 17) — MAML+baseline pipeline code exists but no trained models/results

**Next session should:**
- Execute the MAML pipeline to attempt to verify claims 1, 2, 5, 6, 15, 16, 17
- Check if TCGA data at the expected path is accessible
- Run: `python hierarchical_meta_learning_pipeline.py --data_dir ../data --results_dir ./results --config configs/default_config.yaml`

---

## Session 4 — 2026-04-24
**Purpose:** execution (verify claim 2: Full Model 10-shot accuracy 92.4%)

**Files/context inspected:**
- `fabscore_claude/workspace/claim_1_command_output.txt` — reused artifacts from session 3
- `fabscore_claude/workspace/results/logs/hierarchical_meta_learning_20260424_194541.log` — confirmed MAML pipeline crash

**Execution artifacts reused (no new commands run):**
- Session 3's pkl extraction showed 10-shot nearest-neighbor results: SKCM=1.0, SARC=0.933, TGCT=0.9, LGG=0.967, MESO=0.767 → avg ~91.3%
- Session 3 confirmed MAML pipeline fails: `FileNotFoundError: No RDS files found in ../data`

**Key findings:**
- Same blocker as claim 1: no TCGA RDS data files, MAML pipeline cannot run
- Only 10-shot results available are from nearest-neighbor classifier (~91.3%), NOT the hierarchical MAML Full Model
- The nearest-neighbor 10-shot average (~91.3%) is close to but does not match the claimed 92.4%
- No MAML checkpoints or 10-shot evaluation logs exist

**Verdict for claim 2:** `Insufficient Evidence`
- Code path for MAML exists but was never executed (no data, no checkpoints)
- Cannot verify or refute the 92.4% claim without running the MAML pipeline on actual TCGA data
- The only existing 10-shot result (~91.3% from nearest-neighbor) is from a different model

**Next session should:**
- No further work needed on claim 2 — classified as Insufficient Evidence

---

## Session 5 — 2026-04-24
**Purpose:** execution (verify claim 5: - Attention, 5-shot Accuracy: 81.2%)

**Files/context inspected:**
- `fabscore_claude/workspace/claim_1_command_output.txt` — reused prior session artifacts
- `SupplementaryMaterials/results/` — only pkl files present (nearest-neighbor results)
- `SupplementaryMaterials/code/configs/default_config.yaml` — confirmed `use_attention: true` is default; ablation would require `use_attention: false`
- `SupplementaryMaterials/code/src/models/hierarchical_maml.py` — code path for use_attention=False exists

**Execution artifacts (no new commands run):**
- Reused evidence from Sessions 3 and 4: MAML pipeline crashes with `FileNotFoundError: No RDS files found in ../data`
- No TCGA data, no MAML checkpoints, no result JSONs for any configuration

**Key findings:**
- Same fundamental blocker as claims 1 and 2: no TCGA RDS data files available
- The "- Attention" ablation would require running pipeline with `use_attention: false`
- Code path exists (parameter confirmed in configs and model code)
- No results for ANY MAML configuration exist, let alone the no-attention ablation
- No pkl or JSON file contains ablation results

**Verdict for claim 5:** `Insufficient Evidence`
- The no-attention ablation code path is plausible but requires TCGA data
- Cannot verify or refute the 81.2% claim without running the MAML pipeline on actual data
- No existing result artifacts correspond to this ablation configuration

**Next session should:**
- No further work needed on claim 5 — classified as Insufficient Evidence

---

## Session 6 — 2026-04-24
**Purpose:** execution (verify claim 6: - Attention, 10-shot Accuracy: 88.7%)

**Files/context inspected:**
- `fabscore_claude/progress.md` — reused findings from Sessions 3, 4, and 5
- `fabscore_claude/workspace/claim_1_command_output.txt` — prior MAML pipeline crash evidence

**Execution artifacts (no new commands run):**
- Reused evidence from Sessions 3–5: MAML pipeline crashes with `FileNotFoundError: No RDS files found in ../data`
- No TCGA data, no MAML checkpoints, no result JSONs for any configuration
- No pkl or JSON file contains ablation results for any configuration

**Key findings:**
- Same fundamental blocker as claims 1, 2, and 5: no TCGA RDS data files available
- The "- Attention" 10-shot ablation requires running pipeline with `use_attention: false`
- Code path exists (confirmed in Session 5: `use_attention` parameter in configs and model code)
- No results for ANY MAML configuration exist, let alone the no-attention 10-shot ablation
- Claim 6 is the 10-shot counterpart of claim 5 (5-shot); same evidence applies

**Verdict for claim 6:** `Insufficient Evidence`
- The no-attention ablation code path is plausible but requires TCGA data that is absent from the repository
- Cannot verify or refute the 88.7% claim without running the MAML pipeline on actual data
- No existing result artifacts correspond to this ablation configuration

**Next session should:**
- No further work needed on claim 6 — classified as Insufficient Evidence

---

## Session 8 — 2026-04-24
**Purpose:** execution (verify claim 16: 5-shot learning 85.7% accuracy vs. 62.3% for transfer learning)

**Files/context inspected:**
- `fabscore_claude/progress.md` — reused findings from Sessions 3–7
- `SupplementaryMaterials/code/src/training/baselines.py` — inspected BaselineComparator class

**Execution artifacts (no new commands run):**
- Reused evidence from Sessions 3–7: MAML pipeline crashes with `FileNotFoundError: No RDS files found in ../data`
- No TCGA RDS data files exist anywhere in the repository

**Key findings:**
1. No TCGA RDS data → MAML pipeline cannot run (same blocker as all prior sessions)
2. `BaselineComparator` class in `baselines.py` exists but contains sklearn models (RandomForest, SVM, LogisticRegression, XGBoost, LightGBM), standard neural networks, hierarchical neural networks, and prototypical networks — **NO "transfer learning" baseline is implemented**
3. No "transfer learning" method or class exists anywhere in the codebase
4. No execution artifacts for any comparison between hierarchical MAML and a transfer learning baseline
5. Session 3 found 5-shot nearest-neighbor avg ~85.3%, but this is from a different model (not MAML) and contains no transfer learning comparison

**Verdict for claim 16:** `Insufficient Evidence`
- The hierarchical MAML code path exists but cannot execute (no data)
- The "transfer learning" baseline referenced in the claim has no implementation in the repository
- No concrete code/execution conflict is visible — absence of implementation ≠ fabrication per classification rules
- Cannot verify or refute either the 85.7% or 62.3% values

**Next session should:**
- No further work needed on claim 16 — classified as Insufficient Evidence

---

## Session 7 — 2026-04-24
**Purpose:** execution (verify claim 15: 1-shot learning 70.2% accuracy vs. 45.1% for standard MAML)

**Files/context inspected:**
- `fabscore_claude/progress.md` — reused findings from Sessions 3–6
- `SupplementaryMaterials/code/src/training/baselines.py` — no StandardMAML class; only PrototypicalNetwork, StandardNN, HierarchicalNN
- `SupplementaryMaterials/code/configs/default_config.yaml` — `few_shot_scenarios: [1, 3, 5, 10]` but no 1-shot artifact exists
- `SupplementaryMaterials/results/` — only `hierarchical_meta_learning_analysis.pkl` and `tcga_statistics.pkl`; no 1-shot or MAML results
- `SupplementaryMaterials/manuscript/hierarchical_meta_learning_neurips2025.tex` — confirmed the exact 70.2%/45.1% claim in LaTeX source

**Execution artifacts (no new commands run):**
- Reused evidence from Sessions 3–6: MAML pipeline crashes with `FileNotFoundError: No RDS files found in ../data`
- No TCGA RDS data files exist anywhere in the repository

**Key findings:**
1. No TCGA RDS data → MAML pipeline cannot run (same blocker as all prior sessions)
2. "Standard MAML" baseline is mentioned in manuscript and methodology doc but has NO implementation in the codebase (baselines.py has PrototypicalNetworks, not MAML)
3. No 1-shot evaluation artifacts exist anywhere (pkl files don't contain 1-shot results)
4. The claim in the LaTeX manuscript (70.2%/45.1%) has no corresponding execution artifacts

**Verdict for claim 15:** `Insufficient Evidence`
- Both the hierarchical MAML and standard MAML baseline lack executable code paths that could produce 1-shot results (no data, no standard MAML implementation)
- The code path for hierarchical MAML exists but is blocked by missing data; standard MAML is not implemented at all
- Per classification rules: no concrete code/execution conflict is visible (standard MAML absence ≠ fabrication without more evidence), so Insufficient Evidence applies

**Next session should:**
- No further work needed on claim 15 — classified as Insufficient Evidence

---

## Session 9 — 2026-04-24
**Purpose:** execution (verify claim 17: 10-shot learning 92.4% accuracy vs. 71.8% for prototypical networks)

**Files/context inspected:**
- `fabscore_claude/progress.md` — reused findings from Sessions 3–8
- `SupplementaryMaterials/code/src/training/baselines.py` — confirmed PrototypicalNetwork class (lines 91-490) and `evaluate_prototypical_networks()` method (line 411)
- `SupplementaryMaterials/results/` — only pkl files, no JSON results, no prototypical network evaluation artifacts

**Execution artifacts (no new commands run):**
- Reused evidence from Sessions 3–8: MAML pipeline crashes with `FileNotFoundError: No RDS files found in ../data`
- No TCGA RDS data files exist anywhere in the repository

**Key findings:**
1. No TCGA RDS data → MAML pipeline cannot run (same blocker as all prior sessions)
2. `PrototypicalNetwork` class exists in baselines.py (line 91) with proper implementation
3. `BaselineComparator.evaluate_prototypical_networks()` method exists (line 411) but requires TCGA data to run
4. No evaluation artifacts for prototypical networks exist (no pkl/JSON with 71.8% result)
5. No 10-shot MAML evaluation artifacts exist (same as claim 2)
6. Grep for "71.8" or "prototypical" in results directory: no matches

**Verdict for claim 17:** `Insufficient Evidence`
- Both the hierarchical MAML (92.4%) and prototypical network (71.8%) code paths exist but cannot execute without TCGA data
- No execution artifacts for either model exist
- Cannot verify or refute either accuracy value
- Per classification rules: no concrete code/execution conflict is visible

**Next session should:**
- No further work needed on claim 17 — classified as Insufficient Evidence
