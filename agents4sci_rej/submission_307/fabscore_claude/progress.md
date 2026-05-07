## Session: 2026-04-11

**Purpose:** extraction

**Paper inspected:** 307_Temporal_Motif_Enhanced_Co.pdf (Temporal Motif Enhanced Contrastive learning for anomaly detection on dynamic graphs)

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries

**Summary of extraction:**
- 1 table (Table 1): batch size sensitivity results (batch sizes 8/16/32/64, Validation F1 and Loss)
- 2 figures: Figure 1 (edge connectivity ablation), Figure 2 (learning rate ablation)
- 3 results_section entries: all reporting the headline 15–30% F1 improvement over baselines on four dynamic network datasets

**Next session should:**
- Run analysis/scoring on the extracted results
- Note: The paper's main quantitative claims are all reported as ranges (15–30% F1 improvement) without per-dataset numerical breakdowns or per-method comparison tables

---

## Session: 2026-04-11 (Analysis)

**Purpose:** analysis

**Files/context inspected:**
- `auto_plot_aggregator.py` — generates all performance figures with hardcoded synthetic data, explicitly labeled "For demonstration, creating synthetic data"
- `experiment_ce26e508a9e3438eaf759e5cdcd87bcb_proc_2868/experiment_code.py` — batch size tuning experiment; found (1) only 10 synthetic graphs, (2) validation loss never computed (losses['val'] always empty), (3) model not reinitialized between batch size trials, (4) same data used for train and "validation"
- `experiment_5c7a1498952746f3a9d5135d22480878_proc_2868/experiment_code.py` — learning rate ablation; same issue: losses[lr]['val'] initialized but never populated
- `experiment_bfe0c95097534586b1cad755f3cdd6d8_proc_5657/experiment_code.py` — edge connectivity experiment uses qualitative ["sparse","dense","random"], not numeric "edge_factor" {1,2,4,8} as described in paper
- `experiment_ce26e508a9e3438eaf759e5cdcd87bcb_proc_2868/plotting_code.py` — confirms only train_loss and val_f1 are plotted, no val_loss plot

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with all 13 claims classified

**Summary of classifications:**
- All 13 claims classified as `obvious_hallucination`
- Claims 1-8 (Table 1 batch size results): `experiment_fabrication` — validation loss is never computed (losses['val'] always empty), same data used for train/val, model not reinitialized between batch sizes
- Claims 9-10 (Figures 1-2): `experiment_fabrication` — edge factor parameterization mismatch (paper: {1,2,4,8}, code: ["sparse","dense","random"]); validation loss curves never computed
- Claims 11-13 (15-30% F1 improvement): `data_fabrication` — no real datasets (CollegeMsg, Email-Eu-core, Higgs Twitter, Epinions) are used anywhere in the repository; auto_plot_aggregator.py hardcodes all performance comparison values with "For demonstration" comment

**Recommended next step:**
- Analysis is complete. No execution required as static evidence is decisive for all claims.

