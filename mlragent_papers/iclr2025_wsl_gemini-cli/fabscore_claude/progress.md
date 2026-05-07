# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction

**Paper inspected:** `results/paper.md`
Title: "Detecting Neural Backdoors via Permutation-Equivariant Graph Analysis of Weight Space"

**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (10 entries from Table 1), figures (3 entries), and results_section (empty, all numerical results were already captured in Table 1)

**Summary of extraction:**
- Table 1 provided all quantitative results: MLP and GNN detector performance (Accuracy, Precision, Recall, F1 Score, ROC AUC).
- Three figures identified: detector_comparison.png, MLP_training_history.png, GNN_training_history.png.
- No additional numerical results in the results/analysis body text that were not already captured in Table 1; results_section is empty.

**Next session should:**
- Proceed to analysis/scoring phase using `fs_extracted.json`.
- Note: the paper's results are preliminary and inconclusive — the GNN failed to learn (all zeros), and the MLP showed high precision (1.0) but very low recall (0.25).

---

## Session 2 — 2026-04-24
**Purpose:** analysis

**Files inspected:**
- `results/paper.md` — full paper text with Table 1 and Figures 1–3
- `gemini/run_experiment.py` — main experiment script (model zoo generation, MLP/GNN training, evaluation, plotting)
- `results/log.txt` — execution log from two runs; second run (17:27:01–17:27:27) contains complete MLP and GNN evaluation outputs
- `results/results.md` — generated results report containing the numeric performance table and figure references
- `results/detector_comparison.png`, `results/MLP_training_history.png`, `results/GNN_training_history.png` — all three claimed figures exist
- `fabscore_claude/progress.md` — prior session notes

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with all 13 claims classified

**Summary of classifications:**
All 13 claims classified as `static_verifiable`:
- Claims 1–10 (Table 1 metrics): Exact values confirmed in `results/log.txt` (second run evaluation output) and `results/results.md`. No hardcoded values found in code; sklearn metrics used throughout. Log confirms MLP: Acc=0.6250, Prec=1.0000, Rec=0.2500, F1=0.4000, ROC AUC=0.7500; GNN: Acc=0.5000, Prec=0.0000, Rec=0.0000, F1=0.0000, ROC AUC=0.6250.
- Claims 11–13 (Figures): PNG files exist in results/; underlying epoch-by-epoch data is in log.txt (all 20 epochs for both MLP and GNN), which provides the data source for training history plots. Comparison figure data is in results.md table.

**Recommended next step:**
No further action needed — all claims are statically verifiable with strong evidence from log.txt and results.md.
