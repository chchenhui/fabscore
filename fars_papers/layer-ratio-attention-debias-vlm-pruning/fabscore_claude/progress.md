## Session: 2026-04-24

**Purpose:** extraction

**Paper inspected:** paper.pdf — "Deep-Layer Attention Pruning for Vision-Language Models" (FARS / Analemma). The paper proposes using deeper-layer (L12) attention for visual token pruning in VLMs, avoiding offline calibration. Evaluated on RefCOCO family benchmarks with InternVL2.5-8B.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (82 entries from Tables 1–3), figures (3 entries: Figures 1–3), and results_section (5 entries from Sections 4.2 and 4.5).

**Summary of extraction:**
- Table 1: Main grounding accuracy results across 8 RefCOCO splits for No Pruning, FastV, D2Pruner, and Ours (L12).
- Table 2: Ablation of ratio-based debiasing at L4 and L12 with various Ks values and weighted combo.
- Table 3: Phase-0 diagnostic metrics (prompt stability, position correlation, entropy) across layers 1–12.
- Figures 1–3: architecture comparison, sensitivity plot, spatial token retention analysis.
- Results section: retained/relative performance percentages and IoU comparisons not directly in tables.

**Next session should:**
- Perform analysis/scoring (fs_analysis.json) against paper claims if needed.
- Verify the extraction against any supplementary materials if they exist.

---

## Session: 2026-04-24 (analysis)

**Purpose:** Static analysis — classify all 90 extracted claims against repository evidence.

**Files inspected:**
- `paper.pdf` — read all 8 pages
- `exp/EXPERIMENT_RESULTS/d2pruner_baseline/RESULTS.json` — Table 1 baselines (No Pruning, FastV, D2Pruner)
- `exp/EXPERIMENT_RESULTS/online_prior/RESULTS.json` — Table 1 Ours (L12), raw MIS at L4
- `exp/EXPERIMENT_RESULTS/ablation_amid_only/RESULTS.json` — Full Table 2 ablation data (per-dataset averages and deltas)
- `exp/EXPERIMENT_RESULTS/phase0_diagnostic/RESULTS.json` — Table 3 per-layer diagnostic metrics
- `exp/EXPERIMENT_RESULTS/layer_choice_stress_test/RESULTS.json` — Figure 2 / Claim 89 sensitivity data
- `exp/EXPERIMENT_RESULTS/visualization_analysis/RESULTS.json` — Figure 3 / Claim 90 spatial IoU data

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with all 90 claims classified as static_verifiable

**Summary of classifications:**
- **static_verifiable: 90 / 90** — All claims have exact numerical matches in stored RESULTS.json files.
  - Claims 1–36 (Table 1): matched to d2pruner_baseline/RESULTS.json and online_prior/RESULTS.json.
  - Claims 37–64 (Table 2): matched to ablation_amid_only/RESULTS.json (per-dataset averages and delta values).
  - Claims 65–82 (Table 3): matched to phase0_diagnostic/RESULTS.json (values rounded to 3 sig figs in paper).
  - Claims 83–85 (Figures 1–3): underlying data in respective RESULTS.json files confirms all numerical claims in figure captions.
  - Claims 86–90 (results section): derived statistics confirmed by arithmetic from stored JSON values.

**Key observations:**
- The repository stores pre-computed RESULTS.json files for every experiment that exactly reproduce the paper's table values.
- Table 3 values in the paper are rounded versions of the stored float values (e.g., 0.9966 → 0.997, 5.896 → 5.90).
- The paper notes that absolute scores are ~15 points below published D2Pruner numbers (different evaluation protocol) but all within-paper comparisons are internally consistent and supported by the artifacts.

**Next session:** No further action needed; all claims fully supported by static evidence.
