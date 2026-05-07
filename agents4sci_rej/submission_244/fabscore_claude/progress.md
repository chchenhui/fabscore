# Progress Log

## Session 1 — 2026-04-23

**Purpose:** extraction

**Paper inspected:** `244_Information_Efficient_Tran.pdf` — "Information-Efficient Transformers via Adaptive Token Pruning"

**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (21 entries from Table 2 and Table 3), figures (4 entries: Figures 1–4), and results_section (2 entries from body text of Sections 7.2 and 11 not already captured by tables).

**Summary of extraction:**
- Table 1 (notation) had no numerical results.
- Table 2 (synthetic validation): 3 methods × 5 metrics = 15 entries.
- Table 3 (SST-2 validation): 2 methods × 3 metrics = 6 entries.
- Figures 1–4 captured by label and caption.
- Results section body: 2 claims with numerical values not already in tables — absolute latency proxy decrease (83.92 → 22.48, 73.21%) and SST-2 8.7pp accuracy drop.

**Next session:** No further extraction needed unless additional result sections or supplementary files are identified.

---

## Session 2 — 2026-04-23

**Purpose:** analysis (static analysis of all 27 claims)

**Files inspected:**
- `Gupta_Shivendra_AGI_Assignment_1/code/implementation.py` — core models, training simulation
- `Gupta_Shivendra_AGI_Assignment_1/code/experiment_runner.py` — orchestrator + ablation data
- `Gupta_Shivendra_AGI_Assignment_1/code/experiment_sst2.ipynb` — SST-2 PyTorch notebook (with real GPU execution output)
- `Gupta_Shivendra_AGI_Assignment_1/results/all_results.json` — synthetic experiment metrics
- `Gupta_Shivendra_AGI_Assignment_1/results/efficiency.json` — FLOPs and latency proxy values
- `Gupta_Shivendra_AGI_Assignment_1/results/results_sst2.json` — SST-2 experiment metrics
- `Gupta_Shivendra_AGI_Assignment_1/results/summary.txt` — quick reference table

**JSON created:**
- `fabscore_claude/fs_analysis.json` — classification of all 27 claims

**Key findings:**

1. **experiment_fabrication (claims 1-15, 26):** `Trainer.fit()` in implementation.py does NOT perform any gradient-based training. Model weights are fixed random matrices. The method artificially boosts accuracy/AUC with hardcoded per-epoch increments (boost=0.015 for proposed, 0.005 for attn_prune, 0.0 for baseline; AUC boost=0.02 for ALL models). Final reported metrics are explicitly overwritten with boosted history values (`final_eval['acc'] = history['val_acc'][-1]`). The paper presents these as actual transformer training results.

2. **FLOPs conflict (claims 9, 14):** `efficiency.json` stores flops_proposed=262,144 and flops_full=1,048,576 → ratio=0.25×. The paper claims 0.63× (37.5% reduction via LaTeX macro `\flopsred{37.5\%}`). The code uses `estimate_flops(avg_kept=32, d=64, n_layers=2)` which treats BOTH attention layers as using only k=32 tokens, while the correct formula (layer1 at full L, layer2 at k) would give 0.63×.

3. **SST-2 is real (claims 16-21, 27):** `experiment_sst2.ipynb` contains actual PyTorch GPU training of DistilBERT+EntropyGate on SST-2 (1 epoch, 4210 batches, ~12 min). Notebook execution output shows exact values matching paper: Baseline Acc=0.9140, AUC=0.9725; Proposed Acc=0.8268, AUC=0.8806, FLOPs=9.04e+07 (40.1% reduction).

4. **Figures (claims 22-25):** All PNG/PDF exist; backing data files present (all_results.json, val_true.npy, val_score_proposed.npy).

**Classification summary:**
- `obvious_hallucination` (experiment_fabrication): 16 claims (1-15, 26)
- `static_verifiable`: 11 claims (16-25, 27)

**Recommended next step:** No further analysis needed. All 27 claims classified.
