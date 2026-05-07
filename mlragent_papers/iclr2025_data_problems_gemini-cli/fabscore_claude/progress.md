# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction

**Paper inspected:** `results/paper.md`
- Title: "Generative Data Symbiosis: Mitigating Model Collapse through Co-Evolving Foundation Models"
- Paper evaluates a co-evolutionary framework (Generator + Student) on AG News classification across 3 training generations, compared to Recursive Collapse, Static Synthetic, and Real Data Upper Bound baselines.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (12 entries from Table 1), figures (2 entries: accuracy_comparison.png and loss_curves.png), and results_section (empty — all numerical body-text values were duplicates of Table 1 entries).

**Notes:**
- The only explicit numerical result in the body text of Section 5/6 beyond what's in Table 1 was "26.8% accuracy" in the Analysis section, which duplicates Table 1, Recursive Collapse, Final Accuracy: 26.80%. Correctly excluded from results_section.
- results_section is empty because all quantitative claims in the text refer to values already captured in Table 1.

**Next session should:**
- Proceed to scoring/analysis phase using `fs_extracted.json`.

## Session 2 — 2026-04-24
**Purpose:** analysis (static analysis of all 14 claims)

**Files inspected:**
- `results/paper.md` — paper with Table 1 and figure descriptions
- `gemini/results.json` — raw experiment results (accuracy arrays and train_loss arrays for all 4 methods)
- `gemini/plotting.py` — generates accuracy_comparison.png and loss_curves.png from results.json
- `gemini/run_experiment.py`, `gemini/models.py`, `gemini/generation.py`, `gemini/data_utils.py` — experiment pipeline
- `results/accuracy_comparison.png`, `results/loss_curves.png` — generated figure artifacts
- `results/results.md` — results summary document

**Key findings:**
- All 12 table claims (claims 1–12) match `gemini/results.json` exactly:
  - Real Data Upper Bound: [0.252, 0.86, 0.848, 0.84] → 25.2%→84.0%, +58.8%
  - Static Synthetic: [0.236, 0.466, 0.53, 0.41] → 23.6%→41.0%, +17.4%
  - Recursive Collapse: [0.268, 0.268, 0.268, 0.268] → 26.8%→26.8%, +0.0%
  - Generative Symbiosis: [0.192, 0.25, 0.328, 0.25] → 19.2%→25.0%, +5.8%
- Both figure PNGs exist in `results/` and `plotting.py` generates them directly from `results.json` data.
- Generative Symbiosis train_loss [3.4947, 3.9510, 3.5867, 1.1527, 7.5061, 0.0012] is non-monotonic, consistent with claim 14's description.

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with all 14 claims classified as `static_verifiable`

**Classification summary:**
- static_verifiable: 14 (all claims)
- All other buckets: 0

**Next session should:**
- Proceed to execution phase if needed (not required here — all claims statically verified).
