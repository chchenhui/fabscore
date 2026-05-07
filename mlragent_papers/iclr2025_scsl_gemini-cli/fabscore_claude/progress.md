# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** results/paper.md — "Spurious-Correlation-Aware Adapters (SCA-Adapters): Efficiently Robustifying Foundation Models via Orthogonal Gradient Projection"
**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` (created)

**Summary of extraction:**
- Table 1: 9 entries covering Average Accuracy, Worst-Group Accuracy (WGA), and Trainable Parameters for zero_shot, lora, and sca_adapter.
- Figures: 3 bar-chart comparison figures (wga_comparison.png, avg_acc_comparison.png, params_comparison.png).
- Results section: No unique numerical claims in the body text beyond what is already captured in Table 1. All quantitative statements in Section 6 body text directly restate Table 1 values.

**Next session should:**
- Proceed to analysis/scoring phase using the extracted results.

## Session 2 — 2026-04-24
**Purpose:** analysis
**Files inspected:**
- `results/paper.md` — full paper text
- `gemini/outputs/results.json` — saved experiment outputs
- `gemini/run_experiment.py` — main experiment script
- `results/results.md` — generated summary
- `results/wga_comparison.png`, `results/avg_acc_comparison.png`, `results/params_comparison.png` — figure artifacts

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` (created)

**Summary of classifications:**
All 12 claims classified as `static_verifiable`:
- Claims 1–9 (Table 1): All values match exactly (within standard rounding) in `gemini/outputs/results.json`:
  - zero_shot: avg_acc=0.745, wga=0.05882≈0.059, params=0
  - lora: avg_acc=0.3075≈0.308, wga=0.06557≈0.066, params=491520
  - sca_adapter: avg_acc=0.755, wga=0.0, params=201348
- Claims 10–12 (Figures): Underlying data in `gemini/outputs/results.json` supports all three figure descriptions; PNG artifacts exist in `results/`; `run_experiment.py`'s `generate_results()` shows consistent plotting logic.

**Recommended next step:**
- No further action needed; all claims are statically verified against stored artifacts.
