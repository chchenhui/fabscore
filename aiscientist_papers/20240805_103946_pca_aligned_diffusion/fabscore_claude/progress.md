# Progress Log

## Session 1
- **Purpose**: extraction
- **Paper inspected**: pca_aligned_diffusion.pdf (10-page PDF, AI Scientist generated preprint on PCA-Aligned Diffusion for low-dimensional generative models)
- **JSON files created**: `fabscore_claude/fs_extracted.json`
- **Summary**:
  - Extracted 80 table entries from Tables 1–5 (Baseline, PCA-Aligned, Controlled Generation, Cosine Beta Schedule, Expanded Component Weights), covering Training Time, Eval Loss, Inference Time, and KL Divergence for 4 datasets (circle, dino, line, moons).
  - Extracted 1 figure (Figure 1, placeholder caption "PLEASE FILL IN CAPTION HERE"). Other figures referenced as "Figure ??" had no valid labels or captions.
  - No results_section entries: all numerical claims in the Results section body text were summaries of table data already captured in the tables.
- **Next session**: Run analysis/scoring on fs_extracted.json.

## Session 2 (2026-04-22)
- **Purpose**: analysis
- **Files inspected**:
  - `run_0/final_info.json` — baseline metrics (Table 1)
  - `run_1/final_info.json` — PCA-Aligned metrics (Table 2)
  - `run_2/final_info.json` — Controlled Generation metrics (Table 3)
  - `run_3/final_info.json` — Cosine Beta Schedule metrics (Table 4)
  - `run_4/final_info.json` — Expanded Component Weights metrics (Table 5)
  - `experiment.py`, `datasets.py`, `plot.py`, `ema_pytorch.py`, `run_1.py`–`run_5.py` (via Explore agent)
  - `pca_aligned_diffusion.pdf` (via exploration)
  - `fabscore_claude/fs_extracted.json`
- **JSON files created**: `fabscore_claude/fs_analysis.json`
- **Summary**:
  - All 80 table claims (indices 1–80) classified as `static_verifiable`: every reported value was confirmed against the corresponding `run_X/final_info.json` file to 4 decimal place precision. Percentage changes in Tables 2–5 are computed relative to the immediately preceding table (each run vs the prior run), which is consistent with the actual JSON values.
  - Notable: Table 3, Moons KL (+19651.8%) is computed relative to Table 2's moons KL (0.0085), not the absolute baseline — this is internally consistent.
  - Figure 1 (index 81) classified as `insufficient_evidence`: paper caption is a placeholder ("PLEASE FILL IN CAPTION HERE") with no specific verifiable numerical claim.
- **Next session**: No further analysis required; all table claims statically verified.
