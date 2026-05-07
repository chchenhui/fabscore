# Progress Log

## Session: 2026-03-31

**Purpose:** extraction

**Paper/context inspected:** `layerwise_lr_grokking.pdf` — PDF text extracted with `pdf2txt.py`; cross-checked section boundaries and figure/table captions against `latex/template.tex` where the PDF text extraction was ambiguous.

**JSON files created/updated:**
- `fabscore_codex/fs_extracted.json` — extracted tables, figures, and deduplicated numerical claims from the Results body

**Next session should do:**
- No further extraction work is needed unless the paper file changes.
- If downstream review is required, use this extraction as the source record and verify any suspicious claims against the run artifacts.

## Session: 2026-03-31 (analysis)

**Purpose:** analysis

**Files/context inspected:**
- `latex/template.tex` for the paper text, Table 1/Table 2 definitions, figure caption, and the stated LR configurations
- `experiment.py`, `run_1.py`, `run_2.py`, `run_3.py`, `run_4.py`, and `plot.py` for implementation paths and optimizer parameter-group settings
- `run_0/final_info.json`, `run_1/final_info.json`, `run_2/final_info.json`, `run_3/final_info.json`, `run_4/final_info.json` plus the per-seed `final_info_*.json` files for saved metrics
- `notes.txt`, `log.txt`, and `20240801_032548_layerwise_lr_grokking_aider.txt` for baseline provenance and the recorded evolution from uniform LR to layer-wise LR runs
- Existing plot outputs `val_acc_*.png` and the current `fabscore_codex/progress.md`

**JSON files created/updated:**
- `fabscore_codex/fs_analysis.json` — full claim-by-claim static-analysis classification

**Concise classification summary:**
- Classified 25 claims as `static_verifiable`
- Classified 14 claims as `obvious_hallucination`
- Of those 14, `result_fabrication`: claims 22, 24, 25, 26, 27
- Of those 14, `experiment_fabrication`: claims 28-36 because the equal-LR ablation conditions are not implemented anywhere in the repo and several reported values are sourced from the wrong task/run
- Classified Figure 1 (claim 37) as `insufficient_evidence` because only rendered PNGs are present; `plot.py` requires missing `all_results.npy`, and the baseline generator script for `run_0` is missing
- Key finding: Table 1 and Table 2 permutation rows mix `run_3` and `run_4` metrics, while the paper text alternates between calling Run 3 and the `8e-4 / 2e-3 / 3e-3` configuration the final method

**Recommended next step for the next session:**
- If a downstream synthesis/review step is needed, use `fabscore_codex/fs_analysis.json` as the source of truth and explicitly highlight the two main provenance problems: missing intact baseline/figure curve artifacts, and mis-mapped permutation/ablation results across `run_2`, `run_3`, and `run_4`
