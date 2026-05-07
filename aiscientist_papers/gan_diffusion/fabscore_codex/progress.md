## Session 2026-04-13 - extraction

- Purpose: extraction
- Paper/context inspected: `gan_diffusion.pdf` directly, focusing on the Results section, Tables 1-4, and Figure 1-2 captions from the PDF.
- JSON files created or updated: `fabscore_codex/fs_extracted.json`
- Next session should do: use this extraction output for any downstream scoring, summary, or verification pass.

## Session 2026-04-13 - analysis

- Purpose: analysis
- Files/context inspected: `gan_diffusion.pdf` via `latex/template.tex`; `experiment.py`; `run_1.py`, `run_2.py`, `run_3.py`, `run_4.py`, `run_5.py`; `datasets.py`; `discriminator.py`; `plot.py`; `run_0/final_info.json` through `run_5/final_info.json`; `notes.txt`; `20240716_185529_gan_diffusion_aider.txt`; existing `fabscore_codex/progress.md`.
- JSON files created or updated: `fabscore_codex/fs_analysis.json`
- Classification summary: 66 total claims. `obvious_hallucination`: 48 claims (all Table 2-4 metrics), all as `experiment_fabrication`, because the run scripts train the discriminator on `noisy` samples instead of denoised/generated samples and reuse one dataset object for training, evaluation loss, and KL divergence despite the paper claiming generated-sample feedback and a train/eval split. `insufficient_evidence`: 18 claims (all Table 1 metrics and both figures), because `run_0` is backed only by artifact files with no intact baseline implementation path, and the figure plotting inputs `all_results.pkl` are missing from the labeled run directories.
- Recommended next step: if a later session is allowed to go beyond static analysis, first recover the exact baseline `run_0` implementation and the missing per-run `all_results.pkl` artifacts before attempting any execution-based verification; otherwise use `fabscore_codex/fs_analysis.json` as the final static-audit record.
