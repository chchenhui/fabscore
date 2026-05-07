# Progress Log

## Session 2 (2026-04-24)
- **Purpose**: analysis (static)
- **Files inspected**:
  - `results/paper.md` — read full paper (LemmaGen)
  - `codex/results.json` — primary artifact: exact time/num_lemmas values matching Table 1
  - `codex/run_experiments.py` — experiment script; plots figures from results.json
  - `results/results.md` — markdown table confirming the same values
  - `results/log.txt` — execution log confirming run completed successfully
  - `results/generation_time.png`, `results/num_lemmas.png` — figure artifacts
- **Files created/updated**:
  - `fabscore_claude/fs_analysis.json` — created with all 8 claims classified
- **Summary**:
  - All 8 claims classified as `static_verifiable`
  - Table 1 values (num_lemmas=1 and rounded time values) match exactly between paper and codex/results.json
  - Both figure PNG artifacts exist in results/ and the underlying data (codex/results.json) + plotting code (run_experiments.py) fully supports them
- **Next step**: No further action needed; all claims resolved statically.

## Session 1
- **Purpose**: extraction
- **Paper inspected**: results/paper.md (LemmaGen: LLM-Assisted Lemma Generation for Scalable Theorem Proving)
- **Files created/updated**:
  - `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries
- **Summary**:
  - Table 1: 6 numerical entries extracted (num_lemmas and time per 3 proof goals: append_nil, append_assoc, rev_inv)
  - Figures: 2 entries (generation_time.png, num_lemmas.png)
  - Results section: empty — the only numerical claim in the Analysis section ("Generation latency varied from 0.04 s to 0.82 s") was deduplicated against Table 1 values
- **Next session**: Run scoring/analysis against extracted results; verify figures if image files are accessible
