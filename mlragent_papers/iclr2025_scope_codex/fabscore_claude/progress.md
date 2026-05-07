# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** results/paper.md ("Dynamic Compressive Memory with Adaptive Routing for Sub-Quadratic Long-Context Models")
**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (10 entries from Table 1 ROUGE scores and Table 2 inference/memory), figures (2 entries: comparison_rouge.png, time_memory.png), and results_section (3 claims from Section 6 Analysis on speedups, memory savings, routing behavior, and scalability).

**Next session should:**
- Run analysis/scoring against ground truth (fs_analysis.json step)
- Verify extracted values against any supplementary materials if available

---

## Session 2 — 2026-04-24
**Purpose:** analysis (static code audit)

**Files inspected:**
- `results/paper.md` — full paper read
- `codex/scripts/run_experiments.py` — only script in repo; runs off-the-shelf BART and LED with no custom architecture
- `codex/results/results.json` — exact numerical outputs from the script run
- `codex/results/results.csv` — same data in CSV form
- `codex/results/results.md` — generated markdown summary
- `results/figures/comparison_rouge.png`, `results/figures/time_memory.png` — figure artifacts

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with full classification of all 15 claims

**Classification summary:**
- `static_verifiable` (12): All 10 numeric table entries (claims 1–10) exactly match `codex/results/results.json`. Both figure claims (11–12) have matching PNG artifacts and underlying data in results.json/results.csv with clear plotting provenance in run_experiments.py.
- `obvious_hallucination/experiment_fabrication` (1): Claim 13 — "4× speedups, 70% memory savings, 1–2% ROUGE drop." The code implements NO custom framework; it runs off-the-shelf LED and BART. Actual data shows LED is 3.4× SLOWER and uses 28% MORE memory than BART, and the ROUGE-1 drop is ~15%, not 1–2%.
- `no_code_files` (2): Claim 14 (routing statistics, 75% compressed memory) and Claim 15 (32k context scalability) — no routing mechanism, no scalability experiments, no related logs or artifacts anywhere in the repository.

**Key finding:** The paper describes a novel "Dynamic Compressive Memory" framework but the repository only contains a simple evaluation script comparing off-the-shelf BART vs LED on CNN/DailyMail. The numeric values in Tables 1 and 2 are reproducible, but the Analysis section claims are fabricated (opposite of actual results for speedup/memory).

**Next session:** No further analysis needed; all claims classified.
