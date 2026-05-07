# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** results/paper.md — "Influence-Driven Selective Unlearning for Trusted Large Language Models"
**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (4 entries from Table 1), figures (2 entries: loss_curves.png, canary_prob.png), and results_section (2 entries: ~15.4% canary probability reduction, <25% retraining cost).

**Summary of findings:**
- Paper is a proof-of-concept on SST2 with one injected canary. Main results: projection unlearning reduces canary class probability from 0.3038 to 0.2569 (~15.4% relative reduction); both methods yield 0% test accuracy in single-epoch setup; projection method costs <25% of full retraining time.

**Next session should:**
- Perform analysis/scoring of the extracted results (fs_analysis.json).

---

## Session 2 — 2026-04-24
**Purpose:** analysis (static code audit)
**Files inspected:**
- `results/paper.md` — full paper read
- `codex/run_experiment.py` — full implementation read
- `results/results.csv` — numerical results
- `results/log.txt` — execution log
- `results/loss_curves.png`, `results/canary_prob.png` — figure artifacts

**Files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with analysis classifications

**Key findings:**
- **Critical model mismatch**: Paper claims "GPT-2 medium (345M) on 8×A100 GPUs" but `run_experiment.py` uses `distilbert-base-uncased` (66M, discriminative encoder). This is a clear `experiment_fabrication`.
- **Evaluation method mismatch**: Paper says canary probability computed "via beam-search" but code uses `torch.softmax` on classifier logits — fundamentally different approach.
- `results/results.csv` and `log.txt` confirm the reported numbers (0.3038, 0.2569, accuracy=0.0) were actually generated, but from the wrong model.
- No profiling/timing code exists for the <25% retraining time claim (claim 8) → `no_code_files`.

**Classification summary:**
- `obvious_hallucination` (experiment_fabrication): claims 1, 2, 3, 4, 5, 6, 7 (all affected by DistilBERT vs GPT-2 medium mismatch)
- `no_code_files`: claim 8 (efficiency/timing — no profiling code)

**Next session should:**
- No further action needed; analysis is complete.
