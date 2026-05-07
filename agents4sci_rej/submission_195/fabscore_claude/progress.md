# Progress Log

## Session 1 — 2026-04-24

**Purpose:** extraction

**Paper inspected:** `195_How_Large_Language_Models_.pdf` — "How Large Language Models Handle Arithmetic: A Benchmark Study" evaluating 10 LLMs on a 211-problem arithmetic benchmark (MATH-211) using step-by-step and direct answer prompting.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — Created. Contains:
  - `tables`: empty (paper has no formal numbered tables; all quantitative results are in figures and body text)
  - `figures`: 3 entries (Figure 1 pipeline diagram, Figure 2 comprehensive performance comparison, Figure 3 speed vs accuracy trade-off)
  - `results_section`: 34 entries covering accuracy values per model per prompting mode, speed improvement factors, response times, and operation-specific findings

**Key findings extracted:**
- Claude-Sonnet-4 and Llama-4-Maverick both achieved 100% accuracy in step-by-step mode
- Qwen3-0.6B suffered catastrophic format compliance failure in direct mode (1.4% accuracy)
- Qwen3-8B achieved the highest speed improvement: 39.8x with direct prompting
- Accuracy loss from direct prompting was typically less than 1%

**Next session should:**
- Run scoring/evaluation on `fs_extracted.json` if required
- Verify any figures that were not fully readable in this extraction
- No further extraction tasks anticipated unless the paper is revised

---

## Session 2 — 2026-04-24

**Purpose:** analysis (static code audit of all 37 extracted claims)

**Files inspected:**
- `llm_arithmetic/papers/generate_figures.py` — figure generation script with hardcoded performance data
- `llm_arithmetic/results/phase1/math401/evaluation_summary_*.json` (8 summary files) — contain model accuracy and response times for all 10 models in both evaluation modes
- `llm_arithmetic/results/phase1/math401/huggingface_Qwen_Qwen3-0.6B_direct_answer_20250914_010312.json` — per-operation breakdown for Qwen3-0.6B direct mode (confirmed 0% addition accuracy)
- `llm_arithmetic/experiments/phase1/run_evaluation.py` — evaluation entry point
- `llm_arithmetic/src/llm_arithmetic/evaluation/base_evaluator.py` — core evaluation logic

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — Created. Full classification of all 37 claims.

**Classification summary:**
- `static_verifiable`: 36 claims (indices 1–36)
  - All accuracy values (step-by-step and direct modes) verified against evaluation summary JSON files
  - All response times verified against evaluation summary JSON files and generate_figures.py
  - All speed improvement ratios (Qwen3-8B 39.8x, Qwen3-235B 28.7x, Llama-4-Maverick 14.4x, Qwen3-0.6B 5.5x) confirmed via step_time/direct_time ratios in JSON files
  - Qwen3-0.6B 0% addition accuracy in direct mode confirmed from per-operation breakdown JSON
  - All three figures traceable to generate_figures.py with values cross-checked to JSON result files
- `obvious_hallucination` (`result_fabrication`): 1 claim (index 37)
  - Claim: "GPT-4o-Mini achieved 29.8x speed improvement through direct prompting"
  - generate_figures.py line 96 shows speed_improvements[8] (GPT-4o-Mini) = 8.54x (calculated as 3.50s/0.41s)
  - The value 29.75x (≈ 29.8x) belongs to speed_improvements[7] (Qwen3-4B), not GPT-4o-Mini
  - This is a deliberate misattribution of Qwen3-4B's speed improvement to GPT-4o-Mini in the paper

**Key finding:** Only one claim (index 37 / paper claim #34 about GPT-4o-Mini's 29.8x speed improvement) is fabricated — it is actually Qwen3-4B's 29.75x value misattributed to GPT-4o-Mini. All other 36 claims are directly verifiable from static artifacts in the repository.

**Recommended next step:** No further analysis needed. The classification is complete.
