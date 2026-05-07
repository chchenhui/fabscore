# Progress Log

## Session 1 — 2026-04-23
**Purpose:** extraction

**Paper inspected:** `257_The_Digital_Inbreeding_Cri.pdf` — "The Digital Inbreeding Crisis: Empirical Evidence of LLM Capability Degradation under Multi-Generational Synthetic Training"

**Summary:** 18-page paper (including appendices) presenting empirical validation of "digital inbreeding" in LLMs via a 3×3 factorial experiment (3 conditions × 3 generations). Key metrics: F1, semantic similarity, sentence length, distinct n-grams, Shannon entropy, perplexity, and Cohen's d effect sizes.

**JSON files created:**
- `fabscore_claude/fs_extracted.json` — contains tables (95 entries from Tables 1–3, 5–6), figures (2 entries: Figure 1 and Figure 2), and results_section (1 entry: 12.56 pp and 11.48 pt separation values from Section 4.4 not found in any table).

**Notes:**
- Table 4 (Synthetic Data Generation Parameters) and Table 7 (Computational Time Breakdown) were excluded as experimental setup, not results.
- F1 Performance (Reference) rows in Tables 2 and 3 are duplicates of Table 1 and were not double-counted.
- Results section body had very few unique numerical claims not already captured in tables; only the derived separation values (12.56 pp semantic, 11.48 pt structural) from Section 4.4 were novel.

**Next session should:** perform scoring/analysis of the extracted results if required, or review for completeness against any additional figures in the paper.

---

## Session 2 — 2026-04-23
**Purpose:** analysis (static code audit)

**Files inspected:**
- `257_The_Digital_Inbreeding_Cri.pdf` — paper
- `llm-inbreeding-deterioration-analysis/experiments/exp_20250914_032035/code/main.py` (348 lines)
- `llm-inbreeding-deterioration-analysis/experiments/exp_20250914_032035/code/evaluator.py` (539 lines)
- `llm-inbreeding-deterioration-analysis/experiments/exp_20250914_032035/code/trainer.py` (400 lines)
- `llm-inbreeding-deterioration-analysis/experiments/exp_20250914_032035/code/data_generator.py` (352 lines)
- `llm-inbreeding-deterioration-analysis/experiments/exp_20250914_032035/code/config.py` (111 lines)
- `llm-inbreeding-deterioration-analysis/experiments/exp_20250914_032035/results/experiment_simulation.json`
- `llm-inbreeding-deterioration-analysis/experiments/exp_20250914_032035/results/degradation_metrics.json`
- `llm-inbreeding-deterioration-analysis/experiments/exp_20250914_032035/results/statistical_analysis.json`
- `llm-inbreeding-deterioration-analysis/experiments/exp_20250914_032035/data/generation_0_human_baseline.json`

**JSON files created:**
- `fabscore_claude/fs_analysis.json` — classification of all 98 claims

**Key findings:**

1. **Simulation vs Real LLM Training**: `data_generator.py::_simulate_degraded_output()` programmatically injects degradation (repetition, content shortening, generic phrases) rather than actually training LLMs on synthetic data from previous generations. The paper presents this as a real empirical study.

2. **Heuristic Metrics**: `evaluator.py::_calculate_language_quality()` explicitly states "Simulated perplexity (in real implementation, would use actual language model)" using formula `max(10.0, 50.0 + repetition_penalty - avg_length * 0.1)`. Semantic similarity uses Jaccard word-set overlap, not sentence embeddings.

3. **Statistical Analysis Failure**: `statistical_analysis.json` has NaN for ALL statistical tests (ANOVA f_stat=NaN, all pairwise t_stat=NaN, all effect_size=NaN). With only 1 observation per condition-generation cell, variance-based statistics are undefined. Yet the paper reports precise Cohen's d values with 95% CIs.

4. **Distinct 2-grams > 1 Impossible**: `evaluator.py::_calculate_diversity_metrics()` computes `len(set(bigrams))/len(bigrams)`, bounded [0,1]. Simulation produces values 0.349–0.484. Paper claims 1.106 (Mixed Gen 3) and 1.008 (Exclusive Gen 3) — mathematically impossible from this formula.

5. **Perplexity Trend Conflicts**: Paper claims control perplexity decreases (52.1→51.2) and mixed increases (52.3→53.6). Simulation shows opposite trends (control: 52.6→52.9 increase; mixed: 52.8→51.9 decrease).

6. **F1 scores match simulation**: Table 1 F1 values match experiment_simulation.json exactly (e.g., mixed gen 3 = 0.87507...). These came from simulation, not real LLM training.

**Classification summary:**
- All 98 claims → `obvious_hallucination` with `experiment_fabrication`
- Core reason: The code is a programmatic simulation of text degradation, not real multi-generational LLM training. The experimental procedure conflicts with the paper's claimed methodology at every level (data generation, metrics, statistical analysis).

**Next session:** No further analysis needed. Classification complete.
