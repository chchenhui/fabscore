# Progress Log

## Session 1 — 2026-04-22

**Purpose:** extraction

**Paper inspected:** `199_Multi_LLM_and_Multi_Prompt.pdf` — "Multi-LLM and Multi-Prompt Strategies for COVID-19 Infodemic Detection in Chinese Social Media: An Empirical Evaluation"

**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — Created with extracted tables, figures, and results_section claims.

**Summary of extraction:**
- **Tables:** 20 entries from Table 2 (Performance changes: composite score S1→S5 and ambiguity S1→S5 for each of 10 LLMs). Table 1 is a label-definition table with no performance metrics.
- **Figures:** 6 entries — Figure 1 (aggregate bar chart), Figure 2 (heatmap of 4 metrics × 10 models × 5 strategies), Figure 3 (box plots across strategies), Figure 4 (quarterly strict accuracy trends), Figure A1 (model ranking by composite score), Figure A2 (composite comparison multi-panel).
- **Results section:** 11 entries covering aggregate strict/lenient accuracy (48.0%/61.2%), ambiguity rates (<2% overall, median 1.5%), expert-only vs. context-augmented prompt medians (65-70% vs. 55-60% lenient), error rate increases with context (5-10 pp), persona comparison differences (GPT-4o -10.0 pp, Gemini -8.1 pp), temporal trends (peak ≈75-82% in 2022 Q2, stabilization 40-55%), and model-specific error rates.

**Next session should:**
- Run analysis/scoring against reference if available.
- No further extraction needed unless paper is revised or supplemental data is added.

---

## Session 2 — 2026-04-22

**Purpose:** analysis (static)

**Files/context inspected:**
- All 50 JSON log files in `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Logs/` (10 models × 5 strategies, 640 records each)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Code/improved_classification_stats.py` (metric formulas: composite_score=(2*N1+N2-N4-2*N5)/N, strict=N1/N, lenient=(N1+N2)/N, error=(N4+N5)/N, ambiguity=N3/N)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Code/llm_evaluation_system_batch.py` (confirms log record fields: only judgment_classification, not original post dates)
- `fabscore_claude/progress.md` (session 1)

**Files created:**
- `fabscore_claude/fs_analysis.json` — Final analysis classification (37 claims)

**Summary of classifications:**
- **static_verifiable (27):** All 20 Table 2 claims (indices 1-20) verified EXACTLY from log data using composite score formula. Results-section claims 27 (48.0%/61.2%), 28 (ambiguity 1.90%<2%), 30 (Gemini 13.75 pp≈13.8 pp strict drop), 34 (median ambiguity 1.48%≈1.5%), 35 (DeepSeek-V3/Gemini ≤1% across multiple regimes), 36 (GPT-4o/Llama/Mistral ~50% error), 37 (Doubao/Gemini/GLM/Qwen3-Think 14-32% error) all confirmed from log data.
- **execution_required (8):** Figures 1, 2, 3, A1, A2 (indices 21-23, 25-26) — analysis script present but no pre-generated figures; results-section claims 29 (S2/S3 lenient ~65-70% vs S4/S5 ~55-60% — approximate), 31 (GPT-4o -10.0 pp, Gemini -8.1 pp lenient difference — exact comparison unclear), 32 (Llama +1.1 pp, Mistral +0.9 pp — all static comparisons give large negatives, methodology unclear).
- **no_code_files (2):** Figure 4 (index 24) and claim 33 (quarterly trends) — log files contain only 2025 evaluation timestamps, not original Weibo post dates; SQLite database absent; no quarterly analysis code in repository.

**Key findings:**
- All Table 2 values match log data exactly (10 models × S1→S5 composite scores and ambiguity rates)
- Global metrics from 32,000 predictions exactly reproduce paper's 48.0%/61.2% claim
- Claims 31-32 about per-model lenient accuracy differences (+1.1 pp Llama, +0.9 pp Mistral) cannot be reproduced by any static comparison — execution required
- Quarterly temporal analysis is impossible without the original dataset with post dates

**Recommended next step:**
- Execute `improved_classification_stats.py` from the Logs directory to generate figures and verify claims 21-23, 25-26, 29
- Investigate what comparison methodology produces the -10.0 pp / -8.1 pp / +1.1 pp / +0.9 pp values in claims 31-32

---

## Session 3 — 2026-04-22

**Purpose:** execution (claim 21 — Figure 1)

**Files/context inspected:**
- `fabscore_claude/progress.md` (sessions 1-2)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Code/improved_classification_stats.py` (lines 621-779: `generate_performance_chart` function)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Logs/` (50 JSON files)

**Execution artifacts created:**
- `fabscore_claude/workspace/verify_figure1.py` — custom verification script using correct field name
- `fabscore_claude/workspace/figure1_verified.png` — reproduced Figure 1
- `fabscore_claude/workspace/figure1_verified.pdf` — reproduced Figure 1 (PDF)
- `fabscore_claude/workspace/claim_21_command_output.txt` — full command output

**Key findings:**
- Official script (`improved_classification_stats.py`) has a field name bug: looks for `ai_judgment_classification`/`llm_judgment_classification` but actual field is `judgment_classification`. This causes the script to produce all-zero output.
- Custom script correctly reads all 50 log files (32,000 records) and produces accurate metrics
- Generated figure shows aggregate performance across all 50 model-prompt conditions (10 models × 5 strategies) — exactly matching paper's Figure 1 description
- Global metrics: strict=48.0%, lenient=61.2% — match paper's claims exactly
- Per-model bar chart with composite score line is correctly structured

**Verdict summary (claim 21):** VERIFIED — The underlying data in the 50 log files supports the figure. Figure 1 was successfully regenerated from scratch using the log data. Aggregate performance (48.0% strict, 61.2% lenient) matches the paper, and the figure structure (10 models + "all models" aggregate) corresponds to "50 model-prompt conditions."

**Next session should:**
- Verify claims 22, 23, 25, 26 (other figures), and 29, 31, 32 (results section claims requiring execution)

---

## Session 4 — 2026-04-22

**Purpose:** execution (claim 22 — Figure 2: heatmap)

**Files/context inspected:**
- `fabscore_claude/progress.md` (sessions 1-3)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Logs/` (50 JSON log files, all present)
- Note: pre-existing `evaluation_metrics_heatmap.png` in Logs/ (not sufficient alone for Verified verdict)

**Execution artifacts created:**
- `fabscore_claude/workspace/verify_figure2.py` — script to compute per-model × per-strategy metrics
- `fabscore_claude/workspace/figure2_verified.png` — reproduced heatmap from scratch
- `fabscore_claude/workspace/figure2_verified.pdf` — reproduced heatmap (PDF)
- `fabscore_claude/workspace/claim_22_command_output.txt` — full command output

**Key findings:**
- All 50 log files present and readable (640 records each, using `judgment_classification` field)
- Matrix shape: 10 models × 5 strategies — matches claim exactly
- 4 metrics computed: Strict Accuracy (16.9%–86.2%), Lenient Accuracy (39.2%–90.0%), Error Rate (8.8%–58.3%), Ambiguity Rate (0.2%–6.6%)
- 5 strategies confirmed: no_guide, public_health_expert, respiratory_doctor, detailed_public_health, detailed_respiratory
- Error Rate uses labels 4 and 5 (N4+N5)/N; Ambiguity uses label 3 (N3/N) — matches claim description
- Heatmap structure: rows=models, columns=strategies — matches claim description
- Colormaps appropriate: higher strict/lenient = brighter; higher error = darker — consistent with claim

**Verdict summary (claim 22):** VERIFIED — Successfully regenerated Figure 2 from the 50 log files. The underlying data confirms: 4 panels, 10 model rows, 5 strategy columns, correct metric definitions (error=labels 4/5, ambiguity=label 3), and the described visual encoding (brighter=higher accuracy, darker=higher error).

**Next session should:**
- Verify claims 23, 25, 26 (Figure 3, A1, A2) and 29, 31, 32 (results section claims)

---

## Session 5 — 2026-04-22

**Purpose:** execution (claim 23 — Figure 3: box plots)

**Files/context inspected:**
- `fabscore_claude/progress.md` (sessions 1-4)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Code/improved_classification_stats.py` (lines 781-884: `generate_strategy_boxplot` function)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Logs/` (50 JSON log files, all present)
- Note: pre-existing `strategy_metrics_boxplot.png` in Logs/ (not sufficient alone for Verified verdict)

**Execution artifacts created:**
- `fabscore_claude/workspace/verify_figure3.py` — verification script computing per-strategy cross-model distributions
- `fabscore_claude/workspace/figure3_verified.png` — reproduced Figure 3 box plots from scratch
- `fabscore_claude/workspace/figure3_verified.pdf` — reproduced Figure 3 (PDF)
- `fabscore_claude/workspace/claim_23_command_output.txt` — full command output

**Key findings:**
- All 50 log files readable; 640 records each using `judgment_classification` field
- Per-strategy medians (cross-model distributions):
  - S1 no_guide: strict=37.58%, lenient=58.05%, error=38.05%, ambig=1.88%, composite=0.44
  - S2 public_health_expert: strict=50.55%, lenient=64.22%, error=33.98%, ambig=1.09%, composite=0.74
  - S3 respiratory_doctor: strict=45.00%, lenient=59.61%, error=38.28%, ambig=1.25%, composite=0.52
  - S4 detailed_public_health: strict=44.61%, lenient=58.67%, error=39.92%, ambig=1.80%, composite=0.45
  - S5 detailed_respiratory: strict=41.48%, lenient=52.89%, error=45.47%, ambig=2.11%, composite=0.33
- Claim checks:
  - Expert-only (S1-S3) avg strict > with-time/id (S4-S5): 44.38% > 43.05% ✓ (marginal for strict)
  - Expert-only avg lenient > with-time/id: 60.62% > 55.78% ✓
  - Expert-only avg error < with-time/id: 36.77% < 42.70% ✓ (clear difference, especially S5=45.47%)
  - Ambiguity: S4=1.80% ≈ similar to S1-S3 avg 1.41%; S5=2.11% slightly higher ← "similar or slightly lower" is approximate
  - Zero reference line in composite score: confirmed hard-coded in script (line 868-869)
- All qualitative trends described in the claim are supported by the data

**Verdict summary (claim 23):** VERIFIED — Successfully regenerated Figure 3 from 50 log files. The cross-model distributions confirm: expert-only strategies (S1-S3) show higher strict/lenient medians and lower error medians than time/id strategies (S4-S5); ambiguity values are broadly similar; composite score panels include zero reference. The box plot structure (5 metrics, 5 strategies, cross-model distributions) exactly matches the claim description.

**Next session should:**
- Verify claims 25, 26 (Figures A1, A2) and 29, 31, 32 (results section claims)

---

## Session 6 — 2026-04-22

**Purpose:** execution (claim 25 — Figure A1: Model ranking by average composite score)

**Files/context inspected:**
- `fabscore_claude/progress.md` (sessions 1-5)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Code/improved_classification_stats.py` (lines 1201-1287: `generate_model_overall_scores` function)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Logs/` (50 JSON log files, all present)

**Execution artifacts created:**
- `fabscore_claude/workspace/verify_figureA1.py` — verification script computing per-model mean composite scores
- `fabscore_claude/workspace/figureA1_verified.png` — reproduced Figure A1 from scratch
- `fabscore_claude/workspace/figureA1_verified.pdf` — reproduced Figure A1 (PDF)
- `fabscore_claude/workspace/claim_25_command_output.txt` — full command output

**Key findings:**
- All 50 log files readable; 640 records each using `judgment_classification` field
- Per-model mean composite scores across 5 strategies:
  1. qwen3-235b-a22b-thinking-2507: 1.3909
  2. doubao-seedream: 1.1263
  3. gemini-2.5-pro: 0.9403
  4. glm-4-airx: 0.6616
  5. deepseek-v3: 0.5962
  6. qwen3-235b: 0.4388
  7. deepseek-r1-250528: 0.3997
  8. mistral-large-latest: 0.0141
  9. chatgpt-4o-latest: -0.0141
  10. llama-4-maverick: -0.0153
- Claim checks:
  - Bars rank models by mean composite score ✓
  - Higher is better ✓
  - Ambiguous (N3) = 0 in composite formula ✓
  - Correct rejections (N1, weight=2) > mild judgments (N2, weight=1) ✓
  - Zero line present (break-even indicator) ✓
  - Some models (chatgpt-4o-latest, llama-4-maverick) fall below zero, consistent with claim ✓
- Official script has field-name bug but formula logic is correct

**Verdict summary (claim 25):** VERIFIED — Successfully regenerated Figure A1 from 50 log files. The underlying data confirms: models ranked by mean composite score, zero line present, ambiguous=0 in formula, correct rejections weighted more than mild judgments. All elements of the claim description are supported.

**Next session should:**
- Verify claim 26 (Figure A2) and 29, 31, 32 (results section claims)

---

## Session 7 — 2026-04-22

**Purpose:** execution (claim 26 — Figure A2: Comprehensive analysis dashboard)

**Files/context inspected:**
- `fabscore_claude/progress.md` (sessions 1-6)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Code/improved_classification_stats.py` (lines 1018-1199: `generate_comprehensive_analysis` function)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Logs/` (50 JSON log files, all present)

**Execution artifacts created:**
- `fabscore_claude/workspace/verify_figureA2.py` — verification script reproducing the dashboard from scratch
- `fabscore_claude/workspace/figureA2_verified.png` — reproduced Figure A2 from scratch
- `fabscore_claude/workspace/figureA2_verified.pdf` — reproduced Figure A2 (PDF)
- `fabscore_claude/workspace/claim_26_command_output.txt` — full command output

**Key findings:**
- Official script has field name bug (`ai_classification_total`/`llm_classification_total` vs actual `judgment_classification`)
- Custom script correctly reads all 50 log files using `judgment_classification` field
- 4 strategies used for the comprehensive analysis (no_guide excluded): detailed_public_health, public_health_expert, detailed_respiratory, respiratory_doctor → 40 (model,strategy) pairs
- 4-panel layout confirmed:
  - Top-left: Box plot (Strategy Performance Distribution) — strict accuracy distributions per strategy
  - Top-right: Heatmap (Model Performance Heatmap) — 10 models × 4 strategies
  - Bottom-left: Bar chart (Average Performance by Strategy) — horizontal bars
  - Bottom-right: Pie chart (Judge-label Composition) — Label 1: 48.7%, Label 2: 12.2%, Label 3: 1.6%, Label 4: 20.1%, Label 5: 17.5%
- All 4 chart types described in claim 26 are present and generated from real data

**Verdict summary (claim 26):** VERIFIED — Successfully regenerated Figure A2 from 50 log files. The 4-panel comprehensive analysis dashboard includes exactly the elements described: heatmap (model × strategy), box plot (strategy distributions), bar chart (average performance), and pie chart (label composition).

**Next session should:**
- Verify claims 29, 31, 32 (results section claims requiring execution)

---

## Session 8 — 2026-04-22

**Purpose:** execution (claim 29 — Expert-only vs context-augmented lenient accuracy medians)

**Files/context inspected:**
- `fabscore_claude/progress.md` (sessions 1-7)
- `fabscore_claude/workspace/claim_23_command_output.txt` — reused from Session 5; contains per-strategy cross-model lenient accuracy medians

**Execution artifacts created/reused:**
- Reused `fabscore_claude/workspace/claim_23_command_output.txt` — contains the exact cross-model lenient accuracy medians needed for claim 29

**Key findings:**
- Per-strategy cross-model lenient accuracy medians (from Session 5):
  - S2 (public_health_expert): 64.22%
  - S3 (respiratory_doctor): 59.61%
  - S4 (detailed_public_health): 58.67%
  - S5 (detailed_respiratory): 52.89%
- Claim says S2/S3 are "around 65-70%" and S4/S5 are "around 55-60%"
- S2 at 64.22% is close to "around 65%", but S3 at 59.61% is NOT "around 65-70%"
- S3 and S4 are nearly identical (59.61% vs 58.67%), but the paper places them in different ranges
- S5 at 52.89% is below the claimed "around 55-60%" range
- The directional trend (S2,S3 > S4,S5) is correct, but the specific ranges significantly overstate S3

**Verdict summary (claim 29):** Result Fabrication — The directional claim is correct (S2,S3 > S4,S5 for lenient accuracy), but the claimed ranges are inaccurate. S3 (59.61%) is described as "around 65-70%" when it is actually ~60%, nearly the same as S4 (58.67%). The claim overstates both the expert-only medians (especially S3) and the context-augmented medians (S5 at 52.9% is below claimed "55-60%"). The specific ranges don't match the computed medians.

**Next session should:**
- Verify claims 31 and 32 (GPT-4o/Gemini/Llama/Mistral lenient accuracy differences across persona strategies)

---

## Session 10 — 2026-04-22

**Purpose:** execution (claim 32 — Llama-4 +1.1 pp, Mistral-Large +0.9 pp small positive lenient accuracy differences)

**Files/context inspected:**
- `fabscore_claude/progress.md` (sessions 1-9)
- `fabscore_claude/workspace/claim_31_command_output.txt` — reused from Session 9; contains S3-S2 lenient accuracy differences for all 10 models

**Execution artifacts reused:**
- `fabscore_claude/workspace/claim_31_command_output.txt` — already contains all 10 model S3-S2 differences including Llama-4 and Mistral-Large

**Key findings:**
- Claim 32 refers to S3-S2 (respiratory_doctor - public_health_expert) lenient accuracy differences
- Llama-4 (llama-4-maverick): S3(52.03%) - S2(50.94%) = **+1.09 pp** (paper: +1.1 pp ✓)
- Mistral-Large (mistral-large-latest): S3(54.06%) - S2(53.12%) = **+0.94 pp** (paper: +0.9 pp ✓)
- These are the 2 models out of 10 with positive differences (the paper says "small positives for Llama-4 and Mistral-Large")
- Values match to rounding (1.09≈1.1, 0.94≈0.9)

**Verdict summary (claim 32):** VERIFIED — The S3-S2 lenient accuracy differences for Llama-4 (+1.09 pp ≈ +1.1 pp) and Mistral-Large (+0.94 pp ≈ +0.9 pp) exactly match the paper's claimed values. Reused Session 9 artifacts.

**Next session should:**
- No remaining execution-required claims. All claims 21-23, 25-26, 29, 31-32 have been addressed.

---

## Session 9 — 2026-04-22

**Purpose:** execution (claim 31 — lenient accuracy differences mostly negative, GPT-4o -10.0 pp, Gemini -8.1 pp)

**Files/context inspected:**
- `fabscore_claude/progress.md` (sessions 1-8)
- `199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Logs/` (50 JSON log files)
- Identified that `judgment_classification` field is stored as a string (not int) in log files

**Execution artifacts created:**
- `fabscore_claude/workspace/verify_claim31.py` — script computing S3-S2 (respiratory_doctor - public_health_expert) lenient accuracy differences per model
- `fabscore_claude/workspace/claim_31_command_output.txt` — full command output

**Key findings:**
- The comparison is S3 (respiratory_doctor) minus S2 (public_health_expert) lenient accuracy
- GPT-4o (chatgpt-4o-latest): S3=43.28% - S2=53.28% = **-10.00 pp** (claim: -10.0 pp ✓)
- Gemini (gemini-2.5-pro): S3=72.97% - S2=81.09% = **-8.12 pp** (claim: -8.1 pp ✓)
- 8/10 models have negative S3-S2 differences (80%) → "mostly negative" ✓
- All per-model lenient accuracies match exactly the values noted in Session 2

**Verdict summary (claim 31):** VERIFIED — The S3-S2 (respiratory_doctor vs public_health_expert) lenient accuracy difference is -10.00 pp for GPT-4o and -8.12 pp for Gemini (matching the paper's claimed -10.0 pp and -8.1 pp exactly), and 8/10 models show negative differences confirming "mostly negative."

**Next session should:**
- Verify claim 32 (Llama +1.1 pp, Mistral +0.9 pp lenient accuracy increases)
