# Progress Log

## Session 1 — extraction
- **Purpose**: Extract experimental results from the paper.
- **Paper inspected**: `344_Stylistic_Contrastive_Lear.pdf` (Stylistic Contrastive Learning for Human-Like AI Text Generation)
- **Files created/updated**:
  - `fabscore_claude/fs_extracted.json` (created): Contains tables, figures, and results_section entries.
- **Summary of extraction**:
  - **Tables**: 54 entries extracted from Tables 1–4 (semantic fidelity/factuality, ArgEssay main metrics, ChatDialog out-of-domain detectability, NewsNYT diversity/discourse metrics).
  - **Figures**: None found in the paper (no figures present).
  - **Results section**: 4 text-body claims extracted (human Turing-style test expert/layperson identification rates, ablation results for contrastive objective removal and idiom supervision removal).
- **Next session**: Run analysis or scoring on the extracted results (`fs_analysis.json`, `fs_execution.json`, `fs_summary.json`).

## Session 2 — analysis (2026-04-02)
- **Purpose**: Static analysis of all 58 claims against repository code and artifacts.
- **Files inspected**:
  - `344_Stylistic_Contrastive_Lear.pdf` (paper, all pages)
  - `Supplementary/outputs/results.csv` (12-row CSV with experimental results)
  - `Supplementary/code/evaluate.py` (282-line evaluation script)
  - `Supplementary/outputs/training_example.log` (synthetic demo log)
  - `Supplementary/docs/reproducibility_statement.txt`
  - `fabscore_claude/progress.md`
- **Files created/updated**:
  - `fabscore_claude/fs_analysis.json` (created): Full classification of all 58 claims.
- **Key findings**:
  1. **evaluate.py RoBERTa hardcoded**: `evaluate_roberta_detector` unconditionally returns `{'roberta_accuracy': 0.65}` — bypasses real evaluation. All 8 RoBERTa detection claims (11,17,23,29,35,41,43,45) → `experiment_fabrication`.
  2. **evaluate.py compression_diversity hardcoded**: `total_compressed = total_original * 0.8` always returns 0.80 — cannot produce the claimed 0.71, 0.76 values for Table 4.
  3. **Table 4 model name conflict**: results.csv labels NewsNYT rows as "GPT-5 baseline" and "GPT-5 FT", but the paper claims these rows are "Open-LLM baseline" (unfine-tuned open-weight) and "Open-LLM-FT (human)". All 9 Table 4 claims (46–54) → `experiment_fabrication`.
  4. **Table 4 metric mismatch**: CSV has `distinct2_percent` but Table 4 reports `Dist-3 (%)`. Values also differ (46.7/48.9/52.7 in CSV vs 40.2/45.6/50.9 in paper).
  5. **No BERTScore/QAFactEval/NLI code**: evaluate.py has none of these metrics. No output artifact contains Table 1 values. Claims 1–9 → `no_code_files`.
  6. **No human evaluation code**: results.csv has human_likeness_score column but no code generates it. Claims 15,21,27,33,39 → `insufficient_evidence`.
  7. **No ablation or human Turing-test artifacts**: Claims 55–58 → `no_code_files`. training_example.log uses synthetic demo data explicitly.
  8. **results.csv matches Table 2 and Table 3 stylometric/diversity/idiom/discourse values**: With non-hardcoded code in evaluate.py for these metrics, claims 10,12–14,16,18–20,22,24–26,28,30–32,34,36–38,40,42,44 → `static_verifiable`.
- **Summary**: 23 static_verifiable, 17 experiment_fabrication, 13 no_code_files, 5 insufficient_evidence, 0 execution_required.
- **Next step**: No further sessions needed for analysis. Execution would require real datasets and models (GPT-5 API, actual Open-LLM), which are not available in the repository.
