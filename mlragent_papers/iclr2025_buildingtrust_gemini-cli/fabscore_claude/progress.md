## Session Log

---

### Session 1
- **Purpose:** extraction
- **Paper inspected:** results/paper.md — "Dynamic Policy Enforcers: Adaptive Guardrails for Trustworthy LLM Applications"
- **JSON files created/updated:**
  - `fabscore_claude/fs_extracted.json` (created)
- **Summary:**
  - Extracted 9 table entries from Table 2 (performance: Accuracy and F1-Score BLOCK for 3 methods) and Table 3 (latency for 3 methods). Table 1 was skipped as it contains only training hyperparameters.
  - Extracted 3 figure entries (figures/performance_comparison.png, figures/latency_comparison.png, figures/training_loss.png).
  - Extracted 1 results_section claim: the "over 3.5 times faster" comparative latency result from Section 6.2, which is a derived metric not directly represented in Table 3.
  - All other numerical claims in Section 6 were already captured in Table 2 or Table 3 and excluded from results_section per deduplication rules.
- **Next session should:** Perform analysis/scoring of the extracted results against ground truth, or run the fabscore pipeline on `fs_extracted.json`.

---

### Session 2
- **Purpose:** analysis (static analysis of all 13 extracted claims)
- **Files inspected:**
  - `results/paper.md` — full paper text
  - `gemini/baseline_results.json` — baseline evaluation results (30 examples)
  - `gemini/dpe_results.json` — DPE evaluation results (6 examples, test split)
  - `gemini/training_history.json` — LoRA fine-tuning training/validation loss log
  - `gemini/dynosafe_benchmark.csv` — 30-row benchmark dataset
  - `gemini/02_run_baselines.py` — baseline evaluation code
  - `gemini/03_finetune_dpe.py` — DPE fine-tuning code
  - `gemini/04_evaluate_dpe.py` — DPE evaluation code
  - `gemini/05_visualize_and_analyze.py` — figure generation code
- **JSON files created/updated:**
  - `fabscore_claude/fs_analysis.json` (created)
- **Key Finding — Evaluation Protocol Mismatch:**
  - `02_run_baselines.py` evaluates both baselines on ALL 30 examples from the full CSV (no split).
  - `04_evaluate_dpe.py` evaluates the DPE on only 6 examples (20% test split via `train_test_split(test_size=0.2, seed=42)`).
  - The paper presents all three methods in a unified comparison table ("30 examples spanning 3 distinct policies") without disclosing this mismatch.
  - Additionally, the DPE model is degenerate: it predicts BLOCK for all 6 test examples. Its accuracy (0.6667) and F1 (0.8) are inflated by the coincidence that 4/6 test examples are ground-truth BLOCK.
  - The "over 3.5x faster" claim is also technically incorrect: 967.24/276.65 = 3.497 < 3.5.
- **Summary of Classifications:**
  - `static_verifiable` (7): Claims 1, 2, 3, 4, 7, 8, 12 — baseline metrics and training loss figure all exactly match stored JSON artifacts.
  - `obvious_hallucination` / `experiment_fabrication` (6): Claims 5, 6, 9, 10, 11, 13 — DPE metrics, DPE latency, performance/latency figures, and the "3.5x faster" claim all derive from the mismatched 6-example evaluation compared against 30-example baselines.
- **Next session should:** Run execution if desired to re-evaluate with consistent evaluation sets, or finalize scoring.
