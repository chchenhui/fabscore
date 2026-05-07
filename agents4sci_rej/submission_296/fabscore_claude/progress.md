## Session: 2026-04-23 — extraction

**Purpose:** Extract experimental results from paper

**Paper inspected:** 296_Intelligent_Document_Proce.pdf
- "Intelligent Document Processing for Graduate Admissions: An End-to-End Pipeline with Calibrated Abstention"

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (13 entries from Table 1 and Table 2), figures (3 entries: Figure 1, 2, 3), and results_section (1 entry: 70% time savings)

**Summary of findings:**
- Table 1 (main results): GPA MAE=0.831, Decision Accuracy=12.8%, ECE=0.691, Processing Time=0.0004s, Throughput=10.2M apps/hour
- Table 2 (baseline comparison): Random (33.3% acc, ECE=0.67), GPA-Only (100% acc, MAE=0.0, ECE=0.20), Proposed (12.8% acc, MAE=0.831, ECE=0.691)
- Figures: Baseline Comparison, Processing Time Analysis, GPA Error Distribution
- Only one unique text claim not duplicated in tables: 70% time savings vs. 20-minute manual review

**Next session:** No further extraction needed. Could proceed to analysis/scoring.

---

## Session: 2026-04-23 — analysis

**Purpose:** Static analysis of all 17 claims against repository code and artifacts.

**Files inspected:**
- `296_Intelligent_Document_Proce.pdf` (paper)
- `paper/main_paper.md` (markdown paper with Table 2 baseline values)
- `results/metrics.json` (main results, timestamp 20250912_180002)
- `results/results_20250912_175803/metrics.json` (first run results)
- `results/results_20250912_180002/metrics.json` (second run results)
- `code/run_experiments.py` (main experimental runner, 428 lines)
- `code/evaluate.py` (evaluation metrics computation, 129 lines)
- `code/plotting.py` (figure generation, 256 lines)
- `results/figures/` and timestamped run figure directories

**JSON files created:**
- `fabscore_claude/fs_analysis.json` — full classification of 17 claims

**Summary of classifications:**
- **static_verifiable (15):** Claims 1-5 (Table 1 main metrics), 8-13 (Table 2 GPA-only and Proposed), 14-17 (figures and time savings). All main pipeline values (GPA MAE=0.831, Decision Acc=12.8%, ECE=0.691, Processing Time≈0.0004s, Throughput≈10.2M) confirmed in both stored metrics.json files. GPA-only baseline values (100% acc, 0.0 MAE, 0.20 ECE) confirmed. Figure artifacts exist with matching underlying data in metrics.json.
- **obvious_hallucination / result_fabrication (2):** Claims 6 and 7 (Table 2, Random Assignment). Paper claims Decision Acc=33.3% and ECE=0.67, but both stored metrics files (two independent runs) consistently show Decision Acc=33.8% and ECE=0.413. The ECE discrepancy is particularly egregious (63% relative error: 0.67 vs 0.413).

**Key finding:** The critical discrepancy is in the Random Assignment baseline row of Table 2. The ECE of 0.67 reported in the paper conflicts with the actual computed value of 0.413 stored in the repository. Decision Accuracy of 33.3% also conflicts with the actual 33.8% (both runs use np.random.seed(42) and give identical results).

**Next session:** No further action required. Analysis complete.
