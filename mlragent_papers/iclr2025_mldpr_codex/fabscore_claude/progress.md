## Session: 2026-04-24 — extraction

**Purpose:** Extract experimental results from the paper.

**Paper inspected:** results/paper.md
- Title: "Adaptive Deprecation Scoring and Notification for Machine Learning Data Repositories"
- A proof-of-concept on two OpenML datasets (iris, mnist_784), computing Deprecation Scores using citation-age and reproducibility signals.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created; contains tables (2 entries), figures (1 entry), results_section (1 entry).

**Summary of extraction:**
- Tables: Table 1 with Deprecation Scores for iris (D=1.00) and mnist_784 (D=0.96).
- Figures: One bar chart figure (deprecation_scores.png).
- Results section: One unique numerical claim not in tables — both datasets passed reproducibility checks (S_rep=0).

**Next session should:** Run analysis/scoring (fs_analysis.json) on the extracted results.

---

## Session: 2026-04-24 — analysis

**Purpose:** Static analysis of all 4 extracted claims against repository code and artifacts.

**Files inspected:**
- `results/paper.md` — paper text, Section 4 states equal weights w_cite=w_rep=0.5
- `codex/run_experiment.py` — implementation code
- `codex/results.csv` — execution output (D values, signal values)
- `results/log.txt` — execution log confirming S_rep=0.0 for both datasets
- `results/results.md` — summary of results
- `results/deprecation_scores.png` — figure artifact

**Key finding:**
The paper (Section 4) states "equal weights w_cite=w_rep=0.5" but `codex/run_experiment.py` lines 102-103 implement:
```python
weights = {'cite': 1.0}
D = weights['cite'] * S_cite
```
Only S_cite with weight=1.0 is used. With equal weights and S_rep=0: D_iris=0.5 (not 1.00), D_mnist≈0.48 (not 0.96). The reported values (1.00 and 0.96) only arise from the code's actual cite-only weight of 1.0.

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with all 4 claim classifications.

**Summary of classifications:**
- Claim 1 (iris D=1.00): `obvious_hallucination` / `experiment_fabrication` — paper states equal weights 0.5/0.5, code uses cite weight 1.0; equal weights would give D=0.5
- Claim 2 (mnist_784 D=0.96): `obvious_hallucination` / `experiment_fabrication` — same issue; equal weights would give D≈0.48
- Claim 3 (figure bar chart): `obvious_hallucination` / `experiment_fabrication` — figure generated from wrong weighting scheme
- Claim 4 (S_rep=0): `static_verifiable` — directly confirmed by codex/results.csv and log.txt

**Next session should:** No further action needed; analysis complete. If execution is attempted, note that running the code as-is will reproduce the same D values (1.00, ~0.96) but with the wrong weights vs. what the paper describes.
