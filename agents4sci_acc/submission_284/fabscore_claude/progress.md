# Progress Log

## Session 1 — 2026-04-03
**Purpose:** extraction

**Paper inspected:** 284_BadScientist_Can_a_Researc.pdf (PDF, 17 pages)

**Summary:** Extracted experimental results from the BadScientist paper, which evaluates fabricated AI-generated papers against LLM reviewer systems. The paper contains 4 tables and 1 figure with results.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (102 entries from Tables 1–4), figures (1 entry for Figure 1), and results_section (empty — all numerical claims in body text duplicate table values exactly).

**Key findings:**
- Table 1: Acceptance rates and ICR by fabrication strategy (S1–S5, All) under two thresholds.
- Table 2: Concern-acceptance conflict percentages per model and strategy.
- Table 3: Baseline vs. ReD mitigation comparison (ACPT and ICR metrics).
- Table 4: Detector evaluation (TPR, FPR, Acc, F1) for Random Guess, ReD, DetOnly across o3, o4-mini, GPT-4.1.
- Figure 1: Score distribution histograms across six setups for three models.
- All numerical claims in the body text of Sections 4.2 and 5.2 were exact restatements of table values; results_section is intentionally empty.

**Next session should:** run analysis/scoring (fs_analysis.json) based on the extracted results.

---

## Session 2 — 2026-04-03
**Purpose:** analysis

**Files inspected:**
- `BadScientist/experiment/cs/results/README.md` — reveals results directory is empty; all experiment results are at external URL `https://anonymous.4open.science/r/BadScientist`
- `BadScientist/run_experiments.py` — paper/review generation runner; no metric aggregation code
- `BadScientist/perform_review.py` — individual paper review logic (Accept/Reject per paper); no ACPT/ICR/TPR/FPR/F1 aggregation
- `BadScientist/core.py` (via earlier session) — paper generation pipeline only
- `BadScientist/local_review_config.py` — API keys are empty strings; experiments cannot be run
- `BadScientist/experiment/cs/seed_ideas.json`, `strategy.json` — config only

**Key findings:**
- The local results directory (`experiment/cs/results/`) contains only `.DS_Store` and a README.md pointing to an external anonymous repository for all experiment results.
- No metric computation code for ACPT (acceptance rate), ICR (inter-model consistency rate), concern-acceptance conflict, TPR, FPR, Accuracy, or F1 exists anywhere in the local codebase.
- No pre-computed result artifacts (CSV, JSON, NPY, log files) exist in the local repository.
- API keys are empty so experiments cannot be reproduced locally.
- The local code only implements paper generation (core.py) and individual paper review (perform_review.py), not the metric aggregation needed to reproduce Table 1–4 values.

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with all 103 claims classified as `no_code_files`

**Classification summary:**
- no_code_files: 103 (all claims)
- All other buckets: 0

**Reasoning:** All 103 claims (Tables 1–4, Figure 1) require metric computation code and/or pre-computed result artifacts that are absent from the local repository. The README.md explicitly states results are in an external URL. No suitable repo-native code path exists for computing ACPT, ICR, TPR, FPR, F1, or concern-acceptance conflict values.

**Next session should:** No further analysis needed given all results are in external URL. If external results become available locally, revisit classification of Table 1–4 claims.
