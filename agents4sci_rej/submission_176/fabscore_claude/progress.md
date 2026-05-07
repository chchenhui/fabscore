# Progress Log

## Session: 2026-04-24 — Extraction

**Purpose:** Extract experimental results from paper.

**Paper inspected:** `176_UNMERGE_Verifiable_Model_C.pdf`
- Title: "UNMERGE: Verifiable Model Capability Attribution via Sparse Coding"
- Content: 15 pages including main paper (8 pages), appendix with Tables 4–6, and checklists.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section arrays.

**Summary of extraction:**
- **Tables:** 120 entries extracted from Tables 1–6 covering reconstruction error, precision, recall, perfect match rate, sparsity across algorithms (NNLS, OMP, Ridge, Elastic Net, Dot Product, Lasso) and model categories (Known, Mixed, Unknown) and merging methods (Task Arithmetic, Linear, TIES, DARE).
- **Figures:** None — the PDF contains no embedded figures or figure labels.
- **Results section:** 1 entry — a sparsity range claim (3.2 to 7.2) from the text body that does not match the table values and thus was not deduplicated.

**Next session should:**
- Proceed to analysis/scoring if needed. No further extraction work required.

## Session: 2026-04-24 — Analysis

**Purpose:** Statically classify 121 extracted numerical claims into verification buckets.

**Files inspected:**
- `176_UNMERGE_Verifiable_Model_C_Supplementary Material/results/decomposition_results_analysis.json` — primary verification artifact; 4,500+ lines of hierarchical statistical summaries (mean/std/min/max/median) for all 1,296 experimental runs across algorithms, categories, and merging methods.
- `176_UNMERGE_Verifiable_Model_C_Supplementary Material/PHASE4_DECOMPOSITON_TABLES.md` — pivot table summaries confirming analysis JSON values.
- `176_UNMERGE_Verifiable_Model_C_Supplementary Material/PHASE4.md` — comprehensive phase 4 report with inline analysis.

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with 121 claim classifications.

**Summary of classifications:**
- **120 claims → `static_verifiable`**: All table entries from Tables 1–6 (reconstruction error, sparsity, precision, recall, perfect match rate) were directly matched to corresponding entries in `decomposition_results_analysis.json`. All values agree within normal floating-point rounding (≤0.006 difference).
- **1 claim → `obvious_hallucination` (result_fabrication)**: Claim 121 (results_section) states "Sparsity varies significantly from 3.2 (Lasso) to 7.2 (Ridge) components on average." All stored data contradicts this: Lasso overall sparsity = 0.944, Ridge overall sparsity = 8.0; no category/merging-method subset produces 3.2 or 7.2 for these algorithms.

**Counts:** total=121, static_verifiable=120, obvious_hallucination=1, no_code_files=0, insufficient_evidence=0, execution_required=0, error=0.

**Next session should:**
- No further analysis work required. The fs_analysis.json is complete and ready for scoring.
