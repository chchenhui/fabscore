# Progress Log

## Session: 2026-04-23

**Purpose:** extraction

**Paper inspected:** `255_Neural_Reaction_Diffusion_.pdf` — "Neural Reaction-Diffusion Operators for Spatially Heterogeneous Tumor Modeling"

**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries

**Summary of extraction:**
- **Tables:** 24 entries from Table 1 (quantitative results across 5 heterogeneity scenarios: MSE, Max Error, Time, Speedup for each scenario plus averages)
- **Figures:** 4 figures extracted (Figure 1: NRDO architecture; Figure 2: heterogeneity patterns; Figure 3: accuracy comparison; Figure 4: comprehensive performance analysis)
- **Results section:** 2 claims extracted from body text not duplicated in tables:
  1. 2-3 orders of magnitude speedup (qualitative range not directly in Table 1's specific speedup numbers)
  2. Conservation error below 10⁻⁴ across all scenarios (physics compliance metric, not in Table 1)

**Next session:** No further extraction needed. If a scoring/review task is required, compare extracted results against claim accuracy and verify consistency between table values and text claims.

---

## Session 2: Analysis (2026-04-23)

### Session Purpose
Static analysis of repository to classify all 30 extracted claims from the paper.

### Files/Context Inspected
- **Paper**: `255_Neural_Reaction_Diffusion_.pdf` — Full 13-page paper about NRDOs for solving heterogeneous PDEs in tumor modeling.
- **Repository structure**: Full exploration of `/home/chenhui/fabscore/agents4sci_rej/submission_255/`
- **SupplementaryMaterials/**: Contains code for "Hierarchical Meta-Learning for Cancer Pathway Signatures" (cancer genomics, TCGA data) — **completely unrelated to the paper topic**
- **fabscore_claude/**: Contains prior extraction artifacts

### Key Finding: Critical Mismatch

1. **Repository ≠ Paper**: The repository's SupplementaryMaterials contains code for a different project (cancer pathway meta-learning with TCGA data), not for NRDO/PDE solving. There is NO NRDO implementation code anywhere in the repository.

2. **Internal Paper Contradiction — Speedup**: Figure 4 embedded in the paper PDF shows a "Computational Speedup vs FD" bar chart with values **0.004×, 0.003×, 0.004×, 0.004×, 0.005×** — meaning NRDOs are **200-333× SLOWER** than finite difference, not 189-285× faster as Table 1 claims. The Figure 4 Performance Summary table explicitly states "Avg Speedup: 0.004×" vs Table 1's "223×."

3. **Internal Paper Contradiction — Max Error**: Figure 4 shows Avg Max Error = 0.170, while Table 1 claims 0.0153. Vascular Network max error in Figure 4 = 0.714 vs Table 1's 0.0134 (53× different).

4. **Internal Paper Contradiction — MSE per scenario**: Figure 4 bar chart shows MSE values (3.3e-06, 3.4e-06, 4.2e-05, 2.0e-04, 2.8e-05) inconsistent with Table 1's values (3.21e-05, 5.47e-05, 6.82e-05, 4.95e-05, 6.85e-05), though averages both equal 5.46e-05 (possibly cherry-picked average).

5. **Internal Paper Contradiction — Figure 3**: Claim 27 says "errors typically below 2%" but Figure 3 in the paper shows max errors of 0.024–0.040 (2.4–4%) at multiple time points.

### JSON Files Created/Updated
- Created: `fabscore_claude/fs_analysis.json`

### Classification Summary

| Category | Count | Indices |
|---|---|---|
| `no_code_files` | 4 | 25, 26, 28, 30 |
| `obvious_hallucination` (result_fabrication) | 26 | 1–24, 27, 29 |
| `static_verifiable` | 0 | — |
| `insufficient_evidence` | 0 | — |
| `execution_required` | 0 | — |
| `error` | 0 | — |
| **Total** | **30** | |

### Reasoning

- **Claims 1-24 (Table 1)**: `result_fabrication` — Figure 4 in the paper PDF directly contradicts Table 1. Speedup of 0.003-0.005× (Figure 4) vs 189-285× (Table 1) is the most dramatic conflict.
- **Claim 25 (Figure 1)**: `no_code_files` — Architecture diagram in PDF, no NRDO code or data files.
- **Claim 26 (Figure 2)**: `no_code_files` — Diffusion pattern figure in PDF, no NRDO code.
- **Claim 27 (Figure 3)**: `result_fabrication` — Figure 3 shows errors up to 4% (0.040), contradicting "typically below 2%."
- **Claim 28 (Figure 4)**: `no_code_files` — Figure present in PDF but no underlying data files or NRDO code.
- **Claim 29 (speedup 2-3 OOM)**: `result_fabrication` — Figure 4 shows NRDOs are 200-333× SLOWER than FD.
- **Claim 30 (conservation error)**: `no_code_files` — No NRDO code or artifacts.

### Recommended Next Step
Analysis is complete. No execution needed. All claims classified based on static analysis of the paper PDF (Figure 4 vs Table 1 conflict) and repository inspection (no NRDO code present).
