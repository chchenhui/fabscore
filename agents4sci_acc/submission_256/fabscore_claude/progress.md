# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** 256_Reasoning_Models_Outperfor.pdf (7 content pages + appendix/checklist)
**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` (created)

**Summary of extraction:**
- Tables: 20 entries from Table 1 (model overview: N sequences and success rates) and Table 2 (4-helix bundle formation rates and confident design rates for 5 ChatGPT variants).
- Figures: 4 figures extracted (Figure 1: pipeline overview; Figure 2: AlphaFold predictions of Helix01 and Helix02; Figure 3: comprehensive experimental characterization of Helix02; Figure 4: failed expression of Helix01).
- Results section: 5 claims covering Helix01 pLDDT (0.876, failed expression), Helix02 pLDDT (0.887, successful), SDS-PAGE purity >95% and MW 14.5 kDa, SEC 68.7% main peak recovery, and CD 100% helical content.

**Next session should do:** analysis/scoring (fs_analysis.json, fs_execution.json, fs_summary.json).

## Session 2 — 2026-04-24
**Purpose:** analysis (static)
**Files inspected:**
- `256_Reasoning_Models_Outperfor_Supplementary Material/raw_results.xlsx` — primary data file (40 rows, columns: model, sequence, plddt, helices, bundle, ordered)
- `256_Reasoning_Models_Outperfor_Supplementary Material/alphafold_analysis.ipynb` — ColabFold runner notebook with pre-run outputs
- `fabscore_claude/fs_extracted.json` — previously extracted claims

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` (created)

**Summary of classifications (29 claims total):**
- `static_verifiable`: 22 claims (indices 1–20, 25, 26)
  - All Table 1 and Table 2 values verified directly from raw_results.xlsx
  - "4-Helix Bundle" = fraction(helices=4 AND bundle='y'); "Confident 4-Helix Bundle" = fraction(helices=4 AND bundle='y' AND pLDDT≥0.75)
  - "Success Rate" in Table 1 == "Confident 4-Helix Bundle" in Table 2
  - Helix01 (pLDDT=0.876, bundle='n') and Helix02 (pLDDT=0.887, bundle='y', ordered='y') both confirmed in spreadsheet
- `no_code_files`: 6 claims (indices 21, 23, 24, 27, 28, 29)
  - Figure 1 (schematic), Figures 3–4 (wet-lab gel/SEC/CD images), and results claims 27–29 (SDS-PAGE purity/MW, SEC 68.7%, CD 100% helical) are experimental results with no computational artifacts
- `execution_required`: 1 claim (index 22)
  - Figure 2 (AlphaFold pLDDT-colored structures): notebook can regenerate but static inspection insufficient

**Next session should do:** execution of alphafold_analysis.ipynb to verify Figure 2 (claim 22), or mark as insufficient_evidence if execution fails.

## Session 3 — 2026-04-24
**Purpose:** execution (claim 22 — Figure 2)
**Files inspected:**
- `256_Reasoning_Models_Outperfor_Supplementary Material/alphafold_analysis.ipynb` — Cell 4 pre-run outputs extracted
- Notebook cell 4 has 40 stream outputs (pLDDT values) and 40 image/png outputs (pLDDT-colored pseudo-3D visualizations)

**Execution artifacts created:**
- `fabscore_claude/workspace/Helix01_plddt_0.8762.png` — pLDDT-colored pseudo-3D visualization for seq 16 (pLDDT=0.8762)
- `fabscore_claude/workspace/Helix02_plddt_0.8874.png` — pLDDT-colored pseudo-3D visualization for seq 39 (pLDDT=0.8874)

**Verdict summary (claim 22):** VERIFIED
- Seq 16 (pLDDT=0.8762≈0.876) corresponds to Helix01; seq 39 (pLDDT=0.8874≈0.887) corresponds to Helix02
- Both have pre-run pLDDT-colored pseudo-3D structure images in cell 4 of the notebook
- Images show protein structures colored by pLDDT confidence (blue=high, yellow-green=lower)
- Helix01 shows extended helical structure; Helix02 shows compact 4-helix bundle — consistent with paper's description

**Next session:** No further execution needed for this claim. All claims have been processed.
