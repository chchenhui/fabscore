# Progress Log

## Session 1 — 2026-04-13
- **Purpose**: extraction
- **Paper inspected**: 334_Initiation_of_Programmed_C.pdf (9 pages, PDF)
  - Title: "Initiation of Programmed Cell Death in Cancer Stem Cells: In Silico Mutagenesis for Optimized TRAIL–DR5 Binding with Perplexity AI"
- **JSON files created/updated**:
  - `fabscore_claude/fs_extracted.json` (created)
- **Summary of extraction**:
  - 1 table extracted (Table 1): 6 numerical entries covering Docking Score and Confidence (%) for Wild-Type, Electrostatic Mutant, and Hydrophobic Mutant TRAIL variants
  - 2 figures extracted (Figure 1 and Figure 2)
  - 2 results_section claims: the 8.41% binding affinity improvement for the best hydrophobic variant, and the up to 8.16% gain for electrostatic mutants (both not directly listed in the table)
- **Next session**: No further extraction needed unless re-scoring or additional analysis is requested. The paper is short (4 main pages + appendices) with a single results table and two illustrative figures.

## Session 2 — 2026-04-13
- **Purpose**: analysis (static code audit)
- **Files inspected**:
  - `334_Initiation_of_Programmed_C_Supplementary Material/docking_results.csv`
  - `334_Initiation_of_Programmed_C_Supplementary Material/run_analysis.py`
  - `334_Initiation_of_Programmed_C_Supplementary Material/README.md`
  - `334_Initiation_of_Programmed_C_Supplementary Material/pymol_commands.txt`
  - `334_Initiation_of_Programmed_C_Supplementary Material/apbs_template.in`
  - `334_Initiation_of_Programmed_C_Supplementary Material/hdock_submission_instructions.txt`
- **JSON files created/updated**:
  - `fabscore_claude/fs_analysis.json` (created)
- **Summary of classifications**:
  - Claim 1 (WT Docking Score -285.00): `obvious_hallucination` / `result_fabrication` — docking_results.csv shows -285.52; all other values match, so CSV appears to be the actual source data. -285.52 does not round to -285.00.
  - Claims 2–6 (other table values): `insufficient_evidence` — values match CSV but README explicitly calls it "an example file", and no actual HDOCK web-service result files are present; provenance chain is ambiguous.
  - Claims 7–8 (Figures 1 and 2): `no_code_files` — no PDB input files, no generated figure images, no figure-specific visualization scripts. pymol_commands.txt does only basic protein cleaning, not figure generation.
  - Claim 9 (8.41% hydrophobic improvement): `obvious_hallucination` / `result_fabrication` — computed from CSV: ≈0.23% improvement; no code in repo computes percentage improvements. 8.41% has no mathematical basis.
  - Claim 10 (8.16% electrostatic gains): `obvious_hallucination` / `result_fabrication` — electrostatic score (-281.00) is actually WORSE than WT (-285.52) since lower=better; there are no gains at all. No code computes 8.16%.
- **Recommended next step**: No execution needed. The analysis is complete. Claims 2–6 remain `insufficient_evidence` because the CSV is labeled "example" and actual HDOCK outputs are absent; running the HDOCK web service manually is out of scope for automated verification.
