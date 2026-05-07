## Session 2026-04-13 - extraction

- Purpose: extraction
- Paper/context inspected: `334_Initiation_of_Programmed_C.pdf` (PDF main paper only; reviewed figure captions, Table 1, and the body text of Section 3 "Results" for numeric claims, with table-to-results deduplication).
- JSON files created or updated in this session: `fabscore_codex/fs_extracted.json`
- Other files created or updated in this session: `fabscore_codex/progress.md`
- Next session should do: verify the extracted entries against any downstream validation rules or aggregation pipeline expectations, but no additional paper content appears to require extraction beyond the main PDF.

## Session 2026-04-13 - analysis

- Purpose: analysis
- Files/context inspected:
  - `334_Initiation_of_Programmed_C.pdf` via local PDF text extraction
  - `334_Initiation_of_Programmed_C_Supplementary Material/README.md`
  - `334_Initiation_of_Programmed_C_Supplementary Material/run_analysis.py`
  - `334_Initiation_of_Programmed_C_Supplementary Material/docking_results.csv`
  - `334_Initiation_of_Programmed_C_Supplementary Material/pymol_commands.txt`
  - `334_Initiation_of_Programmed_C_Supplementary Material/apbs_template.in`
  - `334_Initiation_of_Programmed_C_Supplementary Material/hdock_submission_instructions.txt`
  - `334_Initiation_of_Programmed_C_Supplementary Material/requirements.txt`
  - `334_Initiation_of_Programmed_C_Supplementary Material/Dockerfile`
  - repository file listing to confirm absence of raw docking logs, PDB inputs, mutant structures, and figure data
- JSON files created or updated in this session:
  - `fabscore_codex/fs_analysis.json` (created)
- Concise classification summary:
  - `obvious_hallucination`: 4 claims
  - `result_fabrication`: claim 1 (WT docking score mismatch), claim 9 (8.41% hydrophobic gain unsupported), claim 10 (8.16% electrostatic gain contradicted by CSV)
  - `experiment_fabrication`: claim 8 (Figure 2 implementation path targets TRAIL electrostatics, not DR5)
  - `insufficient_evidence`: 6 claims (table values that only appear in a manually editable/example CSV without raw HDOCK outputs, plus Figure 1 lacking exact figure-generation provenance)
- Recommended next step for next session:
  - If a later stage permits broader evidence review, look for any omitted raw HDOCK exports, mutant PDBs, APBS outputs, or figure source files outside the current submission bundle; absent those, the current verdicts already rest on the strongest static evidence available in-repo.
