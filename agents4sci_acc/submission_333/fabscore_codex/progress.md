# Progress Log for Submission 333

---

## Session: extraction (2026-04-11)

**Purpose:** Extract experimental results from the paper.

**Paper/context inspected:** `333_Multi_target_Parallel_Drug.pdf` ("Multi-target Parallel Drug Discovery with Multi-agent Orchestration"). Inspected the PDF text layer for the Results and Discussion section, figure captions, and Table 1. Also verified that page 7 contains an embedded table image referenced in the body text as `Table 4.1`.

**JSON files created or updated in this session:**
- Created: `fabscore_codex/fs_extracted.json`

**Next session should do:** Validate the extraction against any downstream scoring checks, and if stricter PDF-only table recovery is required later, OCR the embedded page-7 table image to independently confirm the `Table 4.1` row text.

---

## Session: analysis (2026-04-11)

**Purpose:** Static analysis of all 63 extracted claims against the repository and saved artifacts.

**Files/context inspected:**
- `333_Multi_target_Parallel_Drug.pdf` (indirectly via prior extraction and repository cross-checks)
- `fabscore_codex/fs_extracted.json`
- `agentic-drug-discovery-main/README.md`
- `agentic-drug-discovery-main/notebooks/1_ml_training_stanfordAIAgentConf.ipynb`
- `agentic-drug-discovery-main/notebooks/2_molecule_evaluation_stanfordAIAgentConf.ipynb`
- `agentic-drug-discovery-main/data/SGLT2_processed.csv`
- `agentic-drug-discovery-main/data/CGAS_processed.csv`
- `agentic-drug-discovery-main/data/SEH_processed.csv`
- `agentic-drug-discovery-main/data/HDAC_processed.csv`
- `agentic-drug-discovery-main/data/DYRK1A_processed.csv`
- `agentic-drug-discovery-main/data/chembl_cache_SGLT2.csv`
- `agentic-drug-discovery-main/data/chembl_cache_CGAS.csv`
- `agentic-drug-discovery-main/data/chembl_cache_SEH.csv`
- `agentic-drug-discovery-main/data/chembl_cache_HDAC.csv`
- `agentic-drug-discovery-main/data/chembl_cache_DYRK1A.csv`
- `agentic-drug-discovery-main/results/inhibition_classification/SGLT2_inhibition_classification_metrics.csv`
- `agentic-drug-discovery-main/results/inhibition_classification/CGAS_inhibition_classification_metrics.csv`
- `agentic-drug-discovery-main/results/inhibition_classification/SEH_inhibition_classification_metrics.csv`
- `agentic-drug-discovery-main/results/inhibition_classification/HDAC_inhibition_classification_metrics.csv`
- `agentic-drug-discovery-main/results/inhibition_classification/DYRK1A_inhibition_classification_metrics.csv`
- `agentic-drug-discovery-main/results/admet_ml/Solubility_AqSolDB_regression_metrics.csv`
- `agentic-drug-discovery-main/results/admet_ml/HalfLife_Human_regression_metrics.csv`
- `agentic-drug-discovery-main/results/admet_ml/hERG_inhibition_classification_metrics.csv`
- `agentic-drug-discovery-main/results/target_inhibitors/seeds.csv`
- `agentic-drug-discovery-main/results/target_inhibitors/summary_pred.csv`
- `agentic-drug-discovery-main/results/target_inhibitors/top_targets.csv`
- `agentic-drug-discovery-main/results/target_inhibitors/target_eval.csv`
- `agentic-drug-discovery-main/figures/final/grachical_abstract.png`
- `agentic-drug-discovery-main/figures/final/result_mols.png`

**JSON files created or updated in this session:**
- Created: `fabscore_codex/fs_analysis.json`

**Classification summary:**
- `static_verifiable`: 59 claims
- `obvious_hallucination`: 1 claim
  - `result_fabrication`: 1 claim (`index` 62 misreports half-life regression output as AUROC 0.46)
- `insufficient_evidence`: 1 claim (`index` 56, conceptual Figure 1 provenance)
- `execution_required`: 2 claims (`index` 58 and `index` 63, Figure 3 / generated-molecule bioactivity scores)
- `no_code_files`: 0 claims
- `error`: 0 claims

**Key findings:**
- Table 1 manual-threshold rows are directly supported by `threshold_dict` plus the saved processed datasets.
- Table 1 AI-threshold rows are also derivable from repository data: applying the notebook's default `pchembl_threshold_active=6.0` to the processed target datasets reproduces all claimed AI-agent counts exactly.
- All Table 4.1 per-molecule ADMET values match `results/target_inhibitors/summary_pred.csv`.
- Figure 2 and results-section performance claims are backed by saved metric CSVs.
- The half-life body-text claim uses the wrong metric label: the repository stores half-life as a regression task with median Spearman about `0.463`, not AUROC.
- Figure 3 lacks saved per-molecule target-bioactivity scores for the final 15 reported molecules, so exact Figure-3B verification still needs execution.

**Recommended next step for the next session:** Execute the repository-native Figure 3 regeneration path in `agentic-drug-discovery-main/notebooks/2_molecule_evaluation_stanfordAIAgentConf.ipynb` and capture the final per-molecule target-bioactivity probabilities for the selected 15 molecules; this is the remaining blocker for claims 58 and 63.

---

## Session: execution (2026-04-11)

**Purpose:** Verify claim 58 (Figure 3 seed selection, generated inhibitors, and bioactivity parenthetical scores) using existing repository artifacts and the minimal claim-relevant execution path.

**Files/context inspected:**
- `333_Multi_target_Parallel_Drug.pdf`
- `fabscore_codex/progress.md`
- `agentic-drug-discovery-main/notebooks/2_molecule_evaluation_stanfordAIAgentConf.ipynb`
- `agentic-drug-discovery-main/results/target_inhibitors/seeds.csv`
- `agentic-drug-discovery-main/results/target_inhibitors/summary_pred.csv`
- `agentic-drug-discovery-main/figures/final/result_mols.png`
- `agentic-drug-discovery-main/models/*`
- `agentic-drug-discovery-main/results/inhibition_classification/*`
- `agentic-drug-discovery-main/results/admet_ml/*`

**Execution artifacts created or updated in this session:**
- Created: `fabscore_codex/workspace/claim_58_command_output.txt`
- Created: `fabscore_codex/workspace/claim_58_evidence.json`

**Concise verdict summary for this claim:**
- `seeds.csv` supports Figure 3A: 15 seeds total, exactly three seeds each for `CGAS`, `DYRK1A`, `HDAC`, `SEH`, and `SGLT2`.
- `summary_pred.csv` plus notebook cell `2768-2797` support the existence of 15 final picked molecules and their ADMET predictions.
- Notebook cell `2587-2636` shows the intended target-bioactivity filtering logic (`CANDIDATE_TH = 0.90`), but the saved executable loop is currently restricted to `for TARGET in ["SGLT2"]` rather than all five targets.
- Notebook cell `2665-2668` preserves explicit picked `(SMILES, probability)` pairs only for `SGLT2`; the final image `figures/final/result_mols.png` visibly shows parenthetical probabilities for all five targets, but those values are not saved in a raw repository table/log for the non-SGLT2 targets.
- A direct local rerun of the notebook inference path is blocked by missing environment dependencies (`rdkit`, `mordred`, `xgboost`), so the exact Figure 3B probabilities cannot be freshly recomputed in this environment.

**Next session should do:**
- Re-run the Figure 3 inference path in an environment with `rdkit`, `mordred`, and `xgboost` available, ideally by executing the notebook logic after changing the loop from `["SGLT2"]` to `TARGETS`, and save the resulting per-target picked `(SMILES, probability)` table for all five targets.

---

## Session: execution (2026-04-11)

**Purpose:** Verify claim 63 (results text about Figure 3B consistently producing novel molecules with high predicted bioactivity scores, often exceeding 0.90).

**Files/context inspected:**
- `333_Multi_target_Parallel_Drug.pdf`
- `fabscore_codex/progress.md`
- `fabscore_codex/execution_log.json`
- `fabscore_codex/workspace/claim_58_evidence.json`
- `fabscore_codex/workspace/claim_63_command_output.txt`
- `agentic-drug-discovery-main/notebooks/2_molecule_evaluation_stanfordAIAgentConf.ipynb`
- `agentic-drug-discovery-main/results/target_inhibitors/summary_pred.csv`
- `agentic-drug-discovery-main/figures/final/result_mols.png`

**Execution artifacts created or updated in this session:**
- Created: `fabscore_codex/workspace/claim_63_command_output.txt`
- Created: `fabscore_codex/workspace/claim_63_evidence.json`

**Concise verdict summary for this claim:**
- Reused `fabscore_codex/workspace/claim_58_evidence.json` because it already extracted the visible Figure 3B parenthetical scores from `result_mols.png` and documented the same notebook bottleneck relevant to this claim.
- Confirmed from the paper text that claim 63 refers specifically to Figure 3B and says the predicted bioactivity scores often exceeded `0.90`.
- Confirmed from notebook lines `2587-2636` that the intended generation filter keeps only molecules with `CANDIDATE_TH = 0.90`, but the saved executable loop is currently restricted to `for TARGET in ["SGLT2"]`.
- Confirmed from notebook lines `2665-2689` that explicit picked `(SMILES, probability)` tuples are preserved only for `SGLT2`, with scores `0.9054189`, `0.9005101`, and `0.90543234`.
- Confirmed from `summary_pred.csv` and notebook lines `2767-2817` that the 15 final molecules are saved with ADMET outputs only; raw target-bioactivity probabilities for the final non-`SGLT2` molecules are not stored in a table/log.
- Fresh execution of the claim-relevant notebook inference path is blocked locally because `rdkit`, `mordred`, and `xgboost` are not installed, so the all-target Figure 3B probabilities cannot be recomputed in this environment.

**Next session should do:**
- Re-run the Figure 3B inference path in an environment with `rdkit`, `mordred`, and `xgboost` installed, and save the per-target picked `(SMILES, probability)` tuples for all five targets so claim 63 can be resolved without relying on the pre-existing figure image alone.
