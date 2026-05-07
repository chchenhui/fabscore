# Progress Log for Submission 333

---

## Session: extraction (2026-04-02)

**Purpose:** Extract experimental results from the paper.

**Paper inspected:** `333_Multi_target_Parallel_Drug.pdf` — "Multi-target Parallel Drug Discovery with Multi-agent Orchestration"

**Paper summary:** Multi-agent framework for early-stage drug discovery applied to Alzheimer's Disease. Identifies 5 protein targets (SGLT2, CGAS, SEH, HDAC, DYRK1A), builds XGBoost classifiers for bioactivity and ADME/Tox properties, and uses NVIDIA MolMIM for de novo molecule generation with a sequential filtering cascade.

**JSON files created/updated:**
- Created: `fabscore_claude/fs_extracted.json`

**Extraction summary:**
- **Tables:** 2 tables extracted.
  - Table 1: Dataset composition (positive/negative counts and pIC50 thresholds for 5 targets under AI Agent vs. Manual thresholding).
  - Table 2: ML predictions (predicted solubility, half-life, hERG inhibition) for 3 compounds per target (15 rows total).
- **Figures:** 3 figures extracted (Figure 1: framework overview; Figure 2: model performance metrics; Figure 3: generated molecules with bioactivity scores).
- **Results section claims:** 5 numerical claims extracted from body text:
  1. Median AUPRC 0.80–0.89 for SGLT2, SEH, HDAC, DYRK1A.
  2. CGAS AUPRC only 0.60.
  3. Water Solubility Spearman=0.78, hERG AUPRC=0.79.
  4. Human microsomal half-life AUC-ROC=0.46 (failed model).
  5. Generated molecules with bioactivity scores often exceeding 0.90.

**Next session should:**
- Run analysis or scoring steps if required by the pipeline.
- Verify Table 2 compound-level values (PDF rendering of SMILES-heavy table made some rows ambiguous; values used are best estimates from the readable PDF text).

---

## Session: analysis (2026-04-02)

**Purpose:** Static analysis of all 63 claims against repository artifacts.

**Files inspected:**
- `agentic-drug-discovery-main/data/SGLT2_processed.csv`, `CGAS_processed.csv`, `SEH_processed.csv`, `HDAC_processed.csv`, `DYRK1A_processed.csv` — counted pos/neg label distributions
- `agentic-drug-discovery-main/notebooks/1_ml_training_stanfordAIAgentConf.ipynb` — confirmed `threshold_dict` uses fixed pIC50 thresholds (manual thresholding)
- `agentic-drug-discovery-main/notebooks/0_target_mining_stanfordAIAgentConf.ipynb` — uses Gemini for target mining from PubMed, no compound-level AI thresholding
- `agentic-drug-discovery-main/notebooks/2_molecule_evaluation_stanfordAIAgentConf.ipynb` — uses CANDIDATE_TH=0.90 filter; bioactivity scores not saved to file
- `agentic-drug-discovery-main/results/target_inhibitors/summary_pred.csv` — 15 compounds with ADMET predictions
- All 8 metric CSV files in `results/inhibition_classification/` and `results/admet_ml/`

**JSON files created/updated:**
- Created: `fabscore_claude/fs_analysis.json`

**Key findings:**

1. **AI Agent Thresholding (claims 1, 5, 7, 9) — `data_fabrication`:**
   - The processed CSV files use MANUAL thresholding (y=1 if pchembl ≥ threshold_dict value)
   - SGLT2 actual: 638/799 (not claimed 1249/188); SEH: 1032/1384 (not 1989/427); HDAC: 696/960 (not 926/730); DYRK1A: 1211/1553 (not 2061/703)
   - No AI Agent compound-level thresholding is implemented anywhere in the codebase
   - CGAS (claim 3) is fine — 100/146 matches both AI and manual at threshold 6.0

2. **CGAS Compound 3 Half-Life (claim 27) — `result_fabrication`:**
   - Paper says 0.74; `summary_pred.csv` shows 0.44 for this compound

3. **Table 1 Manual Thresholding (claims 2, 4, 6, 8, 10) — `static_verifiable`:**
   - All confirmed by actual data file counts

4. **Table 2 ADMET predictions (claims 11-26, 28-55) — `static_verifiable`:**
   - All 44 remaining claims confirmed directly from `summary_pred.csv`

5. **Results section metrics (claims 59-62) — `static_verifiable`:**
   - AUPRC medians (0.80-0.89): consistent at 2 decimal precision
   - CGAS AUPRC median=0.604 ≈ 0.60
   - Solubility Spearman median=0.779 ≈ 0.78; hERG AUPRC median=0.789 ≈ 0.79
   - HalfLife Spearman median=0.463 ≈ 0.46 (paper incorrectly calls it "AUC-ROC" but number matches)

6. **Figure claims:**
   - Figure 1 (claim 56): `insufficient_evidence` — conceptual image, no data file
   - Figure 2 (claim 57): `static_verifiable` — patterns confirmed by metrics CSVs
   - Figure 3 (claim 58): `insufficient_evidence` — bioactivity scores not in static file
   - Claim 63 (bioactivity >0.90): `execution_required` — threshold hardcoded at 0.90 in notebook but actual scores not saved

**Classification summary:**
- obvious_hallucination: 5 (4 data_fabrication + 1 result_fabrication)
- static_verifiable: 55
- insufficient_evidence: 2
- execution_required: 1
- no_code_files: 0

**Next session:** No further static analysis needed. If execution is required for claim 63, run notebook 2 with NVIDIA MolMIM API access.

---

## Session: execution (2026-04-02) — Claim 63

**Purpose:** Verify claim 63 — "framework consistently produced novel molecules with high predicted bioactivity scores, often exceeding 0.90" (Figure 3B).

**Files/context inspected:**
- `agentic-drug-discovery-main/notebooks/2_molecule_evaluation_stanfordAIAgentConf.ipynb` — checked stored cell outputs and code
- `fabscore_claude/progress.md` — prior analysis findings

**Key findings:**
1. Code at line 2589 hardcodes `CANDIDATE_TH = 0.90`; all selected molecules pass `p >= CANDIDATE_TH` filter (lines 2617-2621), so by construction every stored candidate has score ≥ 0.90.
2. Stored notebook cell outputs (lines 1797-1801) show actual bioactivity scores: 0.9131683694222734, 0.9098980579928951, 0.9024158726341833 — all exceeding 0.90.
3. These values are stored raw data in the notebook's execution outputs (not just a figure), qualifying as concrete evidence.

**Execution artifacts created:** None (reused stored notebook cell outputs; no new commands run).

**Verdict summary:** Verified — the stored notebook execution outputs confirm bioactivity scores of 0.913, 0.910, 0.902, all exceeding 0.90, consistent with the claim.

**Next session:** No further action needed for this claim.
