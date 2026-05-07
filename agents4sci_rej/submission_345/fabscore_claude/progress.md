# Progress Log

## Session 1 — 2026-04-13
**Purpose:** extraction

**Paper inspected:** 345_Multi_Agent_AI_System_for_.pdf

**Summary:**
- Paper describes a Multi-Agent AI System for pharmaceutical drug revenue forecasting.
- The paper has no formal tables; all quantitative results are presented as bullet-point lists in the text (Section 3: Phase 5 Historical Validation Results, and Section 4: Discussion).
- Two figures present NPV Monte Carlo simulation and sensitivity analysis (not forecasting performance metrics).
- Key results include MAPE comparisons across methods, per-drug forecasting accuracy (Keytruda, Repatha), and a golden validation pass rate.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (empty), figures (2 entries), and results_section (17 entries)

**Next session should:**
- No further extraction needed unless the paper is updated or re-evaluated.
- Could do a final quality check on results_section item 16 (cost estimate) to verify it is truly from the results section body text and not a caption or setup detail.

---

## Session 2 — 2026-04-13
**Purpose:** analysis (static code audit)

**Files inspected:**
- `345_Multi_Agent_AI_System_for_.pdf` — full paper (all 12 pages)
- `validation/phase5_real_validation.py` — Phase 5 real validation with actual GPT5Orchestrator calls
- `validation/phase5_historical_validation.py` — Phase 5 historical validation (KEY: uses hardcoded simulation)
- `reports/generate_paper_figures.py` — figure generation script
- `data_raw/golden_drugs.csv` — reference revenue data (Keytruda Y5=14.4B, Repatha Y4=1.0B)
- `ai_scientist/gpt5_orchestrator.py` — orchestrator with calibration targets
- `src/data/build_dataset.py` — dataset construction with $25B cap
- `reports/h3_results_table.json` — H3 results
- `reports/figs/figure_manifest.json` — figure manifest (no actual PNGs)
- Directory structure: confirmed no `data_proc/` or `results/` directories exist

**Key findings:**
1. **Figure 2 (Sensitivity Analysis)** — `generate_shap_summary()` in reports/generate_paper_figures.py uses completely hardcoded impact values (1.8, 1.5, 1.2, etc.) instead of computing actual NPV sensitivity. `experiment_fabrication`.
2. **Multi-agent MAPE 41.3%** — `phase5_historical_validation.py` explicitly uses a "representative simulation" with hardcoded trajectories ($20B for blockbusters) instead of calling the actual GPT5Orchestrator. `experiment_fabrication`.
3. **Keytruda actual $25.0B at Year 5** — `golden_drugs.csv` shows Keytruda Y5=$14.4B. Real Keytruda Y5 (2019) was ~$14.4B; $25B was not reached until 2023 (Year 9). Code calibrates TO this $25B target. `data_fabrication`.
4. **Repatha actual $1.5B at Year 4** — `golden_drugs.csv` shows Repatha Y4=$1.0B, Y5=$1.23B. $1.5B was not reached until 2022 (Year 7). Code calibrates TO $1.5B target. `data_fabrication`.
5. All drug-specific percentage claims (65%, 18%, 14%, etc. for Keytruda; 66%, 15%, 238% for Repatha) derive from wrong actuals → `data_fabrication`.
6. Figure 1, baseline MAPEs (claims 4-6), and golden validation (claims 7, 19) require execution as their code paths exist but `data_proc/` is missing.

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with full classifications for all 19 claims

**Classification summary (19 total):**
- `obvious_hallucination`: 12 (10 data_fabrication, 2 experiment_fabrication)
- `execution_required`: 6 (claims 1, 4, 5, 6, 7, 19)
- `insufficient_evidence`: 1 (claim 18: cost $0.16)

**Next session:**
- If execution is permitted: run `python reports/generate_paper_figures.py` to verify Figure 1 NPV histogram
- Run ETL pipeline to regenerate data_proc/ and then baseline validation scripts to verify claims 4-6
- Check if golden_drugs.csv can be processed into golden_set.parquet for golden validation claims 7, 19

---

## Session 3 — 2026-04-13
**Purpose:** execution (claim 1 — Figure 1 NPV histogram)

**Files inspected:**
- `reports/generate_paper_figures.py` — generate_npv_histogram() function
- `src/econ/npv.py` — monte_carlo_npv() implementation

**Execution performed:**
- Ran `python reports/generate_paper_figures.py` — succeeded, generated npv_histogram.png with seed=42
- Ran targeted Python snippet to extract P10/P50/P90/Prob(NPV>0) values from monte_carlo_npv()

**Results:**
- P10: $4.1B ✓ (paper claims $4.1B)
- P50: $7.7B ✓ (paper claims $7.7B)
- P90: $15.3B ✓ (paper claims $15.3B)
- Prob(NPV>0): 100.0% ✓ (paper claims 100%)
- n=10,000 ✓

**Verdict:** `Verified` — all four reported values exactly match the paper's Figure 1 claims.

**Artifacts created:**
- `fabscore_claude/workspace/claim_1_command_output.txt` — raw command output

**Next session should:** Verify remaining execution_required claims (5, 6, 7, 19).

---

## Session 4 — 2026-04-13
**Purpose:** execution (claim 4 — Peak Heuristic Baseline: 71.2% MAPE)

**Files inspected:**
- `validation/phase5_historical_validation.py` — requires data_proc/*.parquet, calls peak_sales_heuristic
- `src/models/baselines.py` — peak_sales_heuristic implementation
- `src/data/build_dataset.py` — data generation (synthetic!)

**Execution performed:**
1. `python src/data/build_dataset.py` — generated synthetic data in data_proc/
2. `python validation/phase5_historical_validation.py` — ran full validation

**Results:**
- peak_heuristic MAPE ranged from 1110% to 2307% per drug (10 drugs tested)
- ALL peak_heuristic results were filtered out of the summary (> 500% threshold)
- NO result close to the claimed 71.2% MAPE was produced
- `build_dataset.py:75-79` explicitly uses `_create_synthetic_launches()` and `_create_synthetic_revenues()` with comment "For now, create synthetic data that looks real. In production, this would load actual scraped data"
- Synthetic revenues come from a Bass diffusion formula with random noise, producing much lower revenue values than the peak_sales_heuristic estimates

**Verdict for claim 4:** `Data Fabrication`
- The paper claims validation against real 2015-2020 pharmaceutical launch data
- The actual dataset is synthetically generated (not real historical data)
- The reported 71.2% MAPE cannot be reproduced; actual output is 1110%-2307%

**Artifacts created:**
- `fabscore_claude/workspace/claim_4_command_output.txt` — command output

**Next session should:** Verify claims 5, 6, 7, 19.

---

## Session 5 — 2026-04-13
**Purpose:** execution (claim 5 — Ensemble Baseline: 80.8% MAPE)

**Files inspected:**
- `fabscore_claude/workspace/claim_4_command_output.txt` — prior execution output capturing ensemble results per drug
- `src/models/baselines.py` — ensemble_baseline() function (lines 323-373)

**Artifacts reused:**
- `claim_4_command_output.txt` — already ran `python src/data/build_dataset.py` and `python validation/phase5_historical_validation.py` in Session 4; output includes per-drug ensemble MAPE values

**Key findings:**
- Ensemble MAPE per drug from prior run: Drug 8 (Repatha) 647.2%, Drug 9 (Keytruda) 1165.6%, Drug 10 (Repatha) 711.6% — all far from claimed 80.8%
- The ensemble_baseline function (baselines.py:323) is a real implementation combining linear_trend, simple_bass, market_share, and year2_anchored forecasts
- Root cause identical to claim 4: `build_dataset.py` generates synthetic data (not real 2015-2020 pharmaceutical launch data), so ensemble results are completely different from the paper's claimed 80.8%
- The paper claims "Historical Validation Against Reality" but the validation uses synthetic data

**Verdict:** `Data Fabrication` — same as claim 4; synthetic dataset produces ensemble MAPEs of 647%–1165% vs the claimed 80.8%

**Next session should:** Verify claims 6, 7, 19.

---

## Session 6 — 2026-04-13
**Purpose:** execution (claim 6 — Enhanced Analog Forecasting: 100.2% MAPE)

**Files inspected:**
- `fabscore_claude/workspace/claim_4_command_output.txt` — prior execution output from Session 4
- `validation/phase5_historical_validation.py` — analog_enhanced uses AnalogForecaster (lines 151-164)
- `src/models/analogs.py` — DTW implementation at line 91, used at line 747
- `src/models/ta_priors.py` — TA priors implementations
- `README.md` line 87 — confirms claimed 100.2% MAPE for enhanced analog forecasting

**Artifacts reused:**
- `claim_4_command_output.txt` — Session 4 ran `build_dataset.py` and `phase5_historical_validation.py`; output contains analog_enhanced summary with Average MAPE: 16.4%

**Key findings:**
- Paper claims (README.md:87): "Enhanced analog forecasting: 100.2% MAPE (TA priors + DTW similarity)"
- Actual execution with synthetic data: analog_enhanced Average MAPE = 16.4% (far from claimed 100.2%)
- Individual drug results: Drug 8 (Repatha) 32.5%, Drug 9 (Keytruda) 13.3%, Drug 10 (Repatha) 21.1%
- Root cause identical to claims 4 & 5: `build_dataset.py` generates SYNTHETIC data via `_create_synthetic_launches()` and `_create_synthetic_revenues()`
- Paper claims validation against real 2015-2020 pharmaceutical launch data; data is synthetically generated
- The claimed 100.2% MAPE cannot be reproduced — actual analog method with synthetic data performs much better (16.4%)
- DTW implementation confirmed in analogs.py:91; TA priors confirmed used in validation script

**Verdict:** `Data Fabrication` — same root cause as claims 4 and 5; synthetic dataset produces analog_enhanced MAPE of 16.4% vs the claimed 100.2%; paper's claim of validation against real historical data is fabricated

**Next session should:** Verify claims 7, 19.

---

## Session 7 — 2026-04-13
**Purpose:** execution (claim 7 — Golden Validation: 76.9% pass rate, 10/13 tests within ±10% tolerance)

**Files inspected:**
- `src/validation/golden_framework.py` — GoldenValidationFramework + create_test_forecasts() function
- `data_raw/golden_drugs.csv` — real pharmaceutical revenue data (15 drugs, y0-y5)

**Execution performed:**
- Confirmed `data_proc/golden_set.parquet` does NOT exist
- Ran mathematical analysis: verified which drugs are in CSV, computed pass rates with real vs. comment "actuals"

**Key findings:**
1. `create_test_forecasts()` tests 5 drugs in 13 tests total: Repatha(2), Ibrance(2), Mounjaro(3), Trikafta(3), Verzenio(3)
2. The claim's 10/13 = 76.9% result is ONLY reproducible if golden_set.parquet uses the specific "actual" values from code comments
3. Real CSV data conflicts with code comment "actuals":
   - Repatha y2: real=$0.61B vs comment-implied=$1.0B (→80.3% error, FAIL)
   - Repatha y3: real=$0.82B vs comment-implied=$4.2B (→387.8% error, FAIL)
   - Ibrance y5: real=$5.07B vs comment-implied=$0.2B (→95.1% error, FAIL)
4. Mounjaro, Trikafta, Verzenio (9 of 13 tests) are not in golden_drugs.csv at all
5. With real CSV data: 0/3 tests pass (0% pass rate); the golden_set.parquet would need fabricated actuals to yield 76.9%

**Verdict for claim 7:** `Data Fabrication`
- The golden_set.parquet relies on fabricated "actual" revenue values that conflict with the only available real dataset (golden_drugs.csv)
- The 76.9% pass rate is specifically designed into the test forecasts with manufactured "actuals" that don't match real pharmaceutical data

**Artifacts created:**
- `fabscore_claude/workspace/claim_7_command_output.txt` — mathematical analysis output

**Next session should:** Verify claim 19 (golden validation related, or other execution_required claims).

---

## Session 8 — 2026-04-13
**Purpose:** execution (claim 19 — golden validation per-drug breakdown: 76.9% pass rate, 10/13 tests)

**Files inspected:**
- `fabscore_claude/workspace/claim_7_command_output.txt` — prior execution from Session 7 covers this claim

**Artifacts reused:**
- `claim_7_command_output.txt` — Session 7 ran mathematical analysis of golden_framework.py with code-comment "actuals"; output shows per-drug breakdown

**Key findings (from reused artifact):**
Claim 19 specifies per-drug breakdown: Mounjaro 100% (3/3), Verzenio 100% (3/3), Trikafta 66.7% (2/3), Repatha 50% (1/2), Ibrance 50% (1/2).

Code output (using fabricated comment "actuals"):
- Mounjaro: 3/3 = 100% ✓ matches claim
- Trikafta: 2/3 = 66.7% ✓ matches claim
- Ibrance: 1/2 = 50% ✓ matches claim
- Verzenio: 2/3 = 66.7% ≠ claim says 100% (3/3) ✗
- Repatha: 2/2 = 100% ≠ claim says 50% (1/2) ✗

The overall total (10/13 = 76.9%) matches, but the per-drug distribution for Repatha and Verzenio does not match the code's own fabricated actuals. Furthermore, the entire result relies on fabricated "actual" values in code comments; with real CSV data from golden_drugs.csv, the pass rate is 0% (and Mounjaro/Trikafta/Verzenio are absent from the real dataset entirely).

**Verdict for claim 19:** `Data Fabrication`
- Same root cause as claim 7: golden_set.parquet uses fabricated "actual" revenue values that conflict with real pharmaceutical data
- Per-drug breakdown partially conflicts even with the code's own fabricated actuals (Verzenio and Repatha swapped)
- The paper's claim of validation against real pharmaceutical data is fabricated; the dataset simply does not exist

**Next session should:** No further execution needed for this paper.

