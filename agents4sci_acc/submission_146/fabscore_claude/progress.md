# Progress Log

## Session 1 — 2026-04-23

**Purpose:** extraction

**Paper inspected:** `146_Educational_attainment_and.pdf`
- Title: "Educational attainment and cognitive profile heterogeneity: age-stratified web-based analysis finds no detectable association"
- A 20-page PDF (including supplementary material, checklists, and references)
- Key content: regression analyses testing whether educational attainment predicts cognitive profile heterogeneity (percentile range and IQR) in 1,083 adults; two main tables (Table 1: multiple regression results; Table 2: age × education interaction results); four figures

**JSON files created/updated this session:**
- `fabscore_claude/fs_extracted.json` — created from scratch

**What was extracted:**
- **Tables (62 entries):** Full extraction of Table 1 (multiple regression for Percentile Range and Percentile IQR, including all education contrasts, covariates, and model fit statistics) and Table 2 (age × education interaction models for both outcomes, including main effects, interaction terms, and model fit statistics).
- **Figures (4 entries):** Figure 1 (discriminant validity scatterplots), Figure 2 (box plots by education level), Figure 3 (age-stratified scatter plots), Figure 4 (sensitivity analysis panels).
- **Results section (13 entries):** Discriminant validity r-values (not in tables), distributional statistics for heterogeneity metrics, primary model R2/F values from text body (which differ from Table 1 values — likely education-only models), Breusch-Pagan homoscedasticity test statistics, Durbin-Watson residual independence statistics, Shapiro-Wilk normality test statistics for both primary and interaction models, VIF values, and coefficient-of-variation sensitivity analysis results.

**Notes:**
- The body text cites R2=0.0148 (F=1.337, p=0.1926) for percentile range and R2=0.0139 (F=1.255, p=0.2380) for percentile IQR from what appears to be an education-only or differently parameterized model; Table 1 shows R2=0.005 and R2=0.012 respectively for the full covariate model. Both were captured.
- No supplementary figures or tables beyond Table 2 and Figure 4 were present in this PDF (Appendix A referred to these as supplementary materials).

**Next session should:**
- Verify extraction completeness if additional supplementary files become available.
- No further extraction needed from this paper unless supplementary data tables (beyond Table 2) are provided separately.

---

## Session 2 — 2026-04-23

**Purpose:** analysis (static)

**Files inspected:**
- `agents4science_supp_materials/analysis/script_execution_logs/step_3_output.txt`
- `agents4science_supp_materials/analysis/script_execution_logs/step_4_output.txt`
- `agents4science_supp_materials/analysis/script_execution_logs/step_5_output.txt`
- `agents4science_supp_materials/analysis/script_execution_logs/step_6_output.txt` (first 100 lines)
- `agents4science_supp_materials/analysis/execution_output_csvs/step4_model_fit_20250707_120331.csv`
- `agents4science_supp_materials/analysis/execution_output_csvs/step4_primary_regression_results_20250707_120331.csv`
- `agents4science_supp_materials/analysis/execution_output_csvs/step5_interaction_regression_results_20250707_123144.csv`
- `agents4science_supp_materials/analysis/execution_output_csvs/step5_model_fit_20250707_123144.csv`

**JSON files created/updated this session:**
- `fabscore_claude/fs_analysis.json` — created from scratch

**Summary of classifications (79 claims total):**

- **static_verifiable (72):** Claims 1–19 (Table 1 Percentile Range — all match step4 log/CSV exactly), claims 20–38 (Table 1 Percentile IQR — all match step4 log/CSV exactly), claims 39–62 (Table 2 interaction model coefficients and fit stats — all match step5 CSV exactly), claim 63 (Figure 1 correlations and distributions — match step3 log), claim 65 (Figure 3 age group sizes match step5 log), claims 67–70 (results section distributional stats and correlations match step3 log), claims 74 (DW), 75 (VIF), 77 (interaction model SW), 78 (interaction model VIF) — all match step4/step5 logs.

- **obvious_hallucination / result_fabrication (5):**
  - Claim 64 (Figure 2): Education level sample sizes don't match execution log and sum to 1323 (not 1083). Log shows edu3=240, edu5=63, edu6=190, edu7=22, edu8=68; paper claims Some College=308, Associate's=240, Master's=173, Professional=80, Ph.D.=22.
  - Claims 71–72 (results section primary model R²/F/p): Paper says R²=0.0148, F=1.337, p=0.1926 (range) and R²=0.0139, F=1.255, p=0.2380 (IQR). Actual log: R²=0.005, F=0.448, p=0.952 (range) and R²=0.012, F=0.982, p=0.467 (IQR).
  - Claim 73 (Breusch-Pagan): Paper reports range LM=9.3037 (same as IQR), but log shows range LM=9.4945. Paper copied IQR's LM for both.
  - Claim 76 (Shapiro-Wilk): Paper reports range W=0.9950 (same as IQR), but log shows range W=0.9543. Paper copied IQR's W for both.

- **obvious_hallucination / experiment_fabrication (1):**
  - Claim 79 (CoV sensitivity analysis): Step6 script uses formula with individual time_of_day (24 levels, Df Model=35) instead of binned time_of_day (4 categories) as in primary models. Actual result: R²=0.060, F=1.926, p=0.001 (SIGNIFICANT). Paper claims: R²=0.0261, F=0.8029, p=0.7869 (null finding). Paper falsely characterizes a significant result as null.

- **execution_required (1):**
  - Claim 66 (Figure 4): Qualitative figure description claiming null findings for all sensitivity analyses. CoV model in step6 shows p=0.001 (significant), conflicting with "null findings" characterization. Figure script (figure_4.py) available but needs execution to verify visual content of all panels.

**Recommended next step:**
- Run `figure_4.py` to verify what Panel A shows for the CoV sensitivity analysis.
- Consider whether a re-run of `analysis_step_6.py` with binned time_of_day matches the paper's claimed R²=0.0261, F=0.8029, p=0.7869 (would need to modify the script and re-run).
- Note: The most serious discrepancy is the falsely-characterized CoV significance, but the session 1 note suggests the R²/F/p values in the results section (claims 71–72) might be from a differently-specified model not present in the codebase.

---

## Session 3 — 2026-04-23

**Purpose:** execution (verify claim 66 — Figure 4)

**Files inspected:**
- `agents4science_supp_materials/visuals/generated_scripts/figure_4.py` — figure generation script
- `agents4science_supp_materials/analysis/execution_output_csvs/step6_sensitivity_coefficient_variation_20250707_125813.csv` — Panel A data
- `agents4science_supp_materials/analysis/script_execution_logs/step_6_output.txt` (lines 1–420, 675–1050) — CoV and collapsed education regression results; split-half reliability results
- `agents4science_supp_materials/analysis/execution_output_csvs/step6_split_half_reliability_20250707_125813.csv` — Panel C/D data

**No new commands run** — existing artifacts were sufficient.

**Key findings for claim 66 (Figure 4):**

1. **Panel A (CoV)**: Individual education-level coefficients in the CoV model are all non-significant (p=0.156–0.837). However, overall model F=1.926, p=0.001 (significant, driven by time_of_day covariates). A boxplot of CoV by education level would show overlapping distributions consistent with "no systematic relationship with educational attainment" for education-specific effects.

2. **Panel B (Collapsed education)**: F=0.738, p=0.846 — genuine null result, education effects non-significant (High p=0.576, Medium p=0.468). Consistent with "overlapping distributions with minimal between-group differences."

3. **Panels C and D (Split-half reliability)**:
   - Range: r=0.0089, p=0.770
   - IQR: r=-0.0324, p=0.286
   - These near-ZERO correlations contradict the claim of "adequate internal consistency." In psychometrics, split-half reliability requires r>0.5 for even marginal adequacy. r≈0.009 indicates NO reliability.

4. **Sample size n=1,083**: Confirmed correct.

**Verdict summary:** `Result Fabrication`
- The claim that "Split-half reliability analyses (Panels C and D) demonstrated adequate internal consistency" is directly contradicted by actual r=0.009 and r=-0.032 (effectively zero correlations).
- The claim that all panels confirm "null findings" is also complicated by the CoV model being statistically significant overall (p=0.001), though education-specific effects are individually non-significant.

**Next session:** No further action needed for claim 66.
