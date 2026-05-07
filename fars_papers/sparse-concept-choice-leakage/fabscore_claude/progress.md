# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction

**Paper inspected:** paper.pdf (sparse-concept-choice-leakage)
- Title: "Anisotropic Noise Fingerprints Reveal Concept Choice in Concept-Aware Embedding Privacy"
- 7 pages, PDF format

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created; contains tables (26 entries from Table 1 and Table 2), figures (2 entries: Figure 1 and Figure 2), and results_section (1 entry for clean STS12 Pearson 0.74 not present in tables)

**Extraction summary:**
- Table 1: 17 entries covering Concept-ID Acc, Macro F1, STS12 Pearson for Chance / Isotropic / Anisotropic / Smoothed (λ=0.2, 0.9, 0.99) conditions
- Table 2: 9 entries covering Anisotropy Ablation (Concept-ID Accuracy), Classifier Attack (Concept-ID Accuracy), and Token Privacy (AUC)
- Figure 1: pipeline overview figure
- Figure 2: covariance smoothing sweep plot
- results_section: 1 claim (STS12 Pearson 0.74 for clean embeddings, not present in any table)

**Next session should:**
- Run analysis/scoring on fs_extracted.json if needed
- Verify results against any supplementary material if added later

---

## Session 2 — 2026-04-24
**Purpose:** analysis (static code audit)

**Files inspected:**
- `paper.pdf` (via prior extraction session)
- `exp/EXPERIMENT_RESULTS/anisotropic_attack/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/isotropic_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/smoothed_mitigation/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/token_presence_privacy/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/classifier_attack/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/random_anisotropy_ablation/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/lambda_sweep/RESULTS.json`
- Full repository structure via Explore agent (all scripts in exp/concept_leakage/)

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created; contains all 29 claim classifications

**Classification summary (all 29 claims → static_verifiable):**
- Claims 1–17 (Table 1): All verified against RESULTS.json files. Numbers match within standard 3-decimal rounding. Key evidence:
  - isotropic_baseline/RESULTS.json: acc=0.1891±0.0146 (claim: 0.189±0.015), pearson=-0.0148±0.0043 (claim: -0.015±0.004)
  - anisotropic_attack/RESULTS.json: acc=1.0±0.0, pearson=0.027913±0.015178 (claim: 0.028±0.015)
  - smoothed_mitigation/RESULTS.json: lambda sweep with macro F1 and STS12 values for 0.2, 0.9, 0.99
- Claims 18–26 (Table 2): All verified against RESULTS.json files.
  - random_anisotropy_ablation/RESULTS.json: learned=0.793±0.023, random=0.033±0.024, isotropic=0.189±0.015
  - classifier_attack/RESULTS.json: A=1.000±0.000, C=0.959±0.007
  - token_presence_privacy/RESULTS.json: clean=0.8648±0.0578, iso=0.5174±0.0290, aniso=0.5193±0.0277, smooth=0.5160±0.0302
- Claim 27 (Figure 1): Qualitative + 100% accuracy claim supported by anisotropic_attack/RESULTS.json
- Claim 28 (Figure 2): Full lambda sweep in smoothed_mitigation/RESULTS.json; lambda=0.95→1.0, lambda=0.99→0.6333
- Claim 29 (results_section): Pearson range [-0.0148, 0.0296] ⊂ [-0.015, 0.030]; clean=0.7423≈0.74

**Notable observation:** No intermediate artifacts (.npy embeddings, mask checkpoints) or figure PNG files present. The smoothed_mitigation RESULTS.json has a pre-formatted `main_results_table` field. However, all RESULTS.json files also contain raw numeric values with per-seed breakdown consistent with the paper claims.

**Recommended next step:**
- Execution session could verify by regenerating results from scratch (requires HuggingFace data download), but static evidence from RESULTS.json files is sufficient for all 29 claims.
