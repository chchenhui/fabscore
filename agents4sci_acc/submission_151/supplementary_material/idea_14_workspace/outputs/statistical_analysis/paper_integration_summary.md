# Statistical Analysis Summary for Paper Integration
Generated: 2025-09-08T10:18:27.369160

## Key Statistical Findings

### H1 - JailbreakBench Results
**llama4scout:**
  SE tau_0.2: AUROC 0.672, FNR 0.850 [0.739, 0.919]
  SE tau_0.3: AUROC 0.625, FNR 0.733 [0.610, 0.829]
  SE tau_0.4: AUROC 0.583, FNR 0.833 [0.720, 0.907]
  BERTScore: AUROC 0.767, FNR 0.600 [0.474, 0.714]

**qwen25:**
  SE tau_0.2: AUROC 0.529, FNR 0.983 [0.911, 0.997]
  SE tau_0.3: AUROC 0.483, FNR 0.983 [0.911, 0.997]
  BERTScore: AUROC 0.615, FNR 0.867 [0.758, 0.931]

### Methodological Notes
- Wilson confidence intervals used for all FNR comparisons (always statistically valid)
- DeLong AUROC confidence intervals inappropriate for SE due to severe score degeneracy
- Score degeneracy (85-100% identical values) constitutes evidence of SE detection failure
- Standard statistical tests confirm SE underperformance vs baselines