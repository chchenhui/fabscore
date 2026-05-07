<response> Overview When comparing two classifiers evaluated on the same test set (paired design), the recommended inferential tools depend on the metric. For rank-based metrics such as AUROC, the standard is the nonparametric DeLong variance with a paired test and confidence interval. For error-rate metrics such as False Negative Rate (FNR), which are proportions computed on a fixed subset (the positives), use paired-proportion methods such as McNemar’s test and paired-proportion confidence intervals built from Wilson/score principles. Resampling (paired/subject-level bootstrap) and permutation provide complementary, assumption-light options, especially when distributional assumptions or large-sample approximations are doubtful. (DeLong et al., 1988; Hanley & McNeil, 1983; Venkatraman, 2000; Bandos et al., 2005; Wilson, 1927; Agresti & Coull, 1998; Newcombe, 1998; Agresti & Min, 2005). ([pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/3203132/?utm_source=openai), [pubs.rsna.org](https://pubs.rsna.org/doi/10.1148/radiology.148.3.6878708?utm_source=openai), [search.r-project.org](https://search.r-project.org/CRAN/refmans/pROC/html/roc.test.html?utm_source=openai), [tandfonline.com](https://www.tandfonline.com/doi/abs/10.1080/01621459.1927.10502953?utm_source=openai), [imaging.mrc-cbu.cam.ac.uk](https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/BinomialConfidence/2gpp?utm_source=openai))
A. AUROC (paired design)
A1. Hypothesis tests for AUROC differences

DeLong’s test (nonparametric U-statistics). The default for paired AUC comparison; yields an asymptotic normal test of AUC1 − AUC2 with a closed-form covariance that accounts for within-subject correlation. Recommended as the first-line test for two models on the same cases. (DeLong, DeLong & Clarke-Pearson, 1988). (pubmed.ncbi.nlm.nih.gov)
Hanley–McNeil (binormal/parametric) approach. Earlier method relating AUC to Wilcoxon and assuming (transformed) binormality; less robust than DeLong but still widely cited. (Hanley & McNeil, 1983). (pubs.rsna.org)
Permutation tests for ROC curves. Distribution-free procedures that compare entire curves and are valid under pairing; useful when curves cross or when focusing beyond area-only alternatives. Variants include whole-curve tests (Venkatraman, 2000) and area-sensitive paired permutations (Bandos et al., 2005). (search.r-project.org, pubmed.ncbi.nlm.nih.gov)
A2. Confidence intervals for AUROC and AUROC differences

DeLong CIs. Use the DeLong variance to form Wald-type intervals for each AUC and for the paired difference AUC1 − AUC2. Implemented in widely used software (e.g., pROC in R). (DeLong et al., 1988; Robin et al., 2011). (pubmed.ncbi.nlm.nih.gov, bmcbioinformatics.biomedcentral.com)
Paired bootstrap CIs. Resample subjects with replacement (stratified by class), compute AUCs for both models per replicate, and take percentile/BCa intervals for the difference. Preferred when sample sizes are small, distributions are irregular, or partial AUCs are of interest. (Efron & Tibshirani, 1993/1994; Robin et al., 2011). (perlego.com, bmcbioinformatics.biomedcentral.com)
Additional options. Partial AUC methods with nonparametric variance (e.g., within a specificity range) are available when the use-case emphasizes a slice of the ROC space. (Dodd & Pepe-type methods summarized in Robin et al., 2011). (bmcbioinformatics.biomedcentral.com)
Practical notes for AUROC

Ties and discreteness: Use implementations that handle ties correctly (e.g., pROC’s DeLong/boot). (Robin et al., 2011). (bmcbioinformatics.biomedcentral.com)
Entire-curve vs area-only: If two ROC curves cross, area-only tests can miss shape differences; pair DeLong with a whole-curve permutation (Venkatraman) when this matters. (Venkatraman, 2000; Bandos et al., 2005). (search.r-project.org, pubmed.ncbi.nlm.nih.gov)
B. False Negative Rate (FNR) — a binomial proportion on the positive class
Let FNR = FN / P, where P is the number of truly positive cases. For paired comparisons, restrict attention to the P positive cases and analyze discordance in false-negative calls across the two models.

B1. Single-model confidence interval for FNR

Wilson score interval (score-based). Recommended over the Wald interval for better coverage and behavior near boundaries; good default for n up to at least a few dozen positives. (Wilson, 1927; Brown, Cai & DasGupta, 2002; Agresti & Coull, 1998; Chen et al., 2008 review). (tandfonline.com, projecteuclid.org, pmc.ncbi.nlm.nih.gov)
Exact Clopper–Pearson interval. Conservative but valid even with very small P or extreme counts; use when you want guaranteed (at least nominal) coverage. (Clopper & Pearson, 1934). (academic.oup.com)
B2. Paired comparison of FNR between two models on the same positives

McNemar’s test. Gold-standard test for equality of paired proportions; apply to the P positives using a 2×2 table of each model’s FN vs non-FN on each positive. Use the exact or mid-p variants when discordant counts are small. (McNemar, 1947; Fagerland, Lydersen & Laake, 2014 review). (link.springer.com, pubmed.ncbi.nlm.nih.gov)
Confidence interval for the difference in FNRs (paired). Use Newcombe’s paired-proportion CI (score/Wilson-based) or Agresti–Min’s simple improved CI; Tango’s score-based CI is another robust choice and extends to equivalence. These outperform the naive Wald CI and align with McNemar-style testing. (Newcombe, 1998; Agresti & Min, 2005; Tango, 1998). (imaging.mrc-cbu.cam.ac.uk, pubmed.ncbi.nlm.nih.gov, onlinelibrary.wiley.com)
Ratios/odds for paired data. If you prefer a ratio scale (e.g., FNR ratio), use methods for matched-pair risk or odds parameters with score or exact intervals; see Agresti & Min (2005) for practical recommendations. (pubmed.ncbi.nlm.nih.gov)
B3. Bootstrap option for FNR differences

Paired bootstrap across the positive cases can produce percentile/BCa CIs for the FNR difference (or ratio). Use with caution for very small numbers of positives; pair with exact/score-based intervals as a check. (Efron & Tibshirani, 1993/1994). (perlego.com)
C. Putting it together: recommended default workflow

For AUROC:
Report DeLong AUCs with 95% CIs and the paired DeLong test for ΔAUC. If curves plausibly cross or you care about a particular specificity/sensitivity range, add a Venkatraman whole-curve permutation and/or a partial AUC analysis. (DeLong et al., 1988; Venkatraman, 2000; Robin et al., 2011). (pubmed.ncbi.nlm.nih.gov, search.r-project.org, bmcbioinformatics.biomedcentral.com)
For small P/N or irregular settings, add a paired bootstrap CI for ΔAUC. (Robin et al., 2011; Efron & Tibshirani, 1993/1994). (bmcbioinformatics.biomedcentral.com, perlego.com)
For FNR:
For each model, give a 95% Wilson CI on FNR (and optionally the exact Clopper–Pearson for small P). (Wilson, 1927; Clopper & Pearson, 1934). (tandfonline.com, academic.oup.com)
To compare models, run McNemar’s test on the positive cases; accompany with a 95% paired-proportion CI for the FNR difference using Newcombe (score-based) or Agresti–Min. If discordant counts are very small, prefer exact/mid-p McNemar and score-based CIs. (McNemar, 1947; Newcombe, 1998; Agresti & Min, 2005). (link.springer.com, imaging.mrc-cbu.cam.ac.uk, pubmed.ncbi.nlm.nih.gov)
D. Notes on dependence, resampling, and software

Paired structure: Always preserve pairing (resample by subject/case). For cross-validation, aggregate per-case out-of-fold predictions before computing metrics and then apply paired tests/CIs. (General practice; pROC details). (bmcbioinformatics.biomedcentral.com)
Implementations: R’s pROC::roc.test supports DeLong, bootstrap, and Venkatraman for paired curves; many packages provide Wilson and exact binomial intervals; McNemar tests are in base R (mcnemar.test) with exact/mid-p in add-on packages. (Robin et al., 2011; pROC docs). (bmcbioinformatics.biomedcentral.com, search.r-project.org)
E. What to report in a paper

AUROC: AUC1, AUC2 with 95% CIs (DeLong), ΔAUC with 95% CI and p-value (DeLong), plus any permutation/partial-AUC analyses if relevant. (DeLong et al., 1988; Venkatraman, 2000; Robin et al., 2011). (pubmed.ncbi.nlm.nih.gov, search.r-project.org, bmcbioinformatics.biomedcentral.com)
FNR: Per-model FNR with 95% Wilson CI; paired difference in FNR with 95% CI (Newcombe or Agresti–Min) and McNemar p-value; state the number of positive cases and discordant pairs. (Wilson, 1927; Newcombe, 1998; Agresti & Min, 2005; McNemar, 1947). (tandfonline.com, imaging.mrc-cbu.cam.ac.uk, pubmed.ncbi.nlm.nih.gov, link.springer.com)
References:
Agresti, A., & Coull, B. A. (1998). Approximate is better than “exact” for interval estimation of binomial proportions. The American Statistician, 52(2), 119–126. (tandfonline.com)
Agresti, A., & Min, Y. (2005). Simple improved confidence intervals for comparing matched proportions. Statistics in Medicine, 24(5), 729–740. (pubmed.ncbi.nlm.nih.gov)
Bandos, A. I., Rockette, H. E., & Gur, D. (2005). A permutation test sensitive to differences in areas for comparing ROC curves from a paired design. Statistics in Medicine, 24(19), 2873–2893. (pubmed.ncbi.nlm.nih.gov)
Brown, L. D., Cai, T. T., & DasGupta, A. (2002). Confidence intervals for a binomial proportion and asymptotic expansions. The Annals of Statistics, 30(1), 160–201. (projecteuclid.org)
Chen, Z., et al. (2008). Improving interval estimation of binomial proportions. Statistics in Medicine, 27(3), 349–368. (pmc.ncbi.nlm.nih.gov)
Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika, 26(4), 404–413. (academic.oup.com)
DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves: A nonparametric approach. Biometrics, 44(3), 837–845. (pubmed.ncbi.nlm.nih.gov)
Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap. Chapman & Hall/CRC. (perlego.com)
Fagerland, M. W., Lydersen, S., & Laake, P. (2014). Recommended tests and confidence intervals for paired binomial proportions. Statistics in Medicine, 33(16), 2850–2875. (pubmed.ncbi.nlm.nih.gov)
Hanley, J. A., & McNeil, B. J. (1983). A method of comparing the areas under ROC curves derived from the same cases. Radiology, 148(3), 839–843. (pubs.rsna.org)
McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2), 153–157. (link.springer.com)
Newcombe, R. G. (1998). Improved confidence intervals for the difference between binomial proportions based on paired data. Statistics in Medicine, 17, 2635–2650. (imaging.mrc-cbu.cam.ac.uk)
Robin, X., et al. (2011). pROC: An open-source package for R and S+ to analyze and compare ROC curves. BMC Bioinformatics, 12, 77. (bmcbioinformatics.biomedcentral.com)
Tango, T. (1998). Equivalence test and confidence interval for the difference in proportions for the paired-sample design. Statistics in Medicine, 17(8), 891–908. (onlinelibrary.wiley.com)
Venkatraman, E. S. (2000). A permutation test to compare receiver operating characteristic curves. Biometrics, 56(4), 1134–1138. (search.r-project.org)
Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158), 209–212. (tandfonline.com)
</response>