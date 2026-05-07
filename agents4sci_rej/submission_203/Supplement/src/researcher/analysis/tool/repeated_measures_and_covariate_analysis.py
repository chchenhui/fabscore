import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import pingouin as pg

def repeated_measures_anova(data, dv, subject, within):
    """
    Repeated Measures ANOVA

    Usage Scenario:
    Used when the same subjects are measured under multiple conditions or time points.

    Parameters:
    data: pandas DataFrame
    dv: dependent variable name (str)
    subject: subject identifier column name (str)
    within: within-subject factor name (str)

    Returns:
    ANOVA table (pandas DataFrame)
    """
    return pg.rm_anova(data=data, dv=dv, within=within, subject=subject, detailed=True)

def friedman_test(*groups):
    """
    Friedman Test (Non-parametric alternative to repeated measures ANOVA)

    Usage Scenario:
    Used for comparing more than two related samples when normality is not assumed.

    Parameters:
    *groups: each group should be a list or array of values (same subjects)

    Returns:
    statistic: test statistic
    pvalue: p-value
    """
    return stats.friedmanchisquare(*groups)

def ancova(data, dv, covar, between):
    """
    Analysis of Covariance (ANCOVA)

    Usage Scenario:
    Tests group differences while controlling for the effect of a covariate.

    Parameters:
    data: pandas DataFrame
    dv: dependent variable (str)
    covar: covariate column name (str)
    between: between-subject factor (str)

    Returns:
    ANCOVA table (pandas DataFrame)
    """
    formula = f"{dv} ~ C({between}) + {covar}"
    model = smf.ols(formula, data=data).fit()
    return sm.stats.anova_lm(model, typ=2)

def multiple_comparisons(data, group_col, value_col, method='tukey'):
    """
    Multiple Comparison Correction (Post-hoc Test)

    Usage Scenario:
    Applied after ANOVA to identify which groups differ significantly.

    Parameters:
    data: pandas DataFrame
    group_col: column name indicating group membership
    value_col: column name of dependent variable
    method: 'tukey' for Tukey HSD, 'bonferroni', 'fdr_bh', etc.

    Returns:
    Tukey: TukeyHSD result (if method='tukey')
    Other: adjusted p-values via multipletests
    """
    if method == 'tukey':
        return pairwise_tukeyhsd(data[value_col], data[group_col])
    else:
        # Example: using multipletests for custom p-value list
        # (here we fake some p-values just for illustration)
        fake_pvals = np.array([0.01, 0.04, 0.10, 0.03])
        reject, pvals_corrected, _, _ = multipletests(fake_pvals, method=method)
        return reject, pvals_corrected

# Example usage
def run_examples():
    print(">>> run_examples() has started")
    print("=== Advanced Statistical Test Examples ===\n")

    # Repeated Measures Example
    # Repeated Measures Example
    df = pd.DataFrame({
        'subject': np.tile(np.arange(1, 11), 3),
        'condition': np.repeat(['A', 'B', 'C'], 10),
        'score': np.concatenate([
            np.random.normal(loc=10, scale=2, size=10),
            np.random.normal(loc=12, scale=2, size=10),
            np.random.normal(loc=11, scale=2, size=10)
        ])
    })
    print("1. Repeated Measures ANOVA:")
    print(repeated_measures_anova(df, dv='score', subject='subject', within='condition'))
    print()

    # Friedman Example
    group1 = np.random.normal(10, 2, 10)
    group2 = np.random.normal(12, 2, 10)
    group3 = np.random.normal(11, 2, 10)
    print("2. Friedman Test:")
    stat, p = friedman_test(group1, group2, group3)
    print(f"   Statistic: {stat:.4f}, p-value: {p:.4f}")
    print()

    # ANCOVA Example
    ancova_df = pd.DataFrame({
        'group': np.repeat(['control', 'treatment'], 15),
        'score': np.concatenate([
            np.random.normal(50, 5, 15),
            np.random.normal(55, 5, 15)
        ]),
        'age': np.random.randint(20, 40, 30)
    })
    print("3. ANCOVA:")
    print(ancova(ancova_df, dv='score', covar='age', between='group'))
    print()

    # Multiple Comparisons Example
    print("4. Multiple Comparisons - Tukey HSD:")
    print(multiple_comparisons(df, group_col='condition', value_col='score', method='tukey'))
    print()

if __name__ == "__main__":
    run_examples()
