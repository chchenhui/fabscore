import scipy.stats as stats 
import numpy as np

def independent_t_test(sample1, sample2, equal_var=True):
    """
    Independent Samples t-Test

    Usage Scenario:
    Compare the means of two independent groups (e.g., control vs treatment group).

    Parameters:
    sample1: data from the first independent group
    sample2: data from the second independent group
    equal_var: assume equal variances (default=True)

    Returns:
    statistic: t-statistic
    pvalue: p-value
    """
    return stats.ttest_ind(sample1, sample2, equal_var=equal_var)

def paired_t_test(sample1, sample2):
    """
    Paired Samples t-Test

    Usage Scenario:
    Compare two related samples, such as pre-test vs post-test on the same individuals.

    Parameters:
    sample1: data before intervention
    sample2: data after intervention

    Returns:
    statistic: t-statistic
    pvalue: p-value
    """
    return stats.ttest_rel(sample1, sample2)

def mann_whitney_u_test(sample1, sample2, alternative='two-sided'):
    """
    Mann-Whitney U Test (Non-parametric alternative to independent t-test)

    Usage Scenario:
    Compare two independent groups when normality is not assumed.

    Parameters:
    sample1: first group
    sample2: second group
    alternative: type of alternative hypothesis ('two-sided', 'less', 'greater')

    Returns:
    statistic: U-statistic
    pvalue: p-value
    """
    return stats.mannwhitneyu(sample1, sample2, alternative=alternative)

def wilcoxon_test(sample1, sample2, alternative='two-sided'):
    """
    Wilcoxon Signed-Rank Test (Non-parametric alternative to paired t-test)

    Usage Scenario:
    Compare two related samples when the differences are not normally distributed.

    Parameters:
    sample1: first paired sample
    sample2: second paired sample
    alternative: type of alternative hypothesis ('two-sided', 'less', 'greater')

    Returns:
    statistic: test statistic
    pvalue: p-value
    """
    return stats.wilcoxon(sample1, sample2, alternative=alternative)

def one_way_anova(*samples):
    """
    One-Way ANOVA

    Usage Scenario:
    Compare the means across three or more independent groups.

    Parameters:
    *samples: multiple groups of data

    Returns:
    statistic: F-statistic
    pvalue: p-value
    """
    return stats.f_oneway(*samples)

def kruskal_wallis_test(*samples):
    """
    Kruskal-Wallis H Test (Non-parametric alternative to one-way ANOVA)

    Usage Scenario:
    Compare three or more independent groups without assuming normality.

    Parameters:
    *samples: multiple groups of data

    Returns:
    statistic: H-statistic
    pvalue: p-value
    """
    return stats.kruskal(*samples)

def chi2_independence_test(contingency_table):
    """
    Chi-Square Test of Independence

    Usage Scenario:
    Test the association between two categorical variables in a contingency table.

    Parameters:
    contingency_table: 2D array (rows = categories of variable A, columns = variable B)

    Returns:
    chi2: chi-square statistic
    p: p-value
    dof: degrees of freedom
    expected: expected frequencies
    """
    return stats.chi2_contingency(contingency_table)

def fisher_exact_test(contingency_table):
    """
    Fisher's Exact Test (used for small sample 2x2 tables)

    Usage Scenario:
    Test independence between two categorical variables in a 2x2 table, especially with small expected counts.

    Parameters:
    contingency_table: 2x2 array

    Returns:
    oddsratio: odds ratio
    pvalue: p-value
    """
    return stats.fisher_exact(contingency_table)

def ks_2samp_test(sample1, sample2, alternative='two-sided'):
    """
    Two-Sample Kolmogorov-Smirnov Test

    Usage Scenario:
    Test whether two samples come from the same distribution (not just mean).

    Parameters:
    sample1: first sample
    sample2: second sample
    alternative: type of alternative hypothesis ('two-sided', 'less', 'greater')

    Returns:
    statistic: KS statistic
    pvalue: p-value
    """
    return stats.ks_2samp(sample1, sample2, alternative=alternative)

# Example runner
def run_examples():
    """
    Run examples for all statistical tests
    """
    print("=== Statistical Test Examples ===\n")
    
    # Generate example data
    np.random.seed(42)
    sample1 = np.random.normal(0, 1, 30)
    sample2 = np.random.normal(0.5, 1, 30)
    sample3 = np.random.normal(1, 1, 30)
    
    print("1. Independent Samples t-Test:")
    result = independent_t_test(sample1, sample2)
    print(f"   t-statistic: {result.statistic:.4f}")
    print(f"   p-value: {result.pvalue:.4f}\n")
    
    print("2. Paired Samples t-Test:")
    result = paired_t_test(sample1, sample2)
    print(f"   t-statistic: {result.statistic:.4f}")
    print(f"   p-value: {result.pvalue:.4f}\n")
    
    print("3. Mann-Whitney U Test:")
    result = mann_whitney_u_test(sample1, sample2)
    print(f"   U-statistic: {result.statistic:.4f}")
    print(f"   p-value: {result.pvalue:.4f}\n")
    
    print("4. Wilcoxon Signed-Rank Test:")
    result = wilcoxon_test(sample1, sample2)
    print(f"   statistic: {result.statistic:.4f}")
    print(f"   p-value: {result.pvalue:.4f}\n")
    
    print("5. One-Way ANOVA:")
    result = one_way_anova(sample1, sample2, sample3)
    print(f"   F-statistic: {result.statistic:.4f}")
    print(f"   p-value: {result.pvalue:.4f}\n")
    
    print("6. Kruskal-Wallis H Test:")
    result = kruskal_wallis_test(sample1, sample2, sample3)
    print(f"   H-statistic: {result.statistic:.4f}")
    print(f"   p-value: {result.pvalue:.4f}\n")
    
    print("7. Chi-Square Test of Independence:")
    contingency_table = np.array([[10, 20, 30], [15, 25, 35]])
    chi2, p, dof, expected = chi2_independence_test(contingency_table)
    print(f"   chi2: {chi2:.4f}")
    print(f"   p-value: {p:.4f}\n")
    
    print("8. Fisher's Exact Test:")
    contingency_table = np.array([[10, 20], [15, 25]])
    oddsratio, pvalue = fisher_exact_test(contingency_table)
    print(f"   odds ratio: {oddsratio:.4f}")
    print(f"   p-value: {pvalue:.4f}\n")
    
    print("9. Kolmogorov-Smirnov Test:")
    result = ks_2samp_test(sample1, sample2)
    print(f"   KS statistic: {result.statistic:.4f}")
    print(f"   p-value: {result.pvalue:.4f}\n")

if __name__ == "__main__":
    run_examples()
