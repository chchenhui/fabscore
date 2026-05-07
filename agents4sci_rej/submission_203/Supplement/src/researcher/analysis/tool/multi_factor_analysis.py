import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# For MANOVA
from statsmodels.multivariate.manova import MANOVA

# For ART: No official Python package, so placeholder function is included
# (If you want real ART, you need to use the R package ARTool or try unofficial Python packages like 'art' or 'pyART' if available.)

def factorial_anova(data, formula):
    """
    Factorial ANOVA

    Usage Scenario:
    Used to analyze the main and interaction effects of two or more categorical independent variables on a continuous dependent variable.

    Parameters:
    data: pandas DataFrame
    formula: a formula string (e.g., 'y ~ A * B' for main effects and interaction)

    Returns:
    ANOVA table (pandas DataFrame)
    """
    model = smf.ols(formula, data=data).fit()
    return anova_lm(model, typ=2)

def manova(data, formula):
    """
    MANOVA (Multivariate Analysis of Variance)

    Usage Scenario:
    Used to test for differences in two or more continuous dependent variables across groups.

    Parameters:
    data: pandas DataFrame
    formula: a formula string (e.g., 'y1 + y2 ~ A * B')

    Returns:
    MANOVA table (object summary)
    """
    maov = MANOVA.from_formula(formula, data)
    return maov.mv_test()

# Example usage
def run_examples():
    print("=== Multivariate and Factorial Analysis Examples ===\n")

    # Factorial ANOVA Example
    np.random.seed(0)
    df = pd.DataFrame({
        'A': np.repeat(['low', 'high'], 30),
        'B': np.tile(np.repeat(['X', 'Y', 'Z'], 10), 2),
        'y': np.random.normal(10, 2, 60)
    })
    print("1. Factorial ANOVA (main and interaction effects):")
    print(factorial_anova(df, 'y ~ A * B'))
    print()

    # MANOVA Example
    df_m = pd.DataFrame({
        'group': np.repeat(['g1', 'g2', 'g3'], 20),
        'y1': np.random.normal(10, 2, 60),
        'y2': np.random.normal(20, 2, 60)
    })
    print("2. MANOVA (multiple DVs across groups):")
    print(manova(df_m, 'y1 + y2 ~ group'))
    print()

if __name__ == "__main__":
    run_examples()
