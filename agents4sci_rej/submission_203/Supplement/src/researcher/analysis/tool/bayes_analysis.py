import numpy as np
import pandas as pd
import pingouin as pg
import pymc as pm

# ==================== Bayesian Factor t-test (pingouin) ====================

def bayes_factor_ttest(x, y, paired=False):
    """
    Bayesian Factor t-test

    Usage Scenario:
    Quantifies the evidence for/against the difference between two means, returning a Bayes Factor.

    Parameters:
    x: 1D array-like, first sample
    y: 1D array-like, second sample
    paired: bool, whether samples are paired

    Returns:
    bayesfactor: float, Bayes factor (BF10: evidence for H1 over H0)
    """
    return pg.bayesfactor_ttest(x, y, paired=paired)

# ==================== Bayesian Simple Regression (PyMC) ====================

def bayesian_simple_regression(x, y):
    """
    Bayesian Simple Linear Regression using PyMC

    Usage Scenario:
    Estimate the relationship between x and y, and quantify uncertainty via posterior distributions.

    Parameters:
    x: 1D array-like predictor
    y: 1D array-like outcome

    Returns:
    Posterior samples (ArviZ InferenceData object)
    """
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = alpha + beta * x
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(1000, tune=1000, cores=1, progressbar=False)
    return trace

# ==================== Bayesian Hierarchical Regression (PyMC) ==============

def bayesian_hierarchical_regression(df):
    """
    Bayesian Hierarchical (Multilevel) Linear Regression using PyMC

    Usage Scenario:
    Estimate group-level effects and variation across groups (e.g. subjects, schools).

    Parameters:
    df: DataFrame with columns 'y', 'x', 'group'

    Returns:
    Posterior samples (ArviZ InferenceData object)
    """
    groups = df['group'].unique()
    group_idx = df['group'].astype('category').cat.codes.values

    with pm.Model() as model:
        # Hyperpriors for group mean and sd
        mu_a = pm.Normal("mu_a", mu=0, sigma=10)
        sigma_a = pm.HalfNormal("sigma_a", 5)
        # Group-level intercepts
        a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=len(groups))
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", 1)
        mu = a[group_idx] + beta * df['x'].values
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=df['y'].values)
        trace = pm.sample(1000, tune=1000, cores=1, progressbar=False)
    return trace

# ==================== Example Usage ====================

def run_bayes_examples():
    print("=== Bayesian Factor t-test Example ===")
    np.random.seed(42)
    sample1 = np.random.normal(0, 1, 30)
    sample2 = np.random.normal(0.5, 1, 30)
    bf = bayes_factor_ttest(sample1, sample2)
    print(f"Bayes Factor (BF10) = {bf:.3f}")
    if bf > 1:
        print("Evidence for H1 (means are different)")
    else:
        print("Evidence for H0 (means are similar)")
    print()

    print("=== Bayesian Simple Linear Regression Example ===")
    x = np.linspace(0, 10, 100)
    y = 2.5 * x + 3 + np.random.normal(0, 2, 100)
    trace = bayesian_simple_regression(x, y)
    print("Posterior mean of beta:", trace.posterior['beta'].mean().item())
    print()

    print("=== Bayesian Hierarchical Regression Example ===")
    # 3 groups, different intercepts
    df = pd.DataFrame({
        'group': np.repeat(['A', 'B', 'C'], 30),
        'x': np.tile(np.linspace(0, 10, 30), 3)
    })
    group_intercepts = {'A': 2, 'B': 5, 'C': 10}
    df['y'] = df['x'] * 2 + df['group'].map(group_intercepts) + np.random.normal(0, 1, 90)
    trace_h = bayesian_hierarchical_regression(df)
    print("Posterior group intercepts mean:", trace_h.posterior['a'].mean(dim=['chain', 'draw']).values)
    print()

if __name__ == "__main__":
    run_bayes_examples()
