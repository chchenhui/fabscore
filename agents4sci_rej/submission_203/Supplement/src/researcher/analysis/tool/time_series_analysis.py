import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from scipy.signal import correlate
from scipy.stats import pearsonr

# ====================== Granger Causality Test ======================

def granger_causality(data, maxlag=2, verbose=True):
    """
    Granger Causality Test

    Usage Scenario:
    Test whether one time series is useful in forecasting another (i.e., Granger-causality).

    Parameters:
    data: 2D array-like or DataFrame, columns = time series (must be at least two)
    maxlag: maximum number of lags to check
    verbose: whether to print summary

    Returns:
    Test result dict from statsmodels
    """
    result = grangercausalitytests(data, maxlag=maxlag, verbose=verbose)
    return result

# ====================== Vector Autoregression (VAR) =================

def var_model(data, lags=1):
    """
    Vector Autoregressive (VAR) Model Fitting and Forecasting

    Usage Scenario:
    Fit a VAR model to multiple time series, capture multivariate dependencies, and forecast future values.

    Parameters:
    data: DataFrame, each column is a variable/time series
    lags: number of lags

    Returns:
    fitted VARResults object
    """
    model = VAR(data)
    results = model.fit(lags)
    return results

# =========== Cross-correlation & Trend (Pearson & Detrending) ===========

def time_series_correlation(x, y):
    """
    Pearson Correlation Coefficient Between Two Time Series

    Usage Scenario:
    Quantify linear correlation (synchrony) between two series.

    Parameters:
    x: 1D array-like
    y: 1D array-like

    Returns:
    correlation, p-value
    """
    return pearsonr(x, y)

def compute_trend(series):
    """
    Compute linear trend (slope) of a time series

    Usage Scenario:
    Assess whether a series shows an increasing or decreasing trend.

    Parameters:
    series: 1D array-like or Series

    Returns:
    slope, intercept
    """
    x = np.arange(len(series))
    coeffs = np.polyfit(x, series, deg=1)
    return coeffs[0], coeffs[1]

# ====================== Example Usage ======================

def run_examples():
    print("=== Granger Causality Test Example ===")
    np.random.seed(0)
    n = 100
    # Simulate two time series where y is partly influenced by past values of x
    x = np.random.normal(0, 1, n)
    y = np.roll(x, 1) * 0.8 + np.random.normal(0, 1, n)
    df = pd.DataFrame({'x': x, 'y': y})

    print("Testing if x Granger-causes y:")
    granger_causality(df[['y', 'x']], maxlag=2, verbose=True)
    print()

    print("=== Vector Autoregression (VAR) Example ===")
    model = var_model(df, lags=2)
    print(model.summary())
    print("One-step ahead forecast:")
    print(model.forecast(df.values[-2:], steps=1))
    print()

    print("=== Time Series Correlation Example ===")
    corr, p = time_series_correlation(df['x'], df['y'])
    print(f"Pearson correlation: {corr:.3f}, p-value: {p:.3f}")
    print()

    print("=== Trend Analysis Example ===")
    slope, intercept = compute_trend(df['y'])
    print(f"Trend slope: {slope:.3f}, intercept: {intercept:.3f}")
    print()

if __name__ == "__main__":
    run_examples()
