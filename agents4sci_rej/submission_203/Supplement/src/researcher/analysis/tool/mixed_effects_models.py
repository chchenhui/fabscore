import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM

def fit_mixedlm(data, formula, groups):
    """
    Fit a Linear Mixed Effects Model (LMM)

    Usage Scenario:
    Used when data has both fixed effects (e.g. treatment group) and random effects (e.g. subject-level intercepts for repeated measures).

    Parameters:
    data: pandas DataFrame
    formula: model formula as string, e.g., 'score ~ group'
    groups: column name indicating random effect group (e.g., 'subject')

    Returns:
    Fitted model result (statsmodels object)
    """
    model = MixedLM.from_formula(formula, groups=groups, data=data)
    result = model.fit()
    return result

def run_example():
    print("=== Mixed Linear Model Example ===")
    np.random.seed(0)
    n_subjects = 30
    n_timepoints = 3

    # Create example data (30 subjects, 3 timepoints each, 2 groups)
    df = pd.DataFrame({
        'subject': np.repeat(np.arange(n_subjects), n_timepoints),
        'time': np.tile(np.arange(n_timepoints), n_subjects),
        'group': np.repeat(['A', 'B'], n_subjects * n_timepoints // 2)
    })

    # Random subject effects (random intercepts)
    subject_effect = np.random.normal(0, 1, n_subjects)
    df['subject_intercept'] = df['subject'].map(dict(enumerate(subject_effect)))
    # Group fixed effect
    group_effect = {'A': 0, 'B': 2}
    df['group_effect'] = df['group'].map(group_effect)
    # Simulated score
    df['score'] = (
        5 + 
        1.5 * df['time'] +
        df['group_effect'] +
        df['subject_intercept'] +
        np.random.normal(0, 1, n_subjects * n_timepoints)
    )

    print(df.head())

    # Fit LMM: random intercept for each subject, fixed effect for group and time
    formula = "score ~ group + time"
    model_result = fit_mixedlm(df, formula, groups='subject')

    print("\nModel summary:")
    print(model_result.summary())

if __name__ == "__main__":
    run_example()
