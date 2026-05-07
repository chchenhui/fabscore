"""
Statistical analysis tools for OneSim researcher.

This module provides a comprehensive set of statistical analysis functions
including parametric tests, non-parametric tests, time series analysis,
Bayesian methods, and more.
"""

# Import main statistical functions for easy access
from .parametric_test import (
    independent_t_test, paired_t_test, mann_whitney_u_test, wilcoxon_test,
    one_way_anova, kruskal_wallis_test, chi2_independence_test,
    fisher_exact_test, ks_2samp_test
)

from .time_series_analysis import (
    granger_causality, var_model, time_series_correlation, compute_trend
)

from .bayes_analysis import (
    bayes_factor_ttest, bayesian_simple_regression, bayesian_hierarchical_regression
)

from .mixed_effects_models import fit_mixedlm

from .multi_factor_analysis import factorial_anova, manova

from .repeated_measures_and_covariate_analysis import (
    repeated_measures_anova, friedman_test, ancova, multiple_comparisons
)

# 新增：OLS 回归工具导入
from .ols_regression import (
    ols_regression, summarize_ols, compute_vif, breusch_pagan_test, white_test, durbin_watson_stat
)
# 新增：交互效应分析与多重比较校正
from .interaction_analysis import interaction_analysis
from .multiple_comparison_correction import multiple_comparison_correction
# 新增 two_way_anova
from .two_way_anova import two_way_anova

# 新增：本次新增工具导入
from .linear_regression import linear_regression
from .fractional_logit_glm import fractional_logit_glm
from .kruskal_wallis_by_group import kruskal_wallis_by_group
from .spearman_rank_correlation import spearman_rank_correlation

__all__ = [
    # Parametric and Non-parametric Tests
    'independent_t_test', 'paired_t_test', 'mann_whitney_u_test', 'wilcoxon_test',
    'one_way_anova', 'kruskal_wallis_test', 'chi2_independence_test',
    'fisher_exact_test', 'ks_2samp_test',
    
    # Time Series Analysis
    'granger_causality', 'var_model', 'time_series_correlation', 'compute_trend',
    
    # Bayesian Methods
    'bayes_factor_ttest', 'bayesian_simple_regression', 'bayesian_hierarchical_regression',
    
    # Mixed Effects Models
    'fit_mixedlm',
    
    # Multi-factor Analysis
    'factorial_anova', 'manova',
    
    # Repeated Measures and Covariate Analysis
    'repeated_measures_anova', 'friedman_test', 'ancova', 'multiple_comparisons',

    # 新增：OLS 回归工具导出
    'ols_regression', 'summarize_ols', 'compute_vif', 'breusch_pagan_test', 'white_test', 'durbin_watson_stat',
    'interaction_analysis', 'multiple_comparison_correction',
    # 新增导出
    'two_way_anova',
    # 本次新增导出
    'linear_regression', 'fractional_logit_glm', 'kruskal_wallis_by_group', 'spearman_rank_correlation'
]