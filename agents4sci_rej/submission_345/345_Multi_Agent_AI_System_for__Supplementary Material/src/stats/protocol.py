"""
Statistical protocol for rigorous evaluation.
Following Linus principle: No bullshit statistics. Proper methods or nothing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import KFold
from pathlib import Path
import sys
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class StatisticalProtocol:
    """
    Defines the statistical evaluation protocol.
    Based on FDA guidance and ISPE standards.
    """
    
    # Data splitting
    train_ratio: float = 0.7
    test_ratio: float = 0.3
    n_folds: int = 5
    
    # Significance testing
    alpha: float = 0.05
    correction_method: str = 'holm'  # Holm-Bonferroni
    
    # Bootstrap parameters
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    
    # Minimum sample sizes
    min_train_samples: int = 35
    min_test_samples: int = 15
    
    # Random seed for reproducibility
    seed: int = 42


class EvaluationMetrics:
    """Calculate pharmaceutical forecasting metrics."""
    
    @staticmethod
    def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error.
        Industry standard metric.
        
        Args:
            actual: Actual revenues
            forecast: Forecasted revenues
        
        Returns:
            MAPE as percentage
        """
        # Avoid division by zero
        mask = actual != 0
        if not any(mask):
            return np.inf
        
        ape = np.abs((actual[mask] - forecast[mask]) / actual[mask])
        return np.mean(ape) * 100
    
    @staticmethod
    def year2_ape(actual_y2, forecast_y2) -> float:
        """
        Year 2 Absolute Percentage Error.
        Critical metric for pharma forecasting.
        
        Args:
            actual_y2: Actual year 2 revenue (scalar or Series)
            forecast_y2: Forecasted year 2 revenue (scalar or Series)
        
        Returns:
            APE as percentage
        """
        # Handle both scalars and Series
        if hasattr(actual_y2, '__len__'):
            # It's an array/Series - take mean
            actual_y2 = np.mean(actual_y2)
        if hasattr(forecast_y2, '__len__'):
            forecast_y2 = np.mean(forecast_y2)
            
        if actual_y2 == 0:
            return np.inf
        
        return abs((actual_y2 - forecast_y2) / actual_y2) * 100
    
    @staticmethod
    def peak_ape(actual_peak, forecast_peak) -> float:
        """
        Peak revenue APE.
        
        Args:
            actual_peak: Actual peak revenue (scalar or Series)
            forecast_peak: Forecasted peak revenue (scalar or Series)
        
        Returns:
            APE as percentage
        """
        # Handle both scalars and Series
        if hasattr(actual_peak, '__len__'):
            actual_peak = np.mean(actual_peak)
        if hasattr(forecast_peak, '__len__'):
            forecast_peak = np.mean(forecast_peak)
            
        if actual_peak == 0:
            return np.inf
        
        return abs((actual_peak - forecast_peak) / actual_peak) * 100
    
    @staticmethod
    def prediction_interval_coverage(actual: np.ndarray, 
                                    lower: np.ndarray,
                                    upper: np.ndarray) -> float:
        """
        Proportion of actuals within prediction intervals.
        
        Args:
            actual: Actual values
            lower: Lower bounds
            upper: Upper bounds
        
        Returns:
            Coverage proportion (0-1)
        """
        in_interval = (actual >= lower) & (actual <= upper)
        return np.mean(in_interval)
    
    @staticmethod
    def directional_accuracy(actual: np.ndarray, forecast: np.ndarray) -> float:
        """
        Proportion of correct direction predictions.
        
        Args:
            actual: Actual changes
            forecast: Forecasted changes
        
        Returns:
            Directional accuracy (0-1)
        """
        if len(actual) < 2:
            return np.nan
        
        actual_dir = np.diff(actual) > 0
        forecast_dir = np.diff(forecast) > 0
        
        return np.mean(actual_dir == forecast_dir)


class CrossValidation:
    """Proper cross-validation for time series."""
    
    def __init__(self, protocol: StatisticalProtocol):
        self.protocol = protocol
    
    def split_temporal(self, data: pd.DataFrame, 
                       time_col: str = 'approval_date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporal train/test split.
        Train on older drugs, test on newer.
        
        Args:
            data: Launch data
            time_col: Column with dates
        
        Returns:
            train_df, test_df
        """
        # Sort by time
        data_sorted = data.sort_values(time_col)
        
        # Split point
        n = len(data_sorted)
        n_train = int(n * self.protocol.train_ratio)
        
        # Ensure minimum sizes
        n_train = max(n_train, self.protocol.min_train_samples)
        n_train = min(n_train, n - self.protocol.min_test_samples)
        
        train_df = data_sorted.iloc[:n_train]
        test_df = data_sorted.iloc[n_train:]
        
        return train_df, test_df
    
    def kfold_validation(self, data: pd.DataFrame,
                        model_fn: callable,
                        metric_fn: callable) -> Dict[str, Any]:
        """
        K-fold cross-validation.
        
        Args:
            data: Full dataset
            model_fn: Function that trains and predicts
            metric_fn: Metric calculation function
        
        Returns:
            Dict with fold results and summary
        """
        kf = KFold(n_splits=self.protocol.n_folds, 
                  shuffle=True, 
                  random_state=self.protocol.seed)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            # Train model and predict
            predictions = model_fn(train_data, val_data)
            
            # Calculate metrics
            metrics = metric_fn(val_data, predictions)
            metrics['fold'] = fold
            
            fold_results.append(metrics)
        
        # Aggregate results
        fold_df = pd.DataFrame(fold_results)
        
        return {
            'fold_results': fold_results,
            'mean_metrics': fold_df.mean().to_dict(),
            'std_metrics': fold_df.std().to_dict(),
            'cv_score': fold_df.mean().to_dict()  # Primary CV score
        }


class HypothesisTesting:
    """Statistical hypothesis testing with proper corrections."""
    
    def __init__(self, protocol: StatisticalProtocol):
        self.protocol = protocol
    
    def compare_models(self, 
                      errors_a: np.ndarray,
                      errors_b: np.ndarray,
                      paired: bool = True) -> Dict[str, Any]:
        """
        Compare two models statistically.
        
        Args:
            errors_a: Errors from model A
            errors_b: Errors from model B
            paired: Whether observations are paired
        
        Returns:
            Dict with test results
        """
        if paired:
            # Paired t-test (same test set)
            stat, pval = stats.ttest_rel(errors_a, errors_b)
            test_name = 'paired_t_test'
        else:
            # Independent t-test
            stat, pval = stats.ttest_ind(errors_a, errors_b)
            test_name = 'independent_t_test'
        
        # Effect size (Cohen's d)
        diff = np.mean(errors_a) - np.mean(errors_b)
        pooled_std = np.sqrt((np.var(errors_a) + np.var(errors_b)) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        return {
            'test': test_name,
            'statistic': float(stat),
            'p_value': float(pval),
            'significant': pval < self.protocol.alpha,
            'effect_size': float(cohens_d),
            'mean_diff': float(diff),
            'a_better': diff < 0  # Lower error is better
        }
    
    def multiple_comparison_correction(self, p_values: List[float]) -> List[bool]:
        """
        Apply Holm-Bonferroni correction.
        
        Args:
            p_values: List of p-values
        
        Returns:
            List of significance decisions
        """
        n = len(p_values)
        if n == 0:
            return []
        
        # Sort p-values with indices
        sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
        
        # Apply Holm correction
        decisions = [False] * n
        
        for rank, (idx, pval) in enumerate(sorted_pairs):
            adjusted_alpha = self.protocol.alpha / (n - rank)
            if pval <= adjusted_alpha:
                decisions[idx] = True
            else:
                # Stop once we fail to reject
                break
        
        return decisions
    
    def bootstrap_confidence_interval(self,
                                     data: np.ndarray,
                                     statistic_fn: callable) -> Dict[str, float]:
        """
        Bootstrap confidence intervals.
        
        Args:
            data: Sample data
            statistic_fn: Function to calculate statistic
        
        Returns:
            Dict with point estimate and CI
        """
        # Original statistic
        original_stat = statistic_fn(data)
        
        # Bootstrap samples
        bootstrap_stats = []
        np.random.seed(self.protocol.seed)
        
        for _ in range(self.protocol.n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_fn(sample))
        
        # Calculate percentiles
        alpha = 1 - self.protocol.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return {
            'estimate': float(original_stat),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_width': float(ci_upper - ci_lower),
            'std_error': float(np.std(bootstrap_stats))
        }


class AcceptanceGates:
    """Define and check acceptance gates."""
    
    @staticmethod
    def check_gate_g3(protocol: StatisticalProtocol,
                      train_size: int,
                      test_size: int) -> Tuple[bool, str]:
        """
        Gate G3: Statistical rigor.
        
        Args:
            protocol: Statistical protocol
            train_size: Training set size
            test_size: Test set size
        
        Returns:
            (passed, message)
        """
        checks = []
        
        # Check train/test split
        if train_size >= protocol.min_train_samples:
            checks.append(True)
        else:
            return False, f"Training set too small: {train_size} < {protocol.min_train_samples}"
        
        if test_size >= protocol.min_test_samples:
            checks.append(True)
        else:
            return False, f"Test set too small: {test_size} < {protocol.min_test_samples}"
        
        # Check protocol settings
        if protocol.n_folds >= 5:
            checks.append(True)
        else:
            return False, f"Too few CV folds: {protocol.n_folds} < 5"
        
        if protocol.n_bootstrap >= 1000:
            checks.append(True)
        else:
            return False, f"Too few bootstrap samples: {protocol.n_bootstrap} < 1000"
        
        return True, "Gate G3 PASSED: Statistical protocol verified"
    
    @staticmethod
    def check_gate_g4(results: Dict[str, Any],
                     baselines: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Gate G4: Beat baselines.
        
        Args:
            results: Model results
            baselines: Baseline results
        
        Returns:
            (passed, message)
        """
        # Check Y2 APE
        model_y2_ape = results.get('y2_ape', np.inf)
        baseline_y2_ape = baselines.get('y2_ape', 40)  # Industry standard
        
        if model_y2_ape >= baseline_y2_ape:
            return False, f"Y2 APE not better: {model_y2_ape:.1f}% >= {baseline_y2_ape:.1f}%"
        
        # Check if significantly better
        if 'statistical_comparison' in results:
            if not results['statistical_comparison']['significant']:
                return False, "Not statistically significantly better than baseline"
        
        # Check PI coverage
        pi_coverage = results.get('pi_coverage', 0)
        if pi_coverage < 0.75:  # Should be ~80% for 80% PIs
            return False, f"PI coverage too low: {pi_coverage:.1%} < 75%"
        
        return True, f"Gate G4 PASSED: Beat baseline ({model_y2_ape:.1f}% < {baseline_y2_ape:.1f}%)"
    
    @staticmethod
    def check_gate_g5(audit_log: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Gate G5: Full reproducibility.
        
        Args:
            audit_log: Audit information
        
        Returns:
            (passed, message)
        """
        required_fields = [
            'git_commit',
            'seed',
            'data_version',
            'model_config',
            'total_cost',
            'api_calls'
        ]
        
        for field in required_fields:
            if field not in audit_log:
                return False, f"Missing audit field: {field}"
        
        # Check if git is clean
        if audit_log.get('git_dirty', True):
            return False, "Git repository has uncommitted changes"
        
        return True, "Gate G5 PASSED: Full audit trail available"


def run_statistical_protocol(data: pd.DataFrame,
                            model_fn: callable,
                            baseline_fn: callable) -> Dict[str, Any]:
    """
    Run complete statistical evaluation.
    
    Args:
        data: Full dataset
        model_fn: Model to evaluate
        baseline_fn: Baseline for comparison
    
    Returns:
        Complete results dict
    """
    # Initialize protocol
    protocol = StatisticalProtocol()
    metrics = EvaluationMetrics()
    cv = CrossValidation(protocol)
    hypothesis = HypothesisTesting(protocol)
    gates = AcceptanceGates()
    
    # Split data
    train_data, test_data = cv.split_temporal(data)
    
    # Check Gate G3
    g3_passed, g3_msg = gates.check_gate_g3(
        protocol, len(train_data), len(test_data)
    )
    
    if not g3_passed:
        return {'error': g3_msg, 'gate_g3': False}
    
    # Run cross-validation
    cv_results = cv.kfold_validation(
        train_data,
        model_fn,
        lambda val, pred: {
            'mape': metrics.mape(val['actual'], pred['forecast']),
            'y2_ape': metrics.year2_ape(val['actual_y2'], pred['forecast_y2'])
        }
    )
    
    # Test set evaluation
    model_pred = model_fn(train_data, test_data)
    baseline_pred = baseline_fn(train_data, test_data)
    
    # Calculate errors
    model_errors = np.abs(test_data['actual'] - model_pred['forecast'])
    baseline_errors = np.abs(test_data['actual'] - baseline_pred['forecast'])
    
    # Statistical comparison
    comparison = hypothesis.compare_models(model_errors, baseline_errors)
    
    # Bootstrap CIs
    y2_ci = hypothesis.bootstrap_confidence_interval(
        model_pred['y2_errors'],
        np.mean
    )
    
    # Compile results
    results = {
        'protocol': protocol.__dict__,
        'cv_results': cv_results,
        'test_results': {
            'mape': metrics.mape(test_data['actual'], model_pred['forecast']),
            'y2_ape': metrics.year2_ape(test_data['actual_y2'], model_pred['forecast_y2']),
            'peak_ape': metrics.peak_ape(test_data['actual_peak'], model_pred['forecast_peak']),
            'pi_coverage': metrics.prediction_interval_coverage(
                test_data['actual'], model_pred['lower'], model_pred['upper']
            )
        },
        'statistical_comparison': comparison,
        'bootstrap_ci': y2_ci,
        'gate_g3': g3_passed,
        'gate_g3_msg': g3_msg
    }
    
    # Check Gate G4
    baseline_results = {
        'y2_ape': metrics.year2_ape(test_data['actual_y2'], baseline_pred['forecast_y2'])
    }
    g4_passed, g4_msg = gates.check_gate_g4(results['test_results'], baseline_results)
    results['gate_g4'] = g4_passed
    results['gate_g4_msg'] = g4_msg
    
    return results