"""
Statistical protocol for pharmaceutical forecasting experiments.
Following Linus principle: Rigorous stats or it's garbage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ExperimentConfig:
    """Configuration for statistical experiments."""
    
    # Data splitting
    train_pct: float = 0.7
    test_pct: float = 0.3
    cv_folds: int = 5
    
    # Temporal splitting
    temporal_split_year: int = 2020  # Train <= 2020, Test >= 2021
    
    # Multiple comparisons
    alpha: float = 0.05
    correction_method: str = 'holm'  # 'holm', 'bonferroni', 'fdr'
    
    # Power analysis
    effect_size: float = 0.1  # 10% improvement in MAPE
    power: float = 0.8
    min_sample_size: int = 50
    
    # Bootstrap
    n_bootstrap: int = 5000
    confidence_level: float = 0.95
    
    # Random seed
    seed: int = 42


class EvaluationMetrics:
    """
    Industry-standard evaluation metrics for pharmaceutical forecasting.
    These are the metrics that actually matter to pharma companies.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def mape(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error.
        Industry standard for forecast accuracy.
        """
        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def year2_ape(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Year 2 Absolute Percentage Error.
        Critical for go/no-go decisions.
        """
        if len(predicted) < 2 or len(actual) < 2:
            return np.inf
        
        if actual[1] == 0:
            return np.inf
        
        return abs(predicted[1] - actual[1]) / actual[1] * 100
    
    def peak_ape(self, predicted: np.ndarray, actual: np.ndarray, years: int = 5) -> float:
        """
        Peak sales Absolute Percentage Error.
        Critical for valuation.
        """
        pred_peak = np.max(predicted[:years])
        actual_peak = np.max(actual[:years])
        
        if actual_peak == 0:
            return np.inf
        
        return abs(pred_peak - actual_peak) / actual_peak * 100
    
    def directional_accuracy(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Directional accuracy - did we predict the right trend?
        """
        if len(predicted) < 2 or len(actual) < 2:
            return 0.5
        
        pred_trend = np.diff(predicted) > 0
        actual_trend = np.diff(actual) > 0
        
        return np.mean(pred_trend == actual_trend)
    
    def prediction_interval_coverage(self, predicted: np.ndarray, actual: np.ndarray, 
                                   lower: np.ndarray, upper: np.ndarray) -> float:
        """
        Prediction interval coverage.
        How often do actuals fall within our confidence intervals?
        """
        if len(predicted) != len(actual) or len(predicted) != len(lower) or len(predicted) != len(upper):
            return 0.0
        
        coverage = np.mean((actual >= lower) & (actual <= upper))
        return coverage
    
    def calculate_all_metrics(self, predicted: np.ndarray, actual: np.ndarray,
                            lower: Optional[np.ndarray] = None, 
                            upper: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        """
        metrics = {
            'mape': self.mape(predicted, actual),
            'year2_ape': self.year2_ape(predicted, actual),
            'peak_ape': self.peak_ape(predicted, actual),
            'directional_accuracy': self.directional_accuracy(predicted, actual),
            'rmse': np.sqrt(np.mean((predicted - actual) ** 2)),
            'mae': np.mean(np.abs(predicted - actual))
        }
        
        if lower is not None and upper is not None:
            metrics['pi_coverage'] = self.prediction_interval_coverage(predicted, actual, lower, upper)
        
        return metrics


class StatisticalProtocol:
    """
    Statistical protocol for pharmaceutical forecasting experiments.
    Implements proper cross-validation, multiple comparison correction, and power analysis.
    """
    
    def __init__(self, config: ExperimentConfig = None, seed: int = 42):
        self.config = config or ExperimentConfig(seed=seed)
        self.seed = seed
        np.random.seed(seed)
        
        self.metrics = EvaluationMetrics()
        self.results = {}
    
    def temporal_split(self, data: pd.DataFrame, date_col: str = 'approval_date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporal train/test split.
        Train on historical data, test on future data.
        """
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
        
        # Split by year
        train_data = data[data[date_col].dt.year <= self.config.temporal_split_year].copy()
        test_data = data[data[date_col].dt.year > self.config.temporal_split_year].copy()
        
        return train_data, test_data
    
    def stratified_split(self, data: pd.DataFrame, stratify_col: str = 'therapeutic_area') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stratified train/test split by therapeutic area.
        Ensures balanced representation across TAs.
        """
        from sklearn.model_selection import train_test_split
        
        train_data, test_data = train_test_split(
            data, 
            test_size=self.config.test_pct,
            stratify=data[stratify_col],
            random_state=self.seed
        )
        
        return train_data, test_data
    
    def cross_validate(self, train_data: pd.DataFrame, model_func, 
                      stratify_col: str = 'therapeutic_area', 
                      target_col: str = 'revenue_usd') -> Dict[str, Any]:
        """
        K-fold cross-validation with stratification.
        """
        # Prepare data
        X = train_data.drop(columns=[target_col], errors='ignore')
        y = train_data[target_col] if target_col in train_data.columns else None
        
        # Stratified K-fold
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.seed)
        
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, X[stratify_col])):
            train_fold = train_data.iloc[train_idx]
            val_fold = train_data.iloc[val_idx]
            
            # Train model
            model = model_func(train_fold)
            
            # Validate
            val_predictions = []
            val_actuals = []
            
            for _, row in val_fold.iterrows():
                # Generate forecast
                forecast = model.forecast(row, years=5)
                actual = self._get_actual_revenues(row['launch_id'])
                
                if actual is not None and len(actual) > 0:
                    val_predictions.append(forecast)
                    val_actuals.append(actual)
            
            if val_predictions:
                # Calculate metrics for this fold
                fold_metrics = self._calculate_fold_metrics(val_predictions, val_actuals)
                fold_metrics['fold'] = fold
                cv_results.append(fold_metrics)
        
        # Aggregate results
        return self._aggregate_cv_results(cv_results)
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = None) -> List[float]:
        """
        Multiple comparison correction for multiple hypotheses.
        """
        method = method or self.config.correction_method
        
        if method == 'holm':
            return self._holm_bonferroni_correction(p_values)
        elif method == 'bonferroni':
            return self._bonferroni_correction(p_values)
        elif method == 'fdr':
            return self._fdr_correction(p_values)
        else:
            raise ValueError(f"Unknown correction method: {method}")
    
    def _holm_bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Holm-Bonferroni correction (step-down)."""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        corrected = np.zeros(n)
        
        for i, idx in enumerate(sorted_indices):
            corrected[idx] = min(p_values[idx] * (n - i), 1.0)
        
        return corrected.tolist()
    
    def _bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Bonferroni correction."""
        n = len(p_values)
        return [min(p * n, 1.0) for p in p_values]
    
    def _fdr_correction(self, p_values: List[float]) -> List[float]:
        """False Discovery Rate correction (Benjamini-Hochberg)."""
        from statsmodels.stats.multitest import multipletests
        _, corrected, _, _ = multipletests(p_values, method='fdr_bh')
        return corrected.tolist()
    
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                    confidence_level: float = None) -> Tuple[float, float]:
        """
        Bootstrap confidence interval.
        """
        confidence_level = confidence_level or self.config.confidence_level
        
        if len(data) == 0:
            return (0.0, 0.0)
        
        # Bootstrap samples
        bootstrap_samples = []
        for _ in range(self.config.n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_samples.append(np.mean(sample))
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
        upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    def power_analysis(self, effect_size: float = None, alpha: float = None, 
                      power: float = None, sample_size: int = None) -> Dict[str, float]:
        """
        Power analysis for detecting effect sizes.
        """
        effect_size = effect_size or self.config.effect_size
        alpha = alpha or self.config.alpha
        power = power or self.config.power
        
        if sample_size is None:
            # Calculate required sample size
            from statsmodels.stats.power import tt_solve_power
            required_n = tt_solve_power(effect_size, alpha=alpha, power=power, alternative='two-sided')
            return {
                'required_sample_size': int(np.ceil(required_n)),
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power
            }
        else:
            # Calculate achieved power
            from statsmodels.stats.power import ttest_power
            achieved_power = ttest_power(effect_size, nobs=sample_size, alpha=alpha, alternative='two-sided')
            return {
                'achieved_power': achieved_power,
                'effect_size': effect_size,
                'alpha': alpha,
                'sample_size': sample_size
            }
    
    def _get_actual_revenues(self, launch_id: str) -> Optional[np.ndarray]:
        """Get actual revenues for a launch (placeholder)."""
        # This would load from launch_revenues.parquet
        # For now, return None to indicate no actual data
        return None
    
    def _calculate_fold_metrics(self, predictions: List[np.ndarray], 
                               actuals: List[np.ndarray]) -> Dict[str, float]:
        """Calculate metrics for a single CV fold."""
        if not predictions or not actuals:
            return {}
        
        # Align predictions and actuals
        min_len = min(len(p) for p in predictions + actuals)
        aligned_preds = [p[:min_len] for p in predictions]
        aligned_actuals = [a[:min_len] for a in actuals]
        
        # Calculate metrics
        all_mape = []
        all_year2_ape = []
        all_peak_ape = []
        
        for pred, actual in zip(aligned_preds, aligned_actuals):
            all_mape.append(self.metrics.mape(pred, actual))
            all_year2_ape.append(self.metrics.year2_ape(pred, actual))
            all_peak_ape.append(self.metrics.peak_ape(pred, actual))
        
        return {
            'mape_mean': np.mean(all_mape),
            'mape_std': np.std(all_mape),
            'year2_ape_mean': np.mean(all_year2_ape),
            'year2_ape_std': np.std(all_year2_ape),
            'peak_ape_mean': np.mean(all_peak_ape),
            'peak_ape_std': np.std(all_peak_ape)
        }
    
    def _aggregate_cv_results(self, cv_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        if not cv_results:
            return {}
        
        # Calculate means and standard errors
        metrics = ['mape_mean', 'mape_std', 'year2_ape_mean', 'year2_ape_std', 
                  'peak_ape_mean', 'peak_ape_std']
        
        aggregated = {}
        for metric in metrics:
            values = [result[metric] for result in cv_results if metric in result]
            if values:
                aggregated[metric] = np.mean(values)
                aggregated[f"{metric}_se"] = np.std(values) / np.sqrt(len(values))
        
        return aggregated


def run_experiment(data: pd.DataFrame, models: Dict[str, Any], 
                  config: ExperimentConfig = None) -> Dict[str, Any]:
    """
    Run a complete statistical experiment.
    
    Args:
        data: Pharmaceutical launch data
        models: Dictionary of model functions
        config: Experiment configuration
    
    Returns:
        Dictionary with experiment results
    """
    config = config or ExperimentConfig()
    protocol = StatisticalProtocol(config)
    
    # Temporal split
    train_data, test_data = protocol.temporal_split(data)
    
    results = {
        'config': config,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'models': {}
    }
    
    # Test each model
    for model_name, model_func in models.items():
        print(f"Testing {model_name}...")
        
        # Cross-validation on training data
        cv_results = protocol.cross_validate(train_data, model_func)
        
        # Test on held-out data
        test_results = []
        for _, row in test_data.iterrows():
            forecast = model_func(train_data).forecast(row, years=5)
            actual = protocol._get_actual_revenues(row['launch_id'])
            
            if actual is not None:
                metrics = protocol.metrics.calculate_all_metrics(forecast, actual)
                test_results.append(metrics)
        
        # Aggregate test results
        if test_results:
            test_aggregated = {}
            for metric in ['mape', 'year2_ape', 'peak_ape', 'directional_accuracy']:
                values = [r[metric] for r in test_results if metric in r]
                if values:
                    test_aggregated[f"{metric}_mean"] = np.mean(values)
                    test_aggregated[f"{metric}_ci"] = protocol.bootstrap_confidence_interval(np.array(values))
        
        results['models'][model_name] = {
            'cv_results': cv_results,
            'test_results': test_aggregated if test_results else {}
        }
    
    # Multiple comparison correction
    if len(models) > 1:
        p_values = []
        for model_name in models.keys():
            # Extract p-value from test results (placeholder)
            p_values.append(0.05)  # This would be calculated from actual statistical tests
        
        corrected_p_values = protocol.multiple_comparison_correction(p_values)
        results['multiple_comparison_correction'] = {
            'method': config.correction_method,
            'corrected_p_values': dict(zip(models.keys(), corrected_p_values))
        }
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Statistical Protocol for Pharmaceutical Forecasting")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_drugs = 100
    
    sample_data = pd.DataFrame({
        'launch_id': [f'DRUG_{i:03d}' for i in range(n_drugs)],
        'approval_date': pd.date_range('2015-01-01', periods=n_drugs, freq='M'),
        'therapeutic_area': np.random.choice(['Oncology', 'Immunology', 'Cardiovascular'], n_drugs),
        'revenue_usd': np.random.lognormal(15, 1, n_drugs)  # Log-normal revenue distribution
    })
    
    # Test the protocol
    config = ExperimentConfig(seed=42)
    protocol = StatisticalProtocol(config)
    
    # Test temporal split
    train_data, test_data = protocol.temporal_split(sample_data)
    print(f"Temporal split: {len(train_data)} train, {len(test_data)} test")
    
    # Test power analysis
    power_results = protocol.power_analysis()
    print(f"Power analysis: {power_results}")
    
    print("✅ Statistical protocol ready!")
