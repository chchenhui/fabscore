"""
Intelligent ensemble forecasting for pharmaceutical revenue prediction.
Following Linus principle: Combine multiple simple methods rather than one complex method.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baselines import (
    peak_sales_heuristic,
    year2_naive,
    linear_trend_forecast,
    market_share_evolution,
    simple_bass_forecast,
    ensemble_baseline
)
from models.analogs import AnalogForecaster
from models.patient_flow import PatientFlowModel


@dataclass
class EnsembleConfig:
    """Configuration for ensemble forecasting."""
    
    # Method weights (will be learned/optimized)
    peak_sales_weight: float = 0.20
    year2_naive_weight: float = 0.15
    linear_trend_weight: float = 0.15
    market_share_weight: float = 0.15
    bass_diffusion_weight: float = 0.15
    analog_weight: float = 0.10
    patient_flow_weight: float = 0.10
    
    # Ensemble strategies
    strategy: str = 'weighted_average'  # 'weighted_average', 'stacking', 'voting'
    
    # Confidence intervals
    confidence_level: float = 0.80
    uncertainty_method: str = 'bootstrap'  # 'bootstrap', 'quantile', 'gaussian'
    
    # Adaptive weighting
    adaptive_weights: bool = True
    weight_learning_rate: float = 0.1
    
    # Minimum methods required
    min_methods: int = 3


class EnsembleForecaster:
    """
    Intelligent ensemble forecasting system.
    Combines multiple forecasting methods with adaptive weighting.
    """
    
    def __init__(self, config: EnsembleConfig = None, data_dir: Path = None):
        self.config = config or EnsembleConfig()
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data_proc"
        
        # Initialize individual forecasters
        self.analog_forecaster = AnalogForecaster(self.data_dir)
        self.patient_flow_model = PatientFlowModel()
        
        # Performance tracking for adaptive weighting
        self.method_performance = {}
        self.method_weights = {
            'peak_sales': self.config.peak_sales_weight,
            'year2_naive': self.config.year2_naive_weight,
            'linear_trend': self.config.linear_trend_weight,
            'market_share': self.config.market_share_weight,
            'bass_diffusion': self.config.bass_diffusion_weight,
            'analog': self.config.analog_weight,
            'patient_flow': self.config.patient_flow_weight
        }
        
        # Validation results for weight optimization
        self.validation_results = []
    
    def forecast(self, drug_row: pd.Series, years: int = 5) -> Dict[str, Any]:
        """
        Generate ensemble forecast for a drug.
        
        Args:
            drug_row: Drug characteristics
            years: Forecast horizon
        
        Returns:
            Dictionary with forecast, confidence intervals, and method breakdown
        """
        # Generate individual forecasts
        individual_forecasts = self._generate_individual_forecasts(drug_row, years)
        
        # Apply ensemble strategy
        if self.config.strategy == 'weighted_average':
            ensemble_forecast, confidence_intervals = self._weighted_average_ensemble(
                individual_forecasts, drug_row
            )
        elif self.config.strategy == 'stacking':
            ensemble_forecast, confidence_intervals = self._stacking_ensemble(
                individual_forecasts, drug_row
            )
        elif self.config.strategy == 'voting':
            ensemble_forecast, confidence_intervals = self._voting_ensemble(
                individual_forecasts, drug_row
            )
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.config.strategy}")
        
        # Calculate method contributions
        method_contributions = self._calculate_method_contributions(
            individual_forecasts, ensemble_forecast
        )
        
        return {
            'forecast': ensemble_forecast,
            'confidence_intervals': confidence_intervals,
            'individual_forecasts': individual_forecasts,
            'method_contributions': method_contributions,
            'ensemble_weights': self.method_weights.copy(),
            'forecast_quality': self._assess_forecast_quality(individual_forecasts)
        }
    
    def _generate_individual_forecasts(self, drug_row: pd.Series, years: int) -> Dict[str, np.ndarray]:
        """Generate forecasts from all individual methods."""
        forecasts = {}
        
        try:
            # Peak sales heuristic
            peak = peak_sales_heuristic(drug_row)
            forecasts['peak_sales'] = linear_trend_forecast(drug_row, years)
            
            # Year 2 naive
            forecasts['year2_naive'] = self._year2_anchored_forecast(drug_row, years)
            
            # Linear trend
            forecasts['linear_trend'] = linear_trend_forecast(drug_row, years)
            
            # Market share evolution
            share_curve = market_share_evolution(drug_row, years)
            market_size = drug_row['eligible_patients_at_launch']
            annual_price = drug_row['list_price_month_usd_launch'] * 12
            gtn = drug_row['net_gtn_pct_launch']
            compliance = 0.70
            forecasts['market_share'] = share_curve * market_size * annual_price * gtn * compliance
            
            # Bass diffusion
            forecasts['bass_diffusion'] = simple_bass_forecast(drug_row, years)
            
            # Analog forecasting
            try:
                analogs = self.analog_forecaster.find_best_analogs(drug_row, n_analogs=3)
                if analogs:
                    analog_forecast = self.analog_forecaster.forecast_from_analogs(drug_row, analogs, years)
                    forecasts['analog'] = analog_forecast
                else:
                    # Fallback to linear trend if no analogs
                    forecasts['analog'] = linear_trend_forecast(drug_row, years)
            except Exception:
                forecasts['analog'] = linear_trend_forecast(drug_row, years)
            
            # Patient flow
            try:
                patient_flow_forecast = self.patient_flow_model.forecast(drug_row, years)
                forecasts['patient_flow'] = patient_flow_forecast
            except Exception:
                forecasts['patient_flow'] = linear_trend_forecast(drug_row, years)
            
        except Exception as e:
            # Fallback to simple linear trend if any method fails
            fallback_forecast = linear_trend_forecast(drug_row, years)
            for method in ['peak_sales', 'year2_naive', 'linear_trend', 'market_share', 
                          'bass_diffusion', 'analog', 'patient_flow']:
                if method not in forecasts:
                    forecasts[method] = fallback_forecast
        
        return forecasts
    
    def _year2_anchored_forecast(self, drug_row: pd.Series, years: int) -> np.ndarray:
        """Create forecast anchored to year 2 naive estimate."""
        y2_estimate = year2_naive(drug_row)
        peak = peak_sales_heuristic(drug_row)
        
        forecast = np.zeros(years)
        
        for i in range(years):
            if i == 0:
                forecast[i] = peak * 0.05  # Launch year
            elif i == 1:
                forecast[i] = y2_estimate  # Year 2 anchor
            elif i < 4:
                # Interpolate to peak
                forecast[i] = y2_estimate + (peak - y2_estimate) * (i - 1) / 2
            else:
                forecast[i] = peak  # Peak and beyond
        
        return forecast
    
    def _weighted_average_ensemble(self, forecasts: Dict[str, np.ndarray], 
                                 drug_row: pd.Series) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Weighted average ensemble with adaptive weights."""
        
        # Get adaptive weights based on drug characteristics
        weights = self._get_adaptive_weights(drug_row)
        
        # Ensure all forecasts have same length
        min_length = min(len(f) for f in forecasts.values())
        aligned_forecasts = {k: v[:min_length] for k, v in forecasts.items()}
        
        # Weighted average
        ensemble = np.zeros(min_length)
        for method, forecast in aligned_forecasts.items():
            if method in weights:
                ensemble += weights[method] * forecast
        
        # Generate confidence intervals
        confidence_intervals = self._generate_confidence_intervals(
            aligned_forecasts, ensemble, weights
        )
        
        return ensemble, confidence_intervals
    
    def _get_adaptive_weights(self, drug_row: pd.Series) -> Dict[str, float]:
        """Get adaptive weights based on drug characteristics."""
        weights = self.method_weights.copy()
        
        if not self.config.adaptive_weights:
            return weights
        
        # Adjust weights based on drug characteristics
        ta = drug_row.get('therapeutic_area', '')
        competitors = drug_row.get('competitor_count_at_launch', 3)
        access = drug_row.get('access_tier_at_launch', 'PA')
        
        # Oncology: favor analogs and patient flow
        if ta == 'Oncology':
            weights['analog'] *= 1.3
            weights['patient_flow'] *= 1.2
            weights['bass_diffusion'] *= 0.8
        
        # Rare disease: favor peak sales and market share
        elif ta == 'Rare Disease':
            weights['peak_sales'] *= 1.4
            weights['market_share'] *= 1.2
            weights['analog'] *= 0.7
        
        # High competition: favor year 2 naive and linear trend
        if competitors > 5:
            weights['year2_naive'] *= 1.2
            weights['linear_trend'] *= 1.1
            weights['peak_sales'] *= 0.9
        
        # Open access: favor faster uptake methods
        if access == 'OPEN':
            weights['bass_diffusion'] *= 1.2
            weights['market_share'] *= 1.1
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _generate_confidence_intervals(self, forecasts: Dict[str, np.ndarray], 
                                     ensemble: np.ndarray, weights: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Generate confidence intervals for ensemble forecast."""
        
        if self.config.uncertainty_method == 'bootstrap':
            return self._bootstrap_confidence_intervals(forecasts, ensemble)
        elif self.config.uncertainty_method == 'quantile':
            return self._quantile_confidence_intervals(forecasts, weights)
        elif self.config.uncertainty_method == 'gaussian':
            return self._gaussian_confidence_intervals(forecasts, weights)
        else:
            # Default to quantile method
            return self._quantile_confidence_intervals(forecasts, weights)
    
    def _bootstrap_confidence_intervals(self, forecasts: Dict[str, np.ndarray], 
                                      ensemble: np.ndarray) -> Dict[str, np.ndarray]:
        """Bootstrap confidence intervals."""
        n_bootstrap = 1000
        n_years = len(ensemble)
        
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            # Sample with replacement from individual forecasts
            sample_forecast = np.zeros(n_years)
            for year in range(n_years):
                year_values = [f[year] for f in forecasts.values()]
                sample_forecast[year] = np.random.choice(year_values)
            bootstrap_samples.append(sample_forecast)
        
        # Calculate percentiles
        alpha = 1 - self.config.confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        lower = np.percentile(bootstrap_samples, lower_percentile, axis=0)
        upper = np.percentile(bootstrap_samples, upper_percentile, axis=0)
        
        return {
            'lower': lower,
            'upper': upper,
            'method': 'bootstrap'
        }
    
    def _quantile_confidence_intervals(self, forecasts: Dict[str, np.ndarray], 
                                     weights: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Quantile-based confidence intervals."""
        n_years = len(list(forecasts.values())[0])
        
        lower = np.zeros(n_years)
        upper = np.zeros(n_years)
        
        for year in range(n_years):
            year_values = [f[year] for f in forecasts.values()]
            year_weights = [weights.get(method, 0) for method in forecasts.keys()]
            
            # Weighted quantiles
            alpha = 1 - self.config.confidence_level
            lower[year] = np.percentile(year_values, 100 * alpha / 2)
            upper[year] = np.percentile(year_values, 100 * (1 - alpha / 2))
        
        return {
            'lower': lower,
            'upper': upper,
            'method': 'quantile'
        }
    
    def _gaussian_confidence_intervals(self, forecasts: Dict[str, np.ndarray], 
                                     weights: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Gaussian approximation confidence intervals."""
        n_years = len(list(forecasts.values())[0])
        
        lower = np.zeros(n_years)
        upper = np.zeros(n_years)
        
        for year in range(n_years):
            year_values = np.array([f[year] for f in forecasts.values()])
            year_weights = np.array([weights.get(method, 0) for method in forecasts.keys()])
            
            # Weighted mean and variance
            weighted_mean = np.average(year_values, weights=year_weights)
            weighted_var = np.average((year_values - weighted_mean) ** 2, weights=year_weights)
            
            # Gaussian confidence interval
            z_score = 1.96  # 95% confidence
            margin = z_score * np.sqrt(weighted_var)
            
            lower[year] = weighted_mean - margin
            upper[year] = weighted_mean + margin
        
        return {
            'lower': lower,
            'upper': upper,
            'method': 'gaussian'
        }
    
    def _calculate_method_contributions(self, forecasts: Dict[str, np.ndarray], 
                                      ensemble: np.ndarray) -> Dict[str, float]:
        """Calculate contribution of each method to ensemble."""
        contributions = {}
        
        for method, forecast in forecasts.items():
            # Correlation with ensemble
            correlation = np.corrcoef(forecast, ensemble)[0, 1]
            contributions[method] = correlation if not np.isnan(correlation) else 0.0
        
        return contributions
    
    def _assess_forecast_quality(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Assess quality of individual forecasts."""
        quality = {}
        
        for method, forecast in forecasts.items():
            # Calculate forecast characteristics
            quality[method] = {
                'mean': np.mean(forecast),
                'std': np.std(forecast),
                'trend': np.polyfit(range(len(forecast)), forecast, 1)[0],
                'peak_year': np.argmax(forecast),
                'peak_value': np.max(forecast)
            }
        
        return quality
    
    def update_weights_from_validation(self, validation_results: List[Dict[str, Any]]):
        """Update method weights based on validation performance."""
        if not validation_results:
            return
        
        # Calculate performance scores for each method
        method_scores = {}
        for method in self.method_weights.keys():
            scores = []
            for result in validation_results:
                if method in result.get('method_metrics', {}):
                    # Use MAPE as performance metric (lower is better)
                    mape = result['method_metrics'][method].get('mape', 1.0)
                    scores.append(1.0 / (1.0 + mape))  # Convert to higher-is-better
            
            if scores:
                method_scores[method] = np.mean(scores)
            else:
                method_scores[method] = 0.5  # Default score
        
        # Update weights using learning rate
        for method, score in method_scores.items():
            old_weight = self.method_weights[method]
            new_weight = old_weight + self.config.weight_learning_rate * (score - 0.5)
            self.method_weights[method] = max(0.01, min(0.5, new_weight))  # Clamp between 0.01 and 0.5
        
        # Renormalize weights
        total_weight = sum(self.method_weights.values())
        self.method_weights = {k: v / total_weight for k, v in self.method_weights.items()}


def create_ensemble_forecaster(config: EnsembleConfig = None) -> EnsembleForecaster:
    """Factory function to create ensemble forecaster."""
    return EnsembleForecaster(config)


def ensemble_forecast(drug_row: pd.Series, years: int = 5, 
                     config: EnsembleConfig = None) -> Dict[str, Any]:
    """
    Convenience function for ensemble forecasting.
    
    Args:
        drug_row: Drug characteristics
        years: Forecast horizon
        config: Ensemble configuration
    
    Returns:
        Ensemble forecast results
    """
    forecaster = create_ensemble_forecaster(config)
    return forecaster.forecast(drug_row, years)


if __name__ == "__main__":
    # Example usage
    print("Ensemble Forecasting System")
    print("=" * 40)
    
    # Create sample drug data
    sample_drug = pd.Series({
        'launch_id': 'TEST_001',
        'drug_name': 'TestDrug',
        'company': 'TestCorp',
        'therapeutic_area': 'Oncology',
        'eligible_patients_at_launch': 100000,
        'list_price_month_usd_launch': 10000,
        'net_gtn_pct_launch': 0.8,
        'competitor_count_at_launch': 2,
        'access_tier_at_launch': 'PA',
        'clinical_efficacy_proxy': 0.8,
        'safety_black_box': False
    })
    
    # Create ensemble forecaster
    config = EnsembleConfig(strategy='weighted_average', adaptive_weights=True)
    forecaster = create_ensemble_forecaster(config)
    
    # Generate forecast
    result = forecaster.forecast(sample_drug, years=5)
    
    print(f"Ensemble Forecast: {result['forecast']}")
    print(f"Confidence Intervals: {result['confidence_intervals']}")
    print(f"Method Weights: {result['ensemble_weights']}")
    print("✅ Ensemble forecasting ready!")
