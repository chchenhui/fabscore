"""
Unit tests for ensemble forecasting and results evaluation.
Tests the G4 gate criteria and ensemble performance.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from models.ensemble import (
    EnsembleConfig,
    EnsembleForecaster,
    create_ensemble_forecaster,
    ensemble_forecast
)
from results_evaluation import (
    G4GateCriteria,
    ResultsEvaluator,
    run_g4_gate_evaluation
)


class TestEnsembleConfig:
    """Test ensemble configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = EnsembleConfig()
        
        assert config.peak_sales_weight == 0.20
        assert config.year2_naive_weight == 0.15
        assert config.strategy == 'weighted_average'
        assert config.confidence_level == 0.80
        assert config.adaptive_weights is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EnsembleConfig(
            peak_sales_weight=0.30,
            strategy='stacking',
            confidence_level=0.95,
            adaptive_weights=False
        )
        
        assert config.peak_sales_weight == 0.30
        assert config.strategy == 'stacking'
        assert config.confidence_level == 0.95
        assert config.adaptive_weights is False


class TestEnsembleForecaster:
    """Test ensemble forecaster."""
    
    def setup_method(self):
        """Create sample drug data for testing."""
        self.sample_drug = pd.Series({
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
    
    def test_ensemble_forecaster_init(self):
        """Test ensemble forecaster initialization."""
        forecaster = EnsembleForecaster()
        
        assert forecaster is not None
        assert forecaster.config is not None
        assert forecaster.method_weights is not None
        assert len(forecaster.method_weights) == 7  # All 7 methods
    
    def test_forecast_basic(self):
        """Test basic ensemble forecasting."""
        forecaster = EnsembleForecaster()
        result = forecaster.forecast(self.sample_drug, years=5)
        
        # Should have all required keys
        required_keys = ['forecast', 'confidence_intervals', 'individual_forecasts', 
                        'method_contributions', 'ensemble_weights', 'forecast_quality']
        for key in required_keys:
            assert key in result
        
        # Forecast should be array of correct length
        assert len(result['forecast']) == 5
        assert all(val >= 0 for val in result['forecast'])
        
        # Confidence intervals should be present
        ci = result['confidence_intervals']
        assert 'lower' in ci
        assert 'upper' in ci
        assert len(ci['lower']) == 5
        assert len(ci['upper']) == 5
    
    def test_individual_forecasts(self):
        """Test individual forecast generation."""
        forecaster = EnsembleForecaster()
        individual_forecasts = forecaster._generate_individual_forecasts(self.sample_drug, years=5)
        
        # Should have forecasts from all methods
        expected_methods = ['peak_sales', 'year2_naive', 'linear_trend', 'market_share', 
                          'bass_diffusion', 'analog', 'patient_flow']
        for method in expected_methods:
            assert method in individual_forecasts
            assert len(individual_forecasts[method]) == 5
            assert all(val >= 0 for val in individual_forecasts[method])
    
    def test_adaptive_weights(self):
        """Test adaptive weight calculation."""
        forecaster = EnsembleForecaster()
        
        # Test with different therapeutic areas
        oncology_drug = self.sample_drug.copy()
        oncology_drug['therapeutic_area'] = 'Oncology'
        
        rare_disease_drug = self.sample_drug.copy()
        rare_disease_drug['therapeutic_area'] = 'Rare Disease'
        
        oncology_weights = forecaster._get_adaptive_weights(oncology_drug)
        rare_disease_weights = forecaster._get_adaptive_weights(rare_disease_drug)
        
        # Weights should be normalized
        assert abs(sum(oncology_weights.values()) - 1.0) < 0.01
        assert abs(sum(rare_disease_weights.values()) - 1.0) < 0.01
        
        # Oncology should favor analogs and patient flow
        assert oncology_weights['analog'] > forecaster.method_weights['analog']
        assert oncology_weights['patient_flow'] > forecaster.method_weights['patient_flow']
        
        # Rare disease should favor peak sales
        assert rare_disease_weights['peak_sales'] > forecaster.method_weights['peak_sales']
    
    def test_confidence_intervals(self):
        """Test confidence interval generation."""
        forecaster = EnsembleForecaster()
        
        # Create sample forecasts
        forecasts = {
            'method1': np.array([100, 200, 300, 400, 500]),
            'method2': np.array([110, 210, 310, 410, 510]),
            'method3': np.array([90, 190, 290, 390, 490])
        }
        weights = {'method1': 0.4, 'method2': 0.3, 'method3': 0.3}
        ensemble = np.array([100, 200, 300, 400, 500])
        
        # Test bootstrap method
        ci_bootstrap = forecaster._bootstrap_confidence_intervals(forecasts, ensemble)
        assert 'lower' in ci_bootstrap
        assert 'upper' in ci_bootstrap
        assert len(ci_bootstrap['lower']) == 5
        assert len(ci_bootstrap['upper']) == 5
        
        # Test quantile method
        ci_quantile = forecaster._quantile_confidence_intervals(forecasts, weights)
        assert 'lower' in ci_quantile
        assert 'upper' in ci_quantile
        assert len(ci_quantile['lower']) == 5
        assert len(ci_quantile['upper']) == 5
    
    def test_forecast_quality_assessment(self):
        """Test forecast quality assessment."""
        forecaster = EnsembleForecaster()
        
        forecasts = {
            'method1': np.array([100, 200, 300, 400, 500]),
            'method2': np.array([110, 210, 310, 410, 510])
        }
        
        quality = forecaster._assess_forecast_quality(forecasts)
        
        assert 'method1' in quality
        assert 'method2' in quality
        
        for method in quality:
            method_quality = quality[method]
            assert 'mean' in method_quality
            assert 'std' in method_quality
            assert 'trend' in method_quality
            assert 'peak_year' in method_quality
            assert 'peak_value' in method_quality


class TestG4GateCriteria:
    """Test G4 gate criteria."""
    
    def test_default_criteria(self):
        """Test default G4 gate criteria."""
        criteria = G4GateCriteria()
        
        assert criteria.min_launches_beat_baseline == 0.60
        assert criteria.max_median_y2_ape == 30.0
        assert criteria.min_pi_coverage == 0.70
        assert criteria.max_pi_coverage == 0.90
        assert criteria.min_sample_size == 50
        assert len(criteria.baseline_methods) == 5
    
    def test_custom_criteria(self):
        """Test custom G4 gate criteria."""
        criteria = G4GateCriteria(
            min_launches_beat_baseline=0.70,
            max_median_y2_ape=25.0,
            min_sample_size=100
        )
        
        assert criteria.min_launches_beat_baseline == 0.70
        assert criteria.max_median_y2_ape == 25.0
        assert criteria.min_sample_size == 100


class TestResultsEvaluator:
    """Test results evaluator."""
    
    def setup_method(self):
        """Create sample data for testing."""
        self.evaluator = ResultsEvaluator()
        
        # Create sample test data
        self.test_data = pd.DataFrame({
            'launch_id': ['DRUG_001', 'DRUG_002', 'DRUG_003'],
            'therapeutic_area': ['Oncology', 'Immunology', 'Cardiovascular'],
            'eligible_patients_at_launch': [100000, 120000, 80000],
            'list_price_month_usd_launch': [10000, 12000, 8000],
            'net_gtn_pct_launch': [0.8, 0.75, 0.85],
            'competitor_count_at_launch': [2, 3, 1],
            'access_tier_at_launch': ['PA', 'OPEN', 'PA'],
            'clinical_efficacy_proxy': [0.8, 0.7, 0.9],
            'safety_black_box': [False, False, True]
        })
        
        # Create sample revenue data
        self.actual_revenues = pd.DataFrame({
            'launch_id': ['DRUG_001', 'DRUG_001', 'DRUG_001', 'DRUG_001', 'DRUG_001',
                         'DRUG_002', 'DRUG_002', 'DRUG_002', 'DRUG_002', 'DRUG_002',
                         'DRUG_003', 'DRUG_003', 'DRUG_003', 'DRUG_003', 'DRUG_003'],
            'year_since_launch': [0, 1, 2, 3, 4] * 3,
            'revenue_usd': [5000000, 15000000, 25000000, 35000000, 40000000,
                           6000000, 18000000, 30000000, 42000000, 48000000,
                           4000000, 12000000, 20000000, 28000000, 32000000]
        })
    
    def test_results_evaluator_init(self):
        """Test results evaluator initialization."""
        assert self.evaluator is not None
        assert self.evaluator.metrics is not None
        assert self.evaluator.protocol is not None
        assert self.evaluator.ensemble_forecaster is not None
    
    def test_revenues_to_array(self):
        """Test revenue DataFrame to array conversion."""
        drug_revenues = self.actual_revenues[self.actual_revenues['launch_id'] == 'DRUG_001']
        revenue_array = self.evaluator._revenues_to_array(drug_revenues)
        
        assert revenue_array is not None
        assert len(revenue_array) == 5
        assert revenue_array[0] == 5000000  # Year 0
        assert revenue_array[1] == 15000000  # Year 1
        assert revenue_array[4] == 40000000  # Year 4
    
    def test_baseline_forecast_generation(self):
        """Test baseline forecast generation."""
        drug_row = self.test_data.iloc[0]
        
        # Test each baseline method
        methods = ['peak_sales_heuristic', 'year2_naive', 'linear_trend_forecast', 
                  'market_share_evolution', 'simple_bass_forecast']
        
        for method in methods:
            forecast = self.evaluator._generate_baseline_forecast(drug_row, method)
            assert len(forecast) == 5
            assert all(val >= 0 for val in forecast)
    
    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        sample_results = [
            {'mape': 20.0, 'year2_ape': 25.0, 'peak_ape': 30.0},
            {'mape': 25.0, 'year2_ape': 30.0, 'peak_ape': 35.0},
            {'mape': 15.0, 'year2_ape': 20.0, 'peak_ape': 25.0}
        ]
        
        aggregated = self.evaluator._aggregate_metrics(sample_results)
        
        assert 'mape_mean' in aggregated
        assert 'mape_median' in aggregated
        assert 'mape_std' in aggregated
        assert 'mape_count' in aggregated
        
        assert aggregated['mape_mean'] == 20.0  # (20+25+15)/3
        assert aggregated['mape_median'] == 20.0  # Median of [15, 20, 25]
        assert aggregated['mape_count'] == 3
    
    def test_find_best_baseline(self):
        """Test finding best baseline method."""
        baseline_performance = {
            'method1': {'mape_median': 25.0},
            'method2': {'mape_median': 20.0},
            'method3': {'mape_median': 30.0}
        }
        
        best_method = self.evaluator._find_best_baseline(baseline_performance)
        assert best_method == 'method2'  # Lowest MAPE
    
    def test_compare_ensemble_vs_baseline(self):
        """Test ensemble vs baseline comparison."""
        ensemble_results = [
            {'launch_id': 'DRUG_001', 'mape': 20.0},
            {'launch_id': 'DRUG_002', 'mape': 25.0},
            {'launch_id': 'DRUG_003', 'mape': 15.0}
        ]
        
        baseline_results = [
            {'launch_id': 'DRUG_001', 'mape': 25.0},
            {'launch_id': 'DRUG_002', 'mape': 30.0},
            {'launch_id': 'DRUG_003', 'mape': 20.0}
        ]
        
        comparison = self.evaluator._compare_ensemble_vs_baseline(ensemble_results, baseline_results)
        
        assert comparison['total_comparisons'] == 3
        assert comparison['ensemble_wins'] == 3  # Ensemble better in all cases
        assert comparison['baseline_wins'] == 0
        assert comparison['win_rate'] == 1.0
        assert comparison['mean_mape_improvement'] > 0  # Positive improvement
    
    def test_assess_g4_gate_criteria(self):
        """Test G4 gate criteria assessment."""
        # Create mock results
        results = {
            'comparison_metrics': {
                'ensemble_performance': {
                    'year2_ape_median': 25.0,  # Below 30% threshold
                    'pi_coverage_mean': 0.80   # Within 70-90% range
                },
                'ensemble_vs_baseline': {
                    'win_rate': 0.65  # Above 60% threshold
                }
            }
        }
        
        assessment = self.evaluator._assess_g4_gate_criteria(results)
        
        assert assessment['overall_status'] == 'PASS'
        assert assessment['criteria_met']['beat_baseline_60pct'] is True
        assert assessment['criteria_met']['median_y2_ape_30pct'] is True
        assert assessment['criteria_met']['pi_coverage_70_90pct'] is True
    
    def test_assess_g4_gate_criteria_fail(self):
        """Test G4 gate criteria assessment with failing criteria."""
        # Create mock results that fail criteria
        results = {
            'comparison_metrics': {
                'ensemble_performance': {
                    'year2_ape_median': 35.0,  # Above 30% threshold
                    'pi_coverage_mean': 0.60   # Below 70% threshold
                },
                'ensemble_vs_baseline': {
                    'win_rate': 0.50  # Below 60% threshold
                }
            }
        }
        
        assessment = self.evaluator._assess_g4_gate_criteria(results)
        
        assert assessment['overall_status'] == 'FAIL'
        assert assessment['criteria_met']['beat_baseline_60pct'] is False
        assert assessment['criteria_met']['median_y2_ape_30pct'] is False
        assert assessment['criteria_met']['pi_coverage_70_90pct'] is False


class TestIntegration:
    """Test integration between components."""
    
    def test_ensemble_forecast_integration(self):
        """Test ensemble forecast integration."""
        sample_drug = pd.Series({
            'launch_id': 'TEST_001',
            'therapeutic_area': 'Oncology',
            'eligible_patients_at_launch': 100000,
            'list_price_month_usd_launch': 10000,
            'net_gtn_pct_launch': 0.8,
            'competitor_count_at_launch': 2,
            'access_tier_at_launch': 'PA',
            'clinical_efficacy_proxy': 0.8,
            'safety_black_box': False
        })
        
        # Test factory function
        result = ensemble_forecast(sample_drug, years=5)
        
        assert 'forecast' in result
        assert 'confidence_intervals' in result
        assert len(result['forecast']) == 5
    
    def test_g4_gate_evaluation_integration(self):
        """Test G4 gate evaluation integration."""
        # This would typically use real data
        # For now, just test that the function exists and can be called
        assert callable(run_g4_gate_evaluation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
