"""
Unit tests for statistical framework and power analysis.
Tests the core statistical methods for pharmaceutical forecasting.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from experiment_protocol import (
    ExperimentConfig,
    EvaluationMetrics,
    StatisticalProtocol,
    run_experiment
)
from power import (
    PowerAnalysisConfig,
    PowerAnalyzer,
    run_power_analysis
)


class TestExperimentConfig:
    """Test experiment configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ExperimentConfig()
        
        assert config.train_pct == 0.7
        assert config.test_pct == 0.3
        assert config.cv_folds == 5
        assert config.alpha == 0.05
        assert config.power == 0.8
        assert config.seed == 42
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ExperimentConfig(
            train_pct=0.8,
            test_pct=0.2,
            cv_folds=10,
            alpha=0.01,
            power=0.9,
            seed=123
        )
        
        assert config.train_pct == 0.8
        assert config.test_pct == 0.2
        assert config.cv_folds == 10
        assert config.alpha == 0.01
        assert config.power == 0.9
        assert config.seed == 123


class TestEvaluationMetrics:
    """Test evaluation metrics."""
    
    def setup_method(self):
        """Create sample data for testing."""
        self.metrics = EvaluationMetrics()
        
        # Sample predictions and actuals
        self.predicted = np.array([100, 200, 300, 400, 500])
        self.actual = np.array([110, 190, 320, 380, 520])
        
        # With prediction intervals
        self.lower = np.array([90, 180, 280, 360, 480])
        self.upper = np.array([110, 220, 320, 440, 520])
    
    def test_mape(self):
        """Test MAPE calculation."""
        mape = self.metrics.mape(self.predicted, self.actual)
        
        # Should be positive
        assert mape > 0
        
        # Should be reasonable (around 5-10% for this data)
        assert 0 < mape < 50
    
    def test_mape_zero_actual(self):
        """Test MAPE with zero actual values."""
        predicted = np.array([100, 200, 300])
        actual = np.array([110, 0, 320])
        
        mape = self.metrics.mape(predicted, actual)
        
        # Should handle zero values gracefully
        assert mape > 0
        assert not np.isinf(mape)
    
    def test_year2_ape(self):
        """Test Year 2 APE calculation."""
        y2_ape = self.metrics.year2_ape(self.predicted, self.actual)
        
        # Should be positive
        assert y2_ape > 0
        
        # Should be reasonable
        assert 0 < y2_ape < 100
    
    def test_peak_ape(self):
        """Test Peak APE calculation."""
        peak_ape = self.metrics.peak_ape(self.predicted, self.actual)
        
        # Should be positive
        assert peak_ape > 0
        
        # Should be reasonable
        assert 0 < peak_ape < 100
    
    def test_directional_accuracy(self):
        """Test directional accuracy."""
        accuracy = self.metrics.directional_accuracy(self.predicted, self.actual)
        
        # Should be between 0 and 1
        assert 0 <= accuracy <= 1
    
    def test_prediction_interval_coverage(self):
        """Test prediction interval coverage."""
        coverage = self.metrics.prediction_interval_coverage(
            self.predicted, self.actual, self.lower, self.upper
        )
        
        # Should be between 0 and 1
        assert 0 <= coverage <= 1
    
    def test_calculate_all_metrics(self):
        """Test calculation of all metrics."""
        all_metrics = self.metrics.calculate_all_metrics(
            self.predicted, self.actual, self.lower, self.upper
        )
        
        # Should have all expected metrics
        expected_metrics = ['mape', 'year2_ape', 'peak_ape', 'directional_accuracy', 'rmse', 'mae', 'pi_coverage']
        for metric in expected_metrics:
            assert metric in all_metrics
            assert isinstance(all_metrics[metric], (int, float))


class TestStatisticalProtocol:
    """Test statistical protocol."""
    
    def setup_method(self):
        """Create sample data for testing."""
        self.config = ExperimentConfig(seed=42)
        self.protocol = StatisticalProtocol(self.config)
        
        # Create sample data
        np.random.seed(42)
        n_drugs = 100
        
        self.sample_data = pd.DataFrame({
            'launch_id': [f'DRUG_{i:03d}' for i in range(n_drugs)],
            'approval_date': pd.date_range('2015-01-01', periods=n_drugs, freq='ME'),
            'therapeutic_area': np.random.choice(['Oncology', 'Immunology', 'Cardiovascular'], n_drugs),
            'revenue_usd': np.random.lognormal(15, 1, n_drugs)
        })
    
    def test_temporal_split(self):
        """Test temporal train/test split."""
        train_data, test_data = self.protocol.temporal_split(self.sample_data)
        
        # Should split by year
        assert len(train_data) > 0
        assert len(test_data) > 0
        assert len(train_data) + len(test_data) == len(self.sample_data)
        
        # Train data should be earlier
        max_train_year = train_data['approval_date'].dt.year.max()
        min_test_year = test_data['approval_date'].dt.year.min()
        assert max_train_year <= self.config.temporal_split_year
        assert min_test_year > self.config.temporal_split_year
    
    def test_stratified_split(self):
        """Test stratified train/test split."""
        train_data, test_data = self.protocol.stratified_split(self.sample_data)
        
        # Should split proportionally
        assert len(train_data) > 0
        assert len(test_data) > 0
        assert len(train_data) + len(test_data) == len(self.sample_data)
        
        # Should maintain therapeutic area proportions
        train_ta_props = train_data['therapeutic_area'].value_counts(normalize=True)
        test_ta_props = test_data['therapeutic_area'].value_counts(normalize=True)
        
        for ta in train_ta_props.index:
            if ta in test_ta_props.index:
                # Proportions should be similar (within 10%)
                assert abs(train_ta_props[ta] - test_ta_props[ta]) < 0.1
    
    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        
        # Test Holm-Bonferroni
        corrected_holm = self.protocol.multiple_comparison_correction(p_values, 'holm')
        assert len(corrected_holm) == len(p_values)
        assert all(p >= 0 and p <= 1 for p in corrected_holm)
        
        # Test Bonferroni
        corrected_bonf = self.protocol.multiple_comparison_correction(p_values, 'bonferroni')
        assert len(corrected_bonf) == len(p_values)
        assert all(p >= 0 and p <= 1 for p in corrected_bonf)
        
        # Bonferroni should be more conservative than Holm
        assert all(corrected_bonf[i] >= corrected_holm[i] for i in range(len(p_values)))
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval."""
        data = np.random.normal(100, 10, 100)
        
        lower, upper = self.protocol.bootstrap_confidence_interval(data)
        
        # Should be reasonable bounds
        assert lower < upper
        assert lower < np.mean(data) < upper
    
    def test_power_analysis(self):
        """Test power analysis."""
        # Test sample size calculation
        power_results = self.protocol.power_analysis(effect_size=0.5, power=0.8)
        
        assert 'required_sample_size' in power_results
        assert power_results['required_sample_size'] > 0
        
        # Test power calculation
        power_results = self.protocol.power_analysis(effect_size=0.5, sample_size=100)
        
        assert 'achieved_power' in power_results
        assert 0 <= power_results['achieved_power'] <= 1


class TestPowerAnalysisConfig:
    """Test power analysis configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PowerAnalysisConfig()
        
        assert config.mape_improvement == 0.10
        assert config.alpha == 0.05
        assert config.power == 0.80
        assert config.min_sample_size == 50
        assert config.max_sample_size == 200
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PowerAnalysisConfig(
            mape_improvement=0.15,
            alpha=0.01,
            power=0.90,
            min_sample_size=100
        )
        
        assert config.mape_improvement == 0.15
        assert config.alpha == 0.01
        assert config.power == 0.90
        assert config.min_sample_size == 100


class TestPowerAnalyzer:
    """Test power analyzer."""
    
    def setup_method(self):
        """Create power analyzer for testing."""
        self.config = PowerAnalysisConfig()
        self.analyzer = PowerAnalyzer(self.config)
    
    def test_calculate_required_sample_size(self):
        """Test required sample size calculation."""
        n_required = self.analyzer.calculate_required_sample_size(effect_size=0.5)
        
        assert n_required > 0
        assert isinstance(n_required, int)
    
    def test_calculate_achieved_power(self):
        """Test achieved power calculation."""
        power = self.analyzer.calculate_achieved_power(effect_size=0.5, sample_size=100)
        
        assert 0 <= power <= 1
    
    def test_mape_effect_size_to_cohens_d(self):
        """Test MAPE to Cohen's d conversion."""
        cohens_d = self.analyzer.mape_effect_size_to_cohens_d(0.10, 0.30)
        
        assert cohens_d > 0
        assert isinstance(cohens_d, float)
    
    def test_analyze_mape_power(self):
        """Test MAPE power analysis."""
        results = self.analyzer.analyze_mape_power()
        
        # Should have all required fields
        required_fields = ['metric', 'baseline_mape', 'improvement', 'effect_size', 
                          'required_sample_size', 'sample_sizes', 'achieved_powers']
        for field in required_fields:
            assert field in results
        
        # Should have reasonable values
        assert results['metric'] == 'MAPE'
        assert results['baseline_mape'] == 0.30
        assert results['improvement'] == 0.10
        assert results['required_sample_size'] > 0
        assert len(results['sample_sizes']) == len(results['achieved_powers'])
    
    def test_analyze_year2_ape_power(self):
        """Test Year 2 APE power analysis."""
        results = self.analyzer.analyze_year2_ape_power()
        
        assert results['metric'] == 'Year2_APE'
        assert results['baseline_year2_ape'] == 0.35
        assert results['improvement'] == 0.15
        assert results['required_sample_size'] > 0
    
    def test_analyze_peak_ape_power(self):
        """Test Peak APE power analysis."""
        results = self.analyzer.analyze_peak_ape_power()
        
        assert results['metric'] == 'Peak_APE'
        assert results['baseline_peak_ape'] == 0.40
        assert results['improvement'] == 0.12
        assert results['required_sample_size'] > 0
    
    def test_analyze_multiple_comparisons_power(self):
        """Test multiple comparisons power analysis."""
        results = self.analyzer.analyze_multiple_comparisons_power()
        
        required_fields = ['correction_method', 'n_comparisons', 'original_alpha', 
                          'adjusted_alpha', 'effect_size', 'required_sample_size']
        for field in required_fields:
            assert field in results
        
        # Adjusted alpha should be smaller than original
        assert results['adjusted_alpha'] < results['original_alpha']
        assert results['required_sample_size'] > 0
    
    def test_comprehensive_power_analysis(self):
        """Test comprehensive power analysis."""
        results = self.analyzer.comprehensive_power_analysis()
        
        # Should have all analysis types
        assert 'config' in results
        assert 'mape_analysis' in results
        assert 'year2_ape_analysis' in results
        assert 'peak_ape_analysis' in results
        assert 'multiple_comparisons_analysis' in results
        assert 'recommendations' in results
        
        # Recommendations should be reasonable
        recs = results['recommendations']
        assert recs['minimum_sample_size'] > 0
        assert recs['recommended_sample_size'] >= recs['minimum_sample_size']
        assert isinstance(recs['power_adequate'], bool)
    
    def test_generate_power_report(self):
        """Test power report generation."""
        results = self.analyzer.comprehensive_power_analysis()
        report = self.analyzer.generate_power_report(results)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "POWER ANALYSIS REPORT" in report
        assert "CONFIGURATION:" in report
        assert "RECOMMENDATIONS:" in report


class TestIntegration:
    """Test integration between components."""
    
    def test_experiment_protocol_integration(self):
        """Test integration of experiment protocol components."""
        config = ExperimentConfig(seed=42)
        protocol = StatisticalProtocol(config)
        metrics = EvaluationMetrics()
        
        # Should work together
        assert protocol.config == config
        assert type(protocol.metrics) == type(metrics)
    
    def test_power_analysis_integration(self):
        """Test integration of power analysis components."""
        config = PowerAnalysisConfig()
        analyzer = PowerAnalyzer(config)
        
        # Should work together
        assert analyzer.config == config
        
        # Should produce consistent results
        results = analyzer.comprehensive_power_analysis()
        assert results['config'] == config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
