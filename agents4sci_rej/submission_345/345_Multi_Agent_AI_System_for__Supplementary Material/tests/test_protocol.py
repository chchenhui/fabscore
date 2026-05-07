"""
Test statistical protocol implementation.
Following Linus principle: Test the contract, not the implementation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stats.protocol import (
    StatisticalProtocol,
    EvaluationMetrics,
    CrossValidation,
    HypothesisTesting,
    AcceptanceGates,
    run_statistical_protocol
)


class TestEvaluationMetrics:
    """Test metric calculations."""
    
    def test_mape(self):
        """Test MAPE calculation."""
        metrics = EvaluationMetrics()
        
        actual = np.array([100, 200, 300])
        forecast = np.array([110, 190, 330])
        
        mape = metrics.mape(actual, forecast)
        
        # Should be (10% + 5% + 10%) / 3 = 8.33%
        assert abs(mape - 8.33) < 0.1
        
        # Test with zeros
        actual_with_zero = np.array([0, 100, 200])
        forecast_with_zero = np.array([10, 110, 190])
        
        mape_zero = metrics.mape(actual_with_zero, forecast_with_zero)
        # Should skip the zero and calculate on [100, 200]
        assert mape_zero < 100  # Reasonable MAPE
    
    def test_year2_ape(self):
        """Test Year 2 APE."""
        metrics = EvaluationMetrics()
        
        ape = metrics.year2_ape(actual_y2=1000, forecast_y2=800)
        assert abs(ape - 20) < 0.01  # 20% error
        
        # Test with zero actual
        ape_inf = metrics.year2_ape(actual_y2=0, forecast_y2=100)
        assert ape_inf == np.inf
    
    def test_prediction_interval_coverage(self):
        """Test PI coverage calculation."""
        metrics = EvaluationMetrics()
        
        actual = np.array([100, 200, 300, 400, 500])
        lower = np.array([80, 150, 250, 350, 450])
        upper = np.array([120, 250, 350, 450, 550])
        
        coverage = metrics.prediction_interval_coverage(actual, lower, upper)
        assert coverage == 1.0  # All within intervals
        
        # Test with some outside
        lower[2] = 310  # 300 now outside [310, 350]
        coverage = metrics.prediction_interval_coverage(actual, lower, upper)
        assert abs(coverage - 0.8) < 0.01  # 4/5 = 80%
    
    def test_directional_accuracy(self):
        """Test direction prediction accuracy."""
        metrics = EvaluationMetrics()
        
        actual = np.array([100, 150, 140, 200])  # Up, Down, Up
        forecast = np.array([100, 120, 130, 180])  # Up, Up, Up
        
        accuracy = metrics.directional_accuracy(actual, forecast)
        # Directions: [Up, Down, Up] vs [Up, Up, Up]
        # Matches: 2/3
        assert abs(accuracy - 0.667) < 0.01


class TestCrossValidation:
    """Test cross-validation methods."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample launch data."""
        n = 100
        dates = pd.date_range('2010-01-01', periods=n, freq='M')
        
        return pd.DataFrame({
            'launch_id': [f'DRUG_{i:04d}' for i in range(n)],
            'approval_date': dates,
            'revenue': np.random.uniform(1e6, 1e9, n)
        })
    
    def test_temporal_split(self, sample_data):
        """Test temporal train/test split."""
        protocol = StatisticalProtocol()
        cv = CrossValidation(protocol)
        
        train, test = cv.split_temporal(sample_data)
        
        # Check sizes
        assert len(train) >= protocol.min_train_samples
        assert len(test) >= protocol.min_test_samples
        assert len(train) + len(test) == len(sample_data)
        
        # Check temporal ordering
        assert train['approval_date'].max() <= test['approval_date'].min()
    
    def test_kfold_validation(self, sample_data):
        """Test k-fold cross-validation."""
        protocol = StatisticalProtocol(n_folds=3)  # Smaller for testing
        cv = CrossValidation(protocol)
        
        def dummy_model(train, val):
            # Simple dummy predictions
            return {'forecast': val['revenue'] * 1.1}
        
        def dummy_metric(val, pred):
            return {'error': np.mean(np.abs(val['revenue'] - pred['forecast']))}
        
        results = cv.kfold_validation(sample_data, dummy_model, dummy_metric)
        
        # Check structure
        assert 'fold_results' in results
        assert 'mean_metrics' in results
        assert len(results['fold_results']) == 3
        
        # Check all folds have results
        for fold_result in results['fold_results']:
            assert 'error' in fold_result
            assert 'fold' in fold_result


class TestHypothesisTesting:
    """Test statistical hypothesis testing."""
    
    def test_compare_models(self):
        """Test model comparison."""
        protocol = StatisticalProtocol()
        hypothesis = HypothesisTesting(protocol)
        
        # Create errors where model A is clearly better
        errors_a = np.random.normal(10, 2, 100)  # Mean 10
        errors_b = np.random.normal(15, 2, 100)  # Mean 15
        
        result = hypothesis.compare_models(errors_a, errors_b, paired=True)
        
        # Should detect significant difference
        assert result['p_value'] < 0.05
        assert result['significant']
        assert result['a_better']  # A has lower error
        assert result['effect_size'] < 0  # Negative because A is better
    
    def test_multiple_comparison_correction(self):
        """Test Holm-Bonferroni correction."""
        protocol = StatisticalProtocol()
        hypothesis = HypothesisTesting(protocol)
        
        # Mix of significant and non-significant p-values
        p_values = [0.001, 0.02, 0.04, 0.06, 0.5]
        
        decisions = hypothesis.multiple_comparison_correction(p_values)
        
        # With Holm correction at alpha=0.05:
        # 0.001 < 0.05/5 = 0.01: True
        # 0.02 < 0.05/4 = 0.0125: False (stops here)
        assert decisions[0] == True
        assert decisions[1] == False  # Would be significant without correction
        assert all(not d for d in decisions[2:])  # Rest are False
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap CI calculation."""
        protocol = StatisticalProtocol(n_bootstrap=100)  # Fewer for speed
        hypothesis = HypothesisTesting(protocol)
        
        # Sample data
        data = np.random.normal(100, 20, 50)
        
        result = hypothesis.bootstrap_confidence_interval(data, np.mean)
        
        # Check structure
        assert 'estimate' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        
        # CI should contain the estimate
        assert result['ci_lower'] <= result['estimate'] <= result['ci_upper']
        
        # CI width should be reasonable
        assert result['ci_width'] > 0
        assert result['ci_width'] < 50  # Not too wide for this data


class TestAcceptanceGates:
    """Test acceptance gate checking."""
    
    def test_gate_g3(self):
        """Test statistical rigor gate."""
        gates = AcceptanceGates()
        protocol = StatisticalProtocol()
        
        # Good case
        passed, msg = gates.check_gate_g3(protocol, train_size=50, test_size=20)
        assert passed
        assert "PASSED" in msg
        
        # Bad case - small training set
        passed, msg = gates.check_gate_g3(protocol, train_size=10, test_size=20)
        assert not passed
        assert "Training set too small" in msg
        
        # Bad case - small test set
        passed, msg = gates.check_gate_g3(protocol, train_size=50, test_size=5)
        assert not passed
        assert "Test set too small" in msg
    
    def test_gate_g4(self):
        """Test beating baselines gate."""
        gates = AcceptanceGates()
        
        # Good case - beats baseline
        results = {
            'y2_ape': 25,  # Better than 40%
            'pi_coverage': 0.80,
            'statistical_comparison': {
                'significant': True
            }
        }
        baselines = {'y2_ape': 40}
        
        passed, msg = gates.check_gate_g4(results, baselines)
        assert passed
        assert "PASSED" in msg
        
        # Bad case - worse than baseline
        results['y2_ape'] = 45
        passed, msg = gates.check_gate_g4(results, baselines)
        assert not passed
        assert "not better" in msg
        
        # Bad case - not significant
        results['y2_ape'] = 25
        results['statistical_comparison']['significant'] = False
        passed, msg = gates.check_gate_g4(results, baselines)
        assert not passed
        assert "Not statistically" in msg
    
    def test_gate_g5(self):
        """Test reproducibility gate."""
        gates = AcceptanceGates()
        
        # Good case
        audit_log = {
            'git_commit': 'abc123',
            'seed': 42,
            'data_version': 'v1.0',
            'model_config': {},
            'total_cost': 10.50,
            'api_calls': 100,
            'git_dirty': False
        }
        
        passed, msg = gates.check_gate_g5(audit_log)
        assert passed
        assert "PASSED" in msg
        
        # Bad case - missing field
        del audit_log['seed']
        passed, msg = gates.check_gate_g5(audit_log)
        assert not passed
        assert "Missing" in msg
        
        # Bad case - dirty git
        audit_log['seed'] = 42
        audit_log['git_dirty'] = True
        passed, msg = gates.check_gate_g5(audit_log)
        assert not passed
        assert "uncommitted" in msg


class TestIntegratedProtocol:
    """Test full protocol execution."""
    
    def test_run_statistical_protocol(self):
        """Test complete protocol run."""
        # Create dummy data
        n = 100
        data = pd.DataFrame({
            'launch_id': [f'DRUG_{i:04d}' for i in range(n)],
            'approval_date': pd.date_range('2010-01-01', periods=n, freq='M'),
            'actual': np.random.uniform(1e6, 1e9, n),
            'actual_y2': np.random.uniform(1e6, 5e8, n),
            'actual_peak': np.random.uniform(5e8, 2e9, n)
        })
        
        # Dummy model function
        def model_fn(train, test):
            return {
                'forecast': test['actual'] * np.random.uniform(0.8, 1.2, len(test)),
                'forecast_y2': test['actual_y2'] * 1.1,
                'forecast_peak': test['actual_peak'] * 0.9,
                'lower': test['actual'] * 0.7,
                'upper': test['actual'] * 1.3,
                'y2_errors': np.random.normal(0, 0.1, 10)
            }
        
        # Dummy baseline
        def baseline_fn(train, test):
            return {
                'forecast': test['actual'] * np.random.uniform(0.7, 1.3, len(test)),
                'forecast_y2': test['actual_y2'] * 1.4  # Worse than model
            }
        
        # Run protocol
        results = run_statistical_protocol(data, model_fn, baseline_fn)
        
        # Check structure
        assert 'protocol' in results
        assert 'cv_results' in results
        assert 'test_results' in results
        assert 'gate_g3' in results
        assert 'gate_g4' in results
        
        # Gate G3 should pass with enough data
        assert results['gate_g3'] == True