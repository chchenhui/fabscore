"""
Test suite for Bass diffusion model.
Tests edge cases, mass conservation, and monotonic behavior.
"""
import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.bass import (
    bass_adopters, bass_cumulative, bass_peak_time,
    estimate_bass_parameters, validate_bass_output
)

class TestBassAdopters:
    """Test bass_adopters function."""
    
    def test_basic_functionality(self):
        """Test basic Bass model functionality."""
        T, m, p, q = 40, 1000000, 0.03, 0.40
        adopters = bass_adopters(T, m, p, q)
        
        assert len(adopters) == T
        assert all(a >= 0 for a in adopters)  # Non-negative
        assert np.sum(adopters) <= m  # Mass conservation
    
    def test_mass_conservation(self):
        """Test that total adoption never exceeds market size."""
        test_cases = [
            (20, 100000, 0.01, 0.2),
            (40, 1000000, 0.05, 0.6),
            (100, 50000, 0.02, 0.3)
        ]
        
        for T, m, p, q in test_cases:
            adopters = bass_adopters(T, m, p, q)
            total_adoption = np.sum(adopters)
            assert total_adoption <= m + 1e-10, f"Mass not conserved: {total_adoption} > {m}"
    
    def test_monotonic_cumulative(self):
        """Test that cumulative adoption is monotonically increasing."""
        T, m, p, q = 40, 500000, 0.025, 0.35
        adopters = bass_adopters(T, m, p, q)
        cumulative = np.cumsum(adopters)
        
        # Check monotonic (weakly increasing)
        diffs = np.diff(cumulative)
        assert all(d >= -1e-10 for d in diffs), "Cumulative adoption not monotonic"
    
    def test_extreme_parameters(self):
        """Test behavior with extreme parameter values."""
        T, m = 20, 100000
        
        # Very small p, q
        adopters_small = bass_adopters(T, m, 0.001, 0.01)
        assert np.sum(adopters_small) < m * 0.1  # Should have low adoption
        
        # Large p (but valid)
        adopters_large_p = bass_adopters(T, m, 0.1, 0.2)
        peak_early = np.argmax(adopters_large_p)
        assert peak_early < T // 4  # Peak should be early with high p
        
        # Large q
        adopters_large_q = bass_adopters(T, m, 0.01, 0.8)
        peak_late = np.argmax(adopters_large_q)
        assert peak_late > T // 4  # Peak should be later with high q
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        T, m = 10, 100000
        
        # Invalid T
        with pytest.raises(ValueError):
            bass_adopters(0, m, 0.03, 0.4)
        
        # Invalid market size
        with pytest.raises(ValueError):
            bass_adopters(T, -1000, 0.03, 0.4)
        
        # Invalid p
        with pytest.raises(ValueError):
            bass_adopters(T, m, 0, 0.4)  # p = 0
        with pytest.raises(ValueError):
            bass_adopters(T, m, 1.1, 0.4)  # p > 1
        
        # Invalid q
        with pytest.raises(ValueError):
            bass_adopters(T, m, 0.03, -0.1)  # q < 0
        with pytest.raises(ValueError):
            bass_adopters(T, m, 0.03, 1.2)  # q > 1
    
    def test_saturation_behavior(self):
        """Test that model handles market saturation correctly."""
        # Small market, high adoption parameters
        T, m, p, q = 50, 1000, 0.1, 0.9
        adopters = bass_adopters(T, m, p, q)
        cumulative = np.cumsum(adopters)
        
        # Should reach saturation
        assert cumulative[-1] >= m * 0.95  # Near full saturation
        
        # Adoption should go to zero after saturation
        saturation_point = np.where(cumulative >= m * 0.99)[0]
        if len(saturation_point) > 0:
            sat_idx = saturation_point[0]
            if sat_idx < len(adopters) - 5:  # If not at end
                late_adoption = adopters[sat_idx+1:]
                assert all(a < m * 0.01 for a in late_adoption)  # Very low adoption after saturation

class TestBassCumulative:
    """Test bass_cumulative function."""
    
    def test_cumulative_consistency(self):
        """Test that cumulative matches sum of adopters."""
        T, m, p, q = 30, 200000, 0.04, 0.45
        adopters = bass_adopters(T, m, p, q)
        cumulative = bass_cumulative(T, m, p, q)
        expected_cumulative = np.cumsum(adopters)
        
        np.testing.assert_array_almost_equal(cumulative, expected_cumulative)

class TestBassPeakTime:
    """Test bass_peak_time function."""
    
    def test_peak_time_calculation(self):
        """Test theoretical peak time calculation."""
        # Test case where p < q (typical)
        p, q = 0.02, 0.3
        peak_time = bass_peak_time(1000000, p, q)
        
        assert peak_time > 0
        assert not np.isnan(peak_time)
        
        # Peak time should be later when p is smaller relative to q
        p2, q2 = 0.01, 0.3  # Smaller p
        peak_time2 = bass_peak_time(1000000, p2, q2)
        assert peak_time2 > peak_time
    
    def test_peak_time_edge_cases(self):
        """Test peak time with edge cases."""
        # p = q case
        peak_time = bass_peak_time(100000, 0.05, 0.05)
        assert np.isclose(peak_time, 0.0)  # ln(1) = 0
        
        # Very small p
        peak_time = bass_peak_time(100000, 0.001, 0.5)
        assert peak_time > 10  # Should be large
        
        # Invalid parameters
        peak_time = bass_peak_time(100000, 0, 0.3)
        assert np.isinf(peak_time)

class TestEstimateBassParameters:
    """Test parameter estimation function."""
    
    def test_parameter_estimation(self):
        """Test basic parameter estimation."""
        # Generate synthetic data
        T, m_true, p_true, q_true = 20, 100000, 0.03, 0.4
        true_adopters = bass_adopters(T, m_true, p_true, q_true)
        
        # Estimate parameters
        m_est, p_est, q_est = estimate_bass_parameters(true_adopters, m_true)
        
        # Should be in reasonable range
        assert 0.001 <= p_est <= 0.1
        assert 0.1 <= q_est <= 0.8
        assert m_est > 0
    
    def test_estimation_without_market_size(self):
        """Test estimation when market size is unknown."""
        adopters = np.array([100, 300, 500, 600, 400, 300, 200])
        m_est, p_est, q_est = estimate_bass_parameters(adopters)
        
        assert m_est > np.sum(adopters)  # Should estimate larger market
        assert 0.001 <= p_est <= 0.1
        assert 0.1 <= q_est <= 0.8
    
    def test_estimation_edge_cases(self):
        """Test estimation with edge cases."""
        # Too few data points
        with pytest.raises(ValueError):
            estimate_bass_parameters(np.array([100, 200]))
        
        # Flat adoption (no clear peak)
        flat_adopters = np.array([100, 100, 100, 100, 100])
        m_est, p_est, q_est = estimate_bass_parameters(flat_adopters)
        # Should still return valid parameters
        assert all(param > 0 for param in [m_est, p_est, q_est])

class TestValidateBassOutput:
    """Test validation function."""
    
    def test_validation_success(self):
        """Test validation with valid output."""
        T, m, p, q = 25, 150000, 0.025, 0.35
        adopters = bass_adopters(T, m, p, q)
        
        # Should pass validation
        assert validate_bass_output(adopters, m)
    
    def test_validation_failures(self):
        """Test validation with invalid outputs."""
        m = 100000
        
        # Negative adopters
        bad_adopters1 = np.array([100, -50, 200, 150])
        with pytest.raises(AssertionError):
            validate_bass_output(bad_adopters1, m)
        
        # Exceeds market size
        bad_adopters2 = np.array([50000, 60000, 40000])  # Sum > m
        with pytest.raises(AssertionError):
            validate_bass_output(bad_adopters2, m)
        
        # Non-monotonic cumulative
        bad_adopters3 = np.array([1000, 2000, -500, 1500])  # Cumulative decreases
        with pytest.raises(AssertionError):
            validate_bass_output(bad_adopters3, m)

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_workflow(self):
        """Test complete Bass model workflow."""
        # Parameters
        T, m, p, q = 40, 1200000, 0.03, 0.40
        
        # Generate adoption curve
        adopters = bass_adopters(T, m, p, q)
        cumulative = bass_cumulative(T, m, p, q)
        
        # Validate
        validate_bass_output(adopters, m)
        
        # Check peak timing
        peak_quarter = np.argmax(adopters) + 1
        theoretical_peak = bass_peak_time(m, p, q)
        
        # Peak should be roughly in the right timeframe
        assert abs(peak_quarter - theoretical_peak) < T * 0.3  # Within 30% of timeframe
        
        # Estimate parameters from output
        m_est, p_est, q_est = estimate_bass_parameters(adopters, m)
        
        # Should be in reasonable range of original parameters
        # Note: Simple estimation method has limitations for extreme parameters
        assert abs(p_est - p) < 0.05  # Within 5 percentage points
        assert abs(q_est - q) < 0.35  # Within 35 percentage points (estimation is approximate)
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        T, m, p, q = 30, 800000, 0.025, 0.38
        
        adopters1 = bass_adopters(T, m, p, q)
        adopters2 = bass_adopters(T, m, p, q)
        
        np.testing.assert_array_equal(adopters1, adopters2)

if __name__ == "__main__":
    pytest.main([__file__])
