"""
Unit tests for industry-standard baseline forecasting methods.
Tests the core baselines that consultants actually use.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.baselines import (
    peak_sales_heuristic,
    estimate_peak_share,
    year2_naive,
    linear_trend_forecast,
    market_share_evolution,
    simple_bass_forecast,
    ensemble_baseline
)
from models.analogs import AnalogForecaster
from models.patient_flow import PatientFlowModel, PatientFlowParams


class TestBaselineMethods:
    """Test core baseline forecasting methods."""
    
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
    
    def test_peak_sales_heuristic(self):
        """Test peak sales calculation."""
        peak = peak_sales_heuristic(self.sample_drug)
        
        # Should be positive
        assert peak > 0
        
        # Should be reasonable magnitude (100k patients * $120k * 0.8 * 0.7 * peak_share)
        # Peak share for oncology with 2 competitors should be ~0.25
        expected_range = (100000 * 120000 * 0.8 * 0.7 * 0.1, 
                         100000 * 120000 * 0.8 * 0.7 * 0.6)
        assert expected_range[0] < peak < expected_range[1]
    
    def test_estimate_peak_share(self):
        """Test peak share estimation."""
        share = estimate_peak_share(self.sample_drug)
        
        # Should be between 0 and 1
        assert 0 < share < 1
        
        # Oncology with 2 competitors should have decent share
        assert share > 0.1
        assert share < 0.6  # Capped at 60%
    
    def test_estimate_peak_share_by_ta(self):
        """Test peak share varies by therapeutic area."""
        # Test different therapeutic areas
        tas = ['Oncology', 'Immunology', 'Cardiovascular', 'Rare Disease']
        shares = []
        
        for ta in tas:
            drug = self.sample_drug.copy()
            drug['therapeutic_area'] = ta
            share = estimate_peak_share(drug)
            shares.append(share)
        
        # Rare disease should have highest share
        rare_idx = tas.index('Rare Disease')
        assert shares[rare_idx] == max(shares)
        
        # All should be reasonable
        for share in shares:
            assert 0.01 <= share <= 0.6
    
    def test_estimate_peak_share_competition_effect(self):
        """Test competition affects peak share."""
        # Test different competition levels
        competitors = [0, 1, 3, 10]
        shares = []
        
        for comp in competitors:
            drug = self.sample_drug.copy()
            drug['competitor_count_at_launch'] = comp
            share = estimate_peak_share(drug)
            shares.append(share)
        
        # More competition should reduce share
        assert shares[0] > shares[1] > shares[2] > shares[3]
    
    def test_estimate_peak_share_access_effect(self):
        """Test access tier affects peak share."""
        # Test different access tiers
        access_tiers = ['OPEN', 'PA', 'NICHE']
        shares = []
        
        for access in access_tiers:
            drug = self.sample_drug.copy()
            drug['access_tier_at_launch'] = access
            share = estimate_peak_share(drug)
            shares.append(share)
        
        # OPEN should have highest share, NICHE lowest
        assert shares[0] > shares[1] > shares[2]
    
    def test_year2_naive(self):
        """Test year 2 naive forecast."""
        y2 = year2_naive(self.sample_drug)
        
        # Should be positive
        assert y2 > 0
        
        # Should be reasonable fraction of peak
        peak = peak_sales_heuristic(self.sample_drug)
        assert 0.1 * peak < y2 < 0.5 * peak
    
    def test_year2_naive_by_ta(self):
        """Test year 2 varies by therapeutic area."""
        # Oncology should have faster uptake than Neurology
        onco_drug = self.sample_drug.copy()
        onco_drug['therapeutic_area'] = 'Oncology'
        
        neuro_drug = self.sample_drug.copy()
        neuro_drug['therapeutic_area'] = 'Neurology'
        
        onco_y2 = year2_naive(onco_drug)
        neuro_y2 = year2_naive(neuro_drug)
        
        assert onco_y2 > neuro_y2
    
    def test_linear_trend_forecast(self):
        """Test linear trend forecast."""
        forecast = linear_trend_forecast(self.sample_drug, years=5)
        
        # Should have correct length
        assert len(forecast) == 5
        
        # Should be increasing to peak then flat
        assert forecast[0] > 0  # Launch year
        assert forecast[1] > forecast[0]  # Growth
        assert forecast[2] > forecast[1]  # More growth
        assert forecast[3] > forecast[2]  # Peak year
        assert forecast[4] == forecast[3]  # Flat after peak
    
    def test_market_share_evolution(self):
        """Test market share evolution."""
        shares = market_share_evolution(self.sample_drug, years=5)
        
        # Should have correct length
        assert len(shares) == 5
        
        # Should be increasing
        for i in range(1, len(shares)):
            assert shares[i] >= shares[i-1]
        
        # Should be between 0 and 1
        for share in shares:
            assert 0 <= share <= 1
        
    def test_simple_bass_forecast(self):
        """Test Bass diffusion forecast."""
        forecast = simple_bass_forecast(self.sample_drug, years=5)
        
        # Should have correct length
        assert len(forecast) == 5
        
        # Should be positive
        for rev in forecast:
            assert rev >= 0
        
        # Should show diffusion pattern (typically peak in middle years)
        assert forecast[0] > 0  # Launch year
        assert max(forecast) > forecast[0]  # Peak higher than launch
    
    def test_ensemble_baseline(self):
        """Test ensemble baseline."""
        ensemble = ensemble_baseline(self.sample_drug, years=5)
        
        # Should have all methods
        expected_methods = ['peak_sales_linear', 'simple_bass', 'market_share', 'year2_anchored', 'ensemble']
        for method in expected_methods:
            assert method in ensemble
        
        # All forecasts should have correct length
        for method, forecast in ensemble.items():
            assert len(forecast) == 5
        
        # Ensemble should be reasonable
        ensemble_forecast = ensemble['ensemble']
        assert len(ensemble_forecast) == 5
        assert all(rev >= 0 for rev in ensemble_forecast)


class TestAnalogForecaster:
    """Test analog-based forecasting."""
    
    def test_analog_forecaster_init(self):
        """Test analog forecaster initialization."""
        forecaster = AnalogForecaster()
        
        # Should initialize without error
        assert forecaster is not None
        assert forecaster.data_dir is not None
    
    def test_find_best_analogs(self):
        """Test finding best analogs."""
        forecaster = AnalogForecaster()
        
        # Test drug
        test_drug = pd.Series({
            'launch_id': 'TEST_DRUG',
            'therapeutic_area': 'Oncology',
            'mechanism': 'PD-1',
            'route': 'IV',
            'eligible_patients_at_launch': 110000,
            'list_price_month_usd_launch': 11000
        })
        
        # Without analogs data, should return empty list
        analogs = forecaster.find_best_analogs(test_drug, n_analogs=2)
        
        # Should return empty list when no analogs data
        assert isinstance(analogs, list)
        assert len(analogs) == 0


class TestPatientFlowModel:
    """Test patient flow modeling."""
    
    def test_patient_flow_params(self):
        """Test patient flow parameters."""
        params = PatientFlowParams(
            eligible_patients=100000,
            annual_incidence=0.05,
            annual_discontinuation=0.1,
            peak_penetration=0.3,
            years_to_peak=3,
            adherence_rate=0.8,
            switching_in_rate=0.05,
            switching_out_rate=0.02,
            annual_price=120000,
            net_to_gross=0.8,
            coverage_evolution=[0.5, 0.7, 0.8, 0.85, 0.9],
            prior_auth_approval=0.8
        )
        
        # Should create without error
        assert params.eligible_patients == 100000
        assert params.annual_price == 120000
    
    def test_patient_flow_model(self):
        """Test patient flow model."""
        params = PatientFlowParams(
            eligible_patients=100000,
            annual_incidence=0.05,
            annual_discontinuation=0.1,
            peak_penetration=0.3,
            years_to_peak=3,
            adherence_rate=0.8,
            switching_in_rate=0.05,
            switching_out_rate=0.02,
            annual_price=120000,
            net_to_gross=0.8,
            coverage_evolution=[0.5, 0.7, 0.8, 0.85, 0.9],
            prior_auth_approval=0.8
        )
        
        # Create a dummy drug row for testing
        test_drug = pd.Series({
            'launch_id': 'TEST_DRUG',
            'therapeutic_area': 'Oncology',
            'eligible_patients_at_launch': 100000,
            'list_price_month_usd_launch': 10000,
            'net_gtn_pct_launch': 0.8
        })
        
        model = PatientFlowModel()
        forecast = model.forecast(test_drug, years=5)
        
        # Should have correct length
        assert len(forecast) == 5
        
        # Should be positive
        for rev in forecast:
            assert rev >= 0
        
        # Should show growth pattern (launch year might be 0, but should have growth)
        assert max(forecast) > 0  # At least one year should have revenue
        if forecast[0] > 0:
            assert max(forecast) > forecast[0]  # Peak higher than launch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])