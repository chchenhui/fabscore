"""
Test suite for price and access simulation functions.
Tests rule-based mapping, access constraints, and pricing scenarios.
"""
import pytest
import numpy as np
import pandas as pd
from src.access.pricing_sim import (
    tier_from_price, gtn_from_tier, adoption_ceiling_from_tier,
    apply_access_constraints, create_pricing_scenario,
    optimize_price_access_tradeoff, ACCESS_TIERS, GTN_BY_TIER
)

class TestTierFromPrice:
    """Test price-to-tier mapping function."""
    
    def test_basic_tier_mapping(self):
        """Test basic tier mapping with default thresholds."""
        # Test OPEN tier
        assert tier_from_price(500) == "OPEN"
        assert tier_from_price(999) == "OPEN"
        
        # Test PA tier
        assert tier_from_price(1000) == "PA"
        assert tier_from_price(2499) == "PA"
        
        # Test NICHE tier
        assert tier_from_price(2500) == "NICHE"
        assert tier_from_price(5000) == "NICHE"
    
    def test_boundary_conditions(self):
        """Test tier mapping at boundary conditions."""
        # Exactly at thresholds
        assert tier_from_price(1000) == "PA"  # At OPEN/PA boundary
        assert tier_from_price(2500) == "NICHE"  # At PA/NICHE boundary
        
        # Just below thresholds
        assert tier_from_price(999.99) == "OPEN"
        assert tier_from_price(2499.99) == "PA"
    
    def test_custom_thresholds(self):
        """Test tier mapping with custom thresholds."""
        custom_thresholds = {
            'open_max': 800,
            'pa_max': 2000
        }
        
        assert tier_from_price(750, custom_thresholds) == "OPEN"
        assert tier_from_price(1500, custom_thresholds) == "PA"
        assert tier_from_price(2500, custom_thresholds) == "NICHE"
    
    def test_edge_cases(self):
        """Test edge cases for price mapping."""
        # Very low price
        assert tier_from_price(1) == "OPEN"
        
        # Very high price
        assert tier_from_price(10000) == "NICHE"
        
        # Zero price (edge case)
        assert tier_from_price(0) == "OPEN"

class TestGtnFromTier:
    """Test gross-to-net mapping function."""
    
    def test_basic_gtn_mapping(self):
        """Test basic GtN mapping."""
        assert gtn_from_tier("OPEN") == GTN_BY_TIER["OPEN"]
        assert gtn_from_tier("PA") == GTN_BY_TIER["PA"]
        assert gtn_from_tier("NICHE") == GTN_BY_TIER["NICHE"]
    
    def test_custom_gtn_mapping(self):
        """Test GtN mapping with custom values."""
        custom_gtn = {
            "OPEN": 0.8,
            "PA": 0.7,
            "NICHE": 0.6
        }
        
        assert gtn_from_tier("OPEN", custom_gtn) == 0.8
        assert gtn_from_tier("PA", custom_gtn) == 0.7
        assert gtn_from_tier("NICHE", custom_gtn) == 0.6
    
    def test_unknown_tier(self):
        """Test GtN mapping with unknown tier."""
        # Should return default PA value
        assert gtn_from_tier("UNKNOWN") == GTN_BY_TIER["PA"]
    
    def test_gtn_ordering(self):
        """Test that GtN decreases with more restrictive tiers."""
        gtn_open = gtn_from_tier("OPEN")
        gtn_pa = gtn_from_tier("PA")
        gtn_niche = gtn_from_tier("NICHE")
        
        assert gtn_open > gtn_pa > gtn_niche  # More restrictive = lower GtN

class TestAdoptionCeilingFromTier:
    """Test adoption ceiling mapping function."""
    
    def test_basic_ceiling_mapping(self):
        """Test basic ceiling mapping."""
        assert adoption_ceiling_from_tier("OPEN") == ACCESS_TIERS["OPEN"]
        assert adoption_ceiling_from_tier("PA") == ACCESS_TIERS["PA"]
        assert adoption_ceiling_from_tier("NICHE") == ACCESS_TIERS["NICHE"]
    
    def test_custom_ceiling_mapping(self):
        """Test ceiling mapping with custom values."""
        custom_ceilings = {
            "OPEN": 0.95,
            "PA": 0.5,
            "NICHE": 0.2
        }
        
        assert adoption_ceiling_from_tier("OPEN", custom_ceilings) == 0.95
        assert adoption_ceiling_from_tier("PA", custom_ceilings) == 0.5
        assert adoption_ceiling_from_tier("NICHE", custom_ceilings) == 0.2
    
    def test_ceiling_ordering(self):
        """Test that ceilings decrease with more restrictive tiers."""
        ceiling_open = adoption_ceiling_from_tier("OPEN")
        ceiling_pa = adoption_ceiling_from_tier("PA")
        ceiling_niche = adoption_ceiling_from_tier("NICHE")
        
        assert ceiling_open > ceiling_pa > ceiling_niche
        assert all(0 <= c <= 1 for c in [ceiling_open, ceiling_pa, ceiling_niche])

class TestApplyAccessConstraints:
    """Test access constraint application function."""
    
    def test_basic_constraint_application(self):
        """Test basic access constraint application."""
        unconstrained = np.array([1000, 2000, 3000, 2500, 2000])
        
        # OPEN tier should not constrain
        constrained_open = apply_access_constraints(unconstrained, "OPEN")
        np.testing.assert_array_almost_equal(constrained_open, unconstrained)
        
        # PA tier should constrain
        constrained_pa = apply_access_constraints(unconstrained, "PA")
        assert np.sum(constrained_pa) <= np.sum(unconstrained)
        assert np.sum(constrained_pa) == np.sum(unconstrained) * ACCESS_TIERS["PA"]
        
        # NICHE tier should constrain more
        constrained_niche = apply_access_constraints(unconstrained, "NICHE")
        assert np.sum(constrained_niche) < np.sum(constrained_pa)
    
    def test_constraint_monotonicity(self):
        """Test that constraints preserve monotonic cumulative adoption."""
        unconstrained = np.array([500, 1500, 2000, 1800, 1200, 800])
        constrained = apply_access_constraints(unconstrained, "PA")
        
        cumulative_constrained = np.cumsum(constrained)
        
        # Should be monotonic (non-decreasing)
        diffs = np.diff(cumulative_constrained)
        assert all(d >= -1e-10 for d in diffs)  # Allow for numerical precision
    
    def test_constraint_mass_conservation(self):
        """Test that constraints don't exceed ceiling."""
        unconstrained = np.array([2000, 4000, 3000, 2000])
        
        for tier in ["OPEN", "PA", "NICHE"]:
            constrained = apply_access_constraints(unconstrained, tier)
            ceiling = adoption_ceiling_from_tier(tier)
            
            total_unconstrained = np.sum(unconstrained)
            total_constrained = np.sum(constrained)
            expected_max = total_unconstrained * ceiling
            
            assert total_constrained <= expected_max + 1e-10  # Numerical precision
    
    def test_edge_cases(self):
        """Test constraint application edge cases."""
        # Empty array
        empty_constrained = apply_access_constraints(np.array([]), "PA")
        assert len(empty_constrained) == 0
        
        # Single element
        single_constrained = apply_access_constraints(np.array([1000]), "NICHE")
        expected = 1000 * ACCESS_TIERS["NICHE"]
        np.testing.assert_almost_equal(single_constrained, [expected])
        
        # Zero adoption
        zero_constrained = apply_access_constraints(np.array([0, 0, 0]), "PA")
        np.testing.assert_array_equal(zero_constrained, [0, 0, 0])
    
    def test_custom_ceilings(self):
        """Test constraints with custom ceiling values."""
        unconstrained = np.array([1000, 2000, 1500])
        custom_ceilings = {"CUSTOM": 0.3}
        
        constrained = apply_access_constraints(unconstrained, "CUSTOM", custom_ceilings)
        expected_total = np.sum(unconstrained) * 0.3
        
        np.testing.assert_almost_equal(np.sum(constrained), expected_total)

class TestCreatePricingScenario:
    """Test pricing scenario creation function."""
    
    def test_basic_scenario_creation(self):
        """Test basic pricing scenario creation."""
        scenario = create_pricing_scenario((500, 3000), n_points=6)
        
        # Check structure
        assert isinstance(scenario, pd.DataFrame)
        expected_columns = ['price_monthly', 'access_tier', 'adoption_ceiling', 
                          'gtn_pct', 'net_price', 'net_price_annual']
        assert all(col in scenario.columns for col in expected_columns)
        
        # Check data
        assert len(scenario) == 6
        assert scenario['price_monthly'].min() == 500
        assert scenario['price_monthly'].max() == 3000
        
        # Check tier progression
        tiers = scenario['access_tier'].tolist()
        assert "OPEN" in tiers
        assert "NICHE" in tiers
    
    def test_scenario_consistency(self):
        """Test internal consistency of pricing scenarios."""
        scenario = create_pricing_scenario((800, 2800), n_points=10)
        
        for _, row in scenario.iterrows():
            # Check tier mapping consistency
            expected_tier = tier_from_price(row['price_monthly'])
            assert row['access_tier'] == expected_tier
            
            # Check GtN consistency
            expected_gtn = gtn_from_tier(row['access_tier'])
            assert abs(row['gtn_pct'] - expected_gtn) < 1e-10
            
            # Check ceiling consistency
            expected_ceiling = adoption_ceiling_from_tier(row['access_tier'])
            assert abs(row['adoption_ceiling'] - expected_ceiling) < 1e-10
            
            # Check net price calculation
            expected_net = row['price_monthly'] * row['gtn_pct']
            assert abs(row['net_price'] - expected_net) < 1e-10
            
            # Check annual calculation
            expected_annual = row['net_price'] * 12
            assert abs(row['net_price_annual'] - expected_annual) < 1e-10
    
    def test_custom_thresholds_in_scenario(self):
        """Test pricing scenario with custom thresholds."""
        custom_thresholds = {'open_max': 1200, 'pa_max': 2200}
        scenario = create_pricing_scenario((600, 2800), n_points=8, 
                                         custom_thresholds=custom_thresholds)
        
        # Check that custom thresholds are respected
        open_prices = scenario[scenario['access_tier'] == 'OPEN']['price_monthly']
        pa_prices = scenario[scenario['access_tier'] == 'PA']['price_monthly']
        niche_prices = scenario[scenario['access_tier'] == 'NICHE']['price_monthly']
        
        if len(open_prices) > 0:
            assert open_prices.max() < custom_thresholds['open_max']
        if len(pa_prices) > 0:
            assert pa_prices.min() >= custom_thresholds['open_max']
            assert pa_prices.max() < custom_thresholds['pa_max']
        if len(niche_prices) > 0:
            assert niche_prices.min() >= custom_thresholds['pa_max']

class TestOptimizePriceAccessTradeoff:
    """Test price-access tradeoff optimization function."""
    
    def test_basic_optimization(self):
        """Test basic price optimization."""
        # Simple adoption curve
        adopters = np.array([5000, 10000, 8000, 6000, 4000])
        
        result = optimize_price_access_tradeoff(adopters, (800, 3200), n_points=10)
        
        # Check result structure
        assert 'optimal_price' in result
        assert 'optimal_tier' in result
        assert 'optimal_patients' in result
        assert 'optimal_revenue' in result
        assert 'scenario_analysis' in result
        
        # Check data types
        assert isinstance(result['optimal_price'], (int, float))
        assert isinstance(result['optimal_tier'], str)
        assert isinstance(result['optimal_patients'], (int, float))
        assert isinstance(result['optimal_revenue'], (int, float))
        assert isinstance(result['scenario_analysis'], pd.DataFrame)
    
    def test_optimization_logic(self):
        """Test that optimization finds reasonable solution."""
        adopters = np.array([3000, 6000, 5000, 4000])
        result = optimize_price_access_tradeoff(adopters, (500, 4000), n_points=20)
        
        scenario_df = result['scenario_analysis']
        
        # Optimal should be the maximum revenue point
        max_revenue_idx = scenario_df['revenue'].idxmax()
        optimal_row = scenario_df.iloc[max_revenue_idx]
        
        assert abs(result['optimal_price'] - optimal_row['price']) < 1e-10
        assert result['optimal_tier'] == optimal_row['tier']
        assert abs(result['optimal_revenue'] - optimal_row['revenue']) < 1e-10
    
    def test_tradeoff_dynamics(self):
        """Test that optimization captures price-access tradeoffs."""
        adopters = np.array([10000, 15000, 12000, 8000])
        result = optimize_price_access_tradeoff(adopters, (600, 3600), n_points=15)
        
        scenario_df = result['scenario_analysis']
        
        # Should see tradeoff: higher price → fewer patients but more revenue per patient
        low_price_rows = scenario_df[scenario_df['price'] < 1500]
        high_price_rows = scenario_df[scenario_df['price'] > 2500]
        
        if len(low_price_rows) > 0 and len(high_price_rows) > 0:
            avg_patients_low = low_price_rows['patients'].mean()
            avg_patients_high = high_price_rows['patients'].mean()
            avg_revenue_per_patient_low = low_price_rows['revenue_per_patient'].mean()
            avg_revenue_per_patient_high = high_price_rows['revenue_per_patient'].mean()
            
            # Higher prices should have fewer patients but higher revenue per patient
            assert avg_patients_low > avg_patients_high
            assert avg_revenue_per_patient_low < avg_revenue_per_patient_high
    
    def test_edge_cases(self):
        """Test optimization edge cases."""
        # Very small adoption
        small_adopters = np.array([10, 20, 15])
        result_small = optimize_price_access_tradeoff(small_adopters, (1000, 2000))
        assert result_small['optimal_revenue'] >= 0
        
        # Single period adoption
        single_adopters = np.array([5000])
        result_single = optimize_price_access_tradeoff(single_adopters, (800, 1200))
        assert result_single['optimal_patients'] <= 5000

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_pricing_workflow(self):
        """Test complete pricing and access workflow."""
        # Start with unconstrained adoption
        unconstrained_adopters = np.array([8000, 15000, 12000, 10000, 7000])
        
        # Test different price points
        test_prices = [750, 1500, 2800]
        
        results = []
        for price in test_prices:
            # Map to tier
            tier = tier_from_price(price)
            
            # Apply constraints
            constrained_adopters = apply_access_constraints(unconstrained_adopters, tier)
            
            # Calculate metrics
            total_patients = np.sum(constrained_adopters)
            gtn = gtn_from_tier(tier)
            annual_revenue = total_patients * price * 12 * gtn
            
            results.append({
                'price': price,
                'tier': tier,
                'patients': total_patients,
                'revenue': annual_revenue
            })
        
        # Check that results make sense
        results_df = pd.DataFrame(results)
        
        # Should have different tiers
        unique_tiers = results_df['tier'].unique()
        assert len(unique_tiers) > 1
        
        # Patient counts should decrease with more restrictive tiers
        open_patients = results_df[results_df['tier'] == 'OPEN']['patients']
        niche_patients = results_df[results_df['tier'] == 'NICHE']['patients']
        
        if len(open_patients) > 0 and len(niche_patients) > 0:
            assert open_patients.iloc[0] > niche_patients.iloc[0]
    
    def test_consistency_across_functions(self):
        """Test consistency between related functions."""
        price = 1800
        
        # Get tier from price
        tier = tier_from_price(price)
        
        # Get associated values
        gtn = gtn_from_tier(tier)
        ceiling = adoption_ceiling_from_tier(tier)
        
        # Create scenario and check consistency
        scenario = create_pricing_scenario((price, price), n_points=1)
        scenario_row = scenario.iloc[0]
        
        assert scenario_row['access_tier'] == tier
        assert abs(scenario_row['gtn_pct'] - gtn) < 1e-10
        assert abs(scenario_row['adoption_ceiling'] - ceiling) < 1e-10
        assert abs(scenario_row['price_monthly'] - price) < 1e-10

if __name__ == "__main__":
    pytest.main([__file__])
