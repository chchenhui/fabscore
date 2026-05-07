"""
Test suite for NPV and financial modeling functions.
Tests discount calculations, NPV/IRR accuracy, and Monte Carlo simulation.
"""
import pytest
import numpy as np
from src.econ.npv import (
    discount_factors, npv, irr, revenue_model, cost_model,
    calculate_cashflows, monte_carlo_npv
)

class TestDiscountFactors:
    """Test discount factor calculations."""
    
    def test_basic_functionality(self):
        """Test basic discount factor calculation."""
        factors = discount_factors(0.1, 5)
        
        assert len(factors) == 5
        assert factors[0] == 1.0  # First period not discounted
        assert all(factors[i] > factors[i+1] for i in range(4))  # Decreasing
        assert all(f > 0 for f in factors)  # All positive
    
    def test_zero_rate(self):
        """Test discount factors with zero rate."""
        factors = discount_factors(0.0, 4)
        expected = np.array([1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(factors, expected)
    
    def test_negative_rate(self):
        """Test discount factors with negative rate (deflation)."""
        factors = discount_factors(-0.05, 3)
        
        assert factors[0] == 1.0
        assert factors[1] > 1.0  # Future cash flows worth more
        assert factors[2] > factors[1]  # Increasing
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            discount_factors(0.1, 0)  # T = 0
        
        with pytest.raises(ValueError):
            discount_factors(0.1, -5)  # Negative T
        
        with pytest.raises(ValueError):
            discount_factors(-1.5, 5)  # Rate < -100%

class TestNPV:
    """Test NPV calculation function."""
    
    def test_basic_npv(self):
        """Test basic NPV calculation."""
        cashflows = np.array([-1000, 300, 400, 500, 400])
        npv_result = npv(cashflows, r_annual=0.1, quarterly=False)
        
        assert isinstance(npv_result, float)
        # Should be positive for this cash flow profile
        assert npv_result > 0
    
    def test_zero_discount_rate(self):
        """Test NPV with zero discount rate equals sum."""
        cashflows = np.array([-1000, 300, 400, 500, 400])
        npv_zero = npv(cashflows, r_annual=0.0, quarterly=False)
        
        np.testing.assert_almost_equal(npv_zero, cashflows.sum())
    
    def test_quarterly_conversion(self):
        """Test quarterly rate conversion."""
        cashflows = np.array([-1000, 250, 250, 250, 250])  # 4 quarters
        
        # Annual calculation
        npv_annual = npv(cashflows, r_annual=0.1, quarterly=False)
        
        # Quarterly calculation (should be different)
        npv_quarterly = npv(cashflows, r_annual=0.1, quarterly=True)
        
        assert npv_annual != npv_quarterly
        assert npv_quarterly > npv_annual  # Lower effective rate
    
    def test_empty_cashflows(self):
        """Test NPV with empty cashflows."""
        npv_empty = npv(np.array([]), r_annual=0.1)
        assert npv_empty == 0.0
    
    def test_single_cashflow(self):
        """Test NPV with single cashflow."""
        npv_single = npv(np.array([100]), r_annual=0.1, quarterly=False)
        assert npv_single == 100.0  # No discounting for period 0

class TestIRR:
    """Test IRR calculation function."""
    
    def test_basic_irr(self):
        """Test basic IRR calculation."""
        cashflows = np.array([-1000, 300, 400, 500, 400])
        irr_result = irr(cashflows, quarterly=False)
        
        assert isinstance(irr_result, float)
        assert not np.isnan(irr_result)
        assert 0.1 < irr_result < 0.6  # Reasonable range
    
    def test_irr_validation(self):
        """Test IRR calculation accuracy by checking NPV=0."""
        cashflows = np.array([-1000, 300, 400, 500, 400])
        irr_result = irr(cashflows, quarterly=False)
        
        # NPV at IRR should be approximately zero
        npv_at_irr = npv(cashflows, r_annual=irr_result, quarterly=False)
        assert abs(npv_at_irr) < 1.0  # Close to zero
    
    def test_quarterly_irr(self):
        """Test quarterly IRR conversion."""
        cashflows = np.array([-1000, 250, 250, 250, 250])
        
        irr_annual = irr(cashflows, quarterly=False)
        irr_quarterly = irr(cashflows, quarterly=True)
        
        # Quarterly should be annualized
        assert irr_quarterly != irr_annual
    
    def test_irr_edge_cases(self):
        """Test IRR edge cases."""
        # No positive cash flows
        bad_cashflows = np.array([-1000, -200, -300])
        irr_bad = irr(bad_cashflows)
        assert np.isnan(irr_bad)
        
        # Too few cash flows
        short_cashflows = np.array([100])
        irr_short = irr(short_cashflows)
        assert np.isnan(irr_short)
        
        # All zero cash flows
        zero_cashflows = np.array([0, 0, 0])
        irr_zero = irr(zero_cashflows)
        assert np.isnan(irr_zero)

class TestRevenueModel:
    """Test revenue modeling function."""
    
    def test_basic_revenue(self):
        """Test basic revenue calculation."""
        adopters = np.array([1000, 2000, 1500, 1000])
        revenue = revenue_model(
            adopters=adopters,
            list_price_monthly=1000,
            gtn_pct=0.7,
            adherence_rate=0.9
        )
        
        assert len(revenue) == len(adopters)
        assert all(r >= 0 for r in revenue)
        assert revenue[1] > revenue[0]  # Should grow with patient base
    
    def test_price_erosion(self):
        """Test price erosion effect."""
        adopters = np.array([1000, 1000, 1000, 1000])  # Constant adoption
        
        # No erosion
        revenue_no_erosion = revenue_model(
            adopters, 1000, 0.7, price_erosion_annual=0.0
        )
        
        # With erosion
        revenue_with_erosion = revenue_model(
            adopters, 1000, 0.7, price_erosion_annual=0.05
        )
        
        # Later periods should have lower revenue with erosion
        assert revenue_with_erosion[-1] < revenue_no_erosion[-1]
    
    def test_adherence_effect(self):
        """Test adherence/persistence effect."""
        adopters = np.array([1000, 0, 0, 0])  # All adoption in first period
        
        # Perfect adherence
        revenue_perfect = revenue_model(adopters, 1000, 0.7, adherence_rate=1.0)
        
        # Imperfect adherence
        revenue_decay = revenue_model(adopters, 1000, 0.7, adherence_rate=0.8)
        
        # Revenue should decay faster with lower adherence
        assert revenue_decay[-1] < revenue_perfect[-1]
    
    def test_empty_adopters(self):
        """Test revenue model with empty adopters array."""
        revenue = revenue_model(np.array([]), 1000, 0.7)
        assert len(revenue) == 0

class TestCostModel:
    """Test cost modeling function."""
    
    def test_basic_costs(self):
        """Test basic cost calculation."""
        revenue = np.array([1000000, 2000000, 3000000, 2500000])
        cogs, sga, rd = cost_model(
            revenue=revenue,
            cogs_pct=0.15,
            sga_launch=50000000,
            sga_decay_to_pct=0.5,
            rd_annual=20000000
        )
        
        assert len(cogs) == len(revenue)
        assert len(sga) == len(revenue)
        assert len(rd) == len(revenue)
        
        # COGS should scale with revenue
        assert cogs[1] > cogs[0]
        
        # SG&A should decay over time
        assert sga[-1] < sga[0]
        
        # R&D should be constant
        assert all(rd_val == rd[0] for rd_val in rd)
    
    def test_cost_components(self):
        """Test individual cost components."""
        revenue = np.array([1000000, 2000000])
        cogs, sga, rd = cost_model(revenue, 0.2, 100000000, 0.3, 40000000)
        
        # COGS should be 20% of revenue
        np.testing.assert_array_almost_equal(cogs, revenue * 0.2)
        
        # R&D should be quarterly amount
        expected_rd_quarterly = 40000000 / 4
        np.testing.assert_array_almost_equal(rd, expected_rd_quarterly)
    
    def test_empty_revenue(self):
        """Test cost model with empty revenue array."""
        cogs, sga, rd = cost_model(np.array([]), 0.15, 50000000, 0.5)
        assert len(cogs) == 0
        assert len(sga) == 0
        assert len(rd) == 0

class TestCalculateCashflows:
    """Test integrated cashflow calculation."""
    
    def test_basic_cashflow_calculation(self):
        """Test basic cashflow calculation."""
        adopters = np.array([10000, 20000, 15000, 10000])
        
        result = calculate_cashflows(
            adopters=adopters,
            list_price_monthly=1000,
            gtn_pct=0.7,
            cogs_pct=0.15,
            sga_launch=50000000,
            sga_decay_to_pct=0.5
        )
        
        # Check all components present
        required_keys = ['revenue', 'cogs', 'sga', 'rd', 'total_costs', 'net_cashflows']
        assert all(key in result for key in required_keys)
        
        # Check array lengths
        assert all(len(result[key]) == len(adopters) for key in required_keys)
        
        # Check relationships
        np.testing.assert_array_almost_equal(
            result['total_costs'], 
            result['cogs'] + result['sga'] + result['rd']
        )
        np.testing.assert_array_almost_equal(
            result['net_cashflows'],
            result['revenue'] - result['total_costs']
        )
    
    def test_profitability_transition(self):
        """Test transition from losses to profits."""
        # Small initial adoption, growing over time
        adopters = np.array([1000, 5000, 10000, 15000, 20000])
        
        result = calculate_cashflows(
            adopters=adopters,
            list_price_monthly=2000,
            gtn_pct=0.65,
            cogs_pct=0.1,
            sga_launch=100000000,
            sga_decay_to_pct=0.3
        )
        
        net_cf = result['net_cashflows']
        
        # Should start negative (high SG&A, low revenue)
        assert net_cf[0] < 0
        
        # Should eventually become positive
        assert any(cf > 0 for cf in net_cf)

class TestMonteCarloNPV:
    """Test Monte Carlo NPV simulation."""
    
    def test_basic_monte_carlo(self):
        """Test basic Monte Carlo simulation."""
        base_params = {
            'adopters': np.array([10000, 20000, 15000, 10000, 5000]),
            'list_price_monthly': 1500,
            'gtn_pct': 0.65,
            'cogs_pct': 0.15,
            'sga_launch': 75000000,
            'sga_decay_to_pct': 0.4,
            'adherence_rate': 0.85,
            'wacc_annual': 0.10
        }
        
        uncertainty_params = {
            'gtn_pct': 0.05,
            'list_price_monthly': 150,
            'adherence_rate': 0.1
        }
        
        results = monte_carlo_npv(
            base_params=base_params,
            uncertainty_params=uncertainty_params,
            n_simulations=1000,
            random_seed=42
        )
        
        # Check result structure
        assert 'npv' in results
        assert 'irr' in results
        assert 'n_simulations' in results
        
        # Check NPV statistics
        npv_stats = results['npv']
        assert 'mean' in npv_stats
        assert 'p10' in npv_stats
        assert 'p50' in npv_stats
        assert 'p90' in npv_stats
        
        # Check ordering
        assert npv_stats['p10'] <= npv_stats['p50'] <= npv_stats['p90']
        
        # Should have run requested simulations
        assert results['n_simulations'] <= 1000  # May be less due to failures
        assert results['success_rate'] > 0.8  # Most should succeed
    
    def test_monte_carlo_reproducibility(self):
        """Test Monte Carlo reproducibility with fixed seed."""
        base_params = {
            'adopters': np.array([5000, 10000, 8000]),
            'list_price_monthly': 1000,
            'gtn_pct': 0.7,
            'cogs_pct': 0.1,
            'sga_launch': 25000000,
            'sga_decay_to_pct': 0.5,
            'wacc_annual': 0.12
        }
        
        uncertainty_params = {'gtn_pct': 0.03}
        
        results1 = monte_carlo_npv(base_params, uncertainty_params, 100, 42)
        results2 = monte_carlo_npv(base_params, uncertainty_params, 100, 42)
        
        # Should get same results with same seed
        np.testing.assert_almost_equal(
            results1['npv']['mean'], results2['npv']['mean']
        )
    
    def test_monte_carlo_edge_cases(self):
        """Test Monte Carlo with edge cases."""
        base_params = {
            'adopters': np.array([1000]),
            'list_price_monthly': 500,
            'gtn_pct': 0.8,
            'cogs_pct': 0.2,
            'sga_launch': 10000000,
            'sga_decay_to_pct': 0.5,
            'wacc_annual': 0.15
        }
        
        # No uncertainty
        results_no_uncertainty = monte_carlo_npv(
            base_params, {}, n_simulations=100, random_seed=42
        )
        
        # Should have very low variance
        assert results_no_uncertainty['npv']['std'] < 1000  # Nearly deterministic

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_financial_model(self):
        """Test complete financial modeling workflow."""
        # Generate adoption curve (would come from Bass model)
        adopters = np.array([5000, 15000, 25000, 30000, 25000, 20000, 15000, 10000])
        
        # Calculate cashflows
        cashflow_result = calculate_cashflows(
            adopters=adopters,
            list_price_monthly=2000,
            gtn_pct=0.65,
            cogs_pct=0.12,
            sga_launch=100000000,
            sga_decay_to_pct=0.3,
            adherence_rate=0.8
        )
        
        # Calculate NPV and IRR
        net_cf = cashflow_result['net_cashflows']
        npv_result = npv(net_cf, r_annual=0.10, quarterly=True)
        irr_result = irr(net_cf, quarterly=True)
        
        # Basic sanity checks
        assert isinstance(npv_result, float)
        assert not np.isnan(npv_result)
        
        if not np.isnan(irr_result):
            assert irr_result > -0.5  # Not completely terrible
        
        # Revenue should generally increase then decrease
        revenue = cashflow_result['revenue']
        peak_revenue_idx = np.argmax(revenue)
        assert 2 <= peak_revenue_idx <= len(revenue) - 2  # Peak not at extremes
    
    def test_sensitivity_analysis(self):
        """Test sensitivity to key parameters."""
        base_adopters = np.array([10000, 20000, 15000, 10000])
        base_npv = npv(calculate_cashflows(
            adopters=base_adopters,
            list_price_monthly=1500,
            gtn_pct=0.65,
            cogs_pct=0.15,
            sga_launch=50000000,
            sga_decay_to_pct=0.4
        )['net_cashflows'], r_annual=0.10)
        
        # Higher price should increase NPV
        high_price_npv = npv(calculate_cashflows(
            adopters=base_adopters,
            list_price_monthly=2000,  # Higher price
            gtn_pct=0.65,
            cogs_pct=0.15,
            sga_launch=50000000,
            sga_decay_to_pct=0.4
        )['net_cashflows'], r_annual=0.10)
        
        assert high_price_npv > base_npv
        
        # Lower GtN should decrease NPV
        low_gtn_npv = npv(calculate_cashflows(
            adopters=base_adopters,
            list_price_monthly=1500,
            gtn_pct=0.55,  # Lower GtN
            cogs_pct=0.15,
            sga_launch=50000000,
            sga_decay_to_pct=0.4
        )['net_cashflows'], r_annual=0.10)
        
        assert low_gtn_npv < base_npv

if __name__ == "__main__":
    pytest.main([__file__])
