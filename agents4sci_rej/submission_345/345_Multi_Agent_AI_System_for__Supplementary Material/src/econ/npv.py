"""
NPV and financial modeling module for pharmaceutical commercial forecasting.

Provides functions for:
- Net Present Value (NPV) calculations with quarterly discounting
- Internal Rate of Return (IRR) calculations  
- Monte Carlo simulation for uncertainty analysis
- Revenue and cost modeling

Example usage:
    >>> import numpy as np
    >>> cashflows = np.array([100, 200, 300, 250, 200])
    >>> npv_result = npv(cashflows, r_annual=0.10, quarterly=True)
    >>> print(f"NPV: ${npv_result:,.0f}")
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.optimize import fsolve
import warnings
import pandas as pd
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

def discount_factors(r: float, T: int) -> np.ndarray:
    """
    Calculate discount factors for each period.
    
    Args:
        r: Discount rate per period
        T: Number of periods
    
    Returns:
        np.ndarray: Discount factors (1 + r)^(-t) for t = 0, 1, ..., T-1
        
    Example:
        >>> factors = discount_factors(0.025, 4)  # 2.5% quarterly
        >>> factors.round(4)
        array([1.0000, 0.9756, 0.9518, 0.9286])
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if r < -1:
        raise ValueError("Discount rate cannot be less than -100%")
    
    periods = np.arange(T)
    return (1 + r) ** (-periods)

def npv(cashflows: np.ndarray, r_annual: float, quarterly: bool = True) -> float:
    """
    Calculate Net Present Value of cashflows.
    
    Args:
        cashflows: Array of cashflows per period
        r_annual: Annual discount rate (e.g., 0.10 for 10%)
        quarterly: If True, convert annual rate to quarterly
    
    Returns:
        float: Net Present Value
        
    Example:
        >>> cashflows = np.array([-1000, 300, 400, 500, 400])
        >>> result = npv(cashflows, r_annual=0.10, quarterly=False)
        >>> result > 0  # Positive NPV
        True
    """
    if len(cashflows) == 0:
        return 0.0
    
    # Convert to quarterly rate if needed
    r = r_annual / 4 if quarterly else r_annual
    
    # Calculate discount factors
    disc_factors = discount_factors(r, len(cashflows))
    
    # Calculate NPV
    return float(np.sum(cashflows * disc_factors))

def irr(cashflows: np.ndarray, quarterly: bool = True) -> float:
    """
    Calculate Internal Rate of Return using numerical methods.
    
    Args:
        cashflows: Array of cashflows per period
        quarterly: If True, return annualized IRR from quarterly cashflows
    
    Returns:
        float: Internal Rate of Return (annualized if quarterly=True)
        
    Example:
        >>> cashflows = np.array([-1000, 300, 400, 500, 400])
        >>> result = irr(cashflows, quarterly=False)
        >>> 0.2 < result < 0.4  # Reasonable IRR range
        True
    """
    if len(cashflows) < 2:
        return float('nan')
    
    if np.sum(cashflows) <= 0:
        return float('nan')  # No positive return possible
    
    def npv_at_rate(rate):
        """NPV as function of discount rate."""
        if rate <= -1:
            return float('inf')
        periods = np.arange(len(cashflows))
        disc_factors = (1 + rate) ** (-periods)
        return np.sum(cashflows * disc_factors)
    
    try:
        # Solve for rate where NPV = 0
        irr_rate = fsolve(npv_at_rate, 0.1)[0]  # Start guess at 10%
        
        # Convert to annual rate if quarterly
        if quarterly and irr_rate > -1:
            irr_rate = (1 + irr_rate) ** 4 - 1
        
        return float(irr_rate)
    
    except:
        return float('nan')

def revenue_model(adopters: np.ndarray, 
                 list_price_monthly: float,
                 gtn_pct: float,
                 adherence_rate: float = 1.0,
                 price_erosion_annual: float = 0.0) -> np.ndarray:
    """
    Calculate quarterly revenue from adoption and pricing assumptions.
    
    Args:
        adopters: New adopters per quarter
        list_price_monthly: Monthly list price
        gtn_pct: Gross-to-net percentage (net = list * gtn_pct)
        adherence_rate: Patient adherence/persistence rate
        price_erosion_annual: Annual price erosion rate
    
    Returns:
        np.ndarray: Quarterly net revenue
    """
    if len(adopters) == 0:
        return np.array([])
    
    # Calculate net price per quarter (3 months)
    net_price_quarterly = list_price_monthly * 3 * gtn_pct
    
    # Apply price erosion over time
    quarters = np.arange(len(adopters))
    price_erosion_quarterly = (1 - price_erosion_annual) ** (quarters / 4)
    net_prices = net_price_quarterly * price_erosion_quarterly
    
    # Calculate patient base (cumulative with adherence decay)
    patient_base = np.zeros(len(adopters))
    for t in range(len(adopters)):
        # New patients this quarter
        new_patients = adopters[t]
        
        # Existing patients with adherence decay
        if t > 0:
            existing_patients = patient_base[t-1] * adherence_rate
        else:
            existing_patients = 0
        
        patient_base[t] = new_patients + existing_patients
    
    # Calculate revenue
    revenue = patient_base * net_prices
    
    return revenue

def cost_model(revenue: np.ndarray,
               cogs_pct: float,
               sga_launch: float,
               sga_decay_to_pct: float,
               rd_annual: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate quarterly costs (COGS, SG&A, R&D).
    
    Args:
        revenue: Quarterly revenue
        cogs_pct: Cost of goods sold as % of revenue
        sga_launch: Initial quarterly SG&A spend
        sga_decay_to_pct: SG&A decay target as % of launch spend
        rd_annual: Annual R&D spend
    
    Returns:
        Tuple of (cogs, sga, rd) cost arrays
    """
    if len(revenue) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # COGS: percentage of revenue
    cogs = revenue * cogs_pct
    
    # SG&A: starts high, decays over time
    quarters = np.arange(len(revenue))
    sga_decay_rate = (sga_decay_to_pct) ** (1 / len(revenue))  # Geometric decay
    sga_multipliers = sga_decay_rate ** quarters
    sga = sga_launch * sga_multipliers
    
    # R&D: constant quarterly amount
    rd_quarterly = rd_annual / 4
    rd = np.full(len(revenue), rd_quarterly)
    
    return cogs, sga, rd

def calculate_cashflows(adopters: np.ndarray,
                       list_price_monthly: float,
                       gtn_pct: float,
                       cogs_pct: float,
                       sga_launch: float,
                       sga_decay_to_pct: float,
                       adherence_rate: float = 0.85,
                       price_erosion_annual: float = 0.02,
                       rd_annual: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Calculate complete quarterly cashflow model.
    
    Args:
        adopters: New adopters per quarter
        list_price_monthly: Monthly list price
        gtn_pct: Gross-to-net percentage
        cogs_pct: COGS as % of revenue
        sga_launch: Initial quarterly SG&A
        sga_decay_to_pct: SG&A decay target
        adherence_rate: Patient persistence rate
        price_erosion_annual: Annual price decline
        rd_annual: Annual R&D spend
    
    Returns:
        Dict with revenue, costs, and net cashflows
    """
    # Calculate revenue
    revenue = revenue_model(adopters, list_price_monthly, gtn_pct, 
                           adherence_rate, price_erosion_annual)
    
    # Calculate costs
    cogs, sga, rd = cost_model(revenue, cogs_pct, sga_launch, 
                              sga_decay_to_pct, rd_annual)
    
    # Net cashflows
    total_costs = cogs + sga + rd
    net_cashflows = revenue - total_costs
    
    return {
        'revenue': revenue,
        'cogs': cogs,
        'sga': sga,
        'rd': rd,
        'total_costs': total_costs,
        'net_cashflows': net_cashflows
    }

def monte_carlo_npv(base_params: Dict[str, Any],
                   uncertainty_params: Dict[str, float],
                   n_simulations: int = 10000,
                   random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation for NPV under uncertainty.
    
    Args:
        base_params: Base case parameters
        uncertainty_params: Standard deviations for uncertain parameters
        n_simulations: Number of simulation runs
        random_seed: Random seed for reproducibility
    
    Returns:
        Dict with NPV distribution and statistics
    """
    # Import provenance tracking
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from utils.provenance import log_experiment_provenance
        # Log provenance
        provenance = log_experiment_provenance(
            experiment_name="monte_carlo_npv",
            seed=random_seed if random_seed is not None else -1,
            parameters={
                "n_simulations": n_simulations,
                "base_params": list(base_params.keys()),
                "uncertainty_params": list(uncertainty_params.keys())
            }
        )
    except ImportError:
        # Fallback if provenance module not available
        pass
    
    if random_seed is not None:
        np.random.seed(random_seed)
        print(f"[MONTE CARLO] Random seed set to: {random_seed}")
    
    npv_results = []
    irr_results = []
    
    for _ in range(n_simulations):
        # Sample uncertain parameters
        params = base_params.copy()
        
        # Apply uncertainty to key parameters
        for param, std in uncertainty_params.items():
            if param in params:
                if param in ['gtn_pct', 'adherence_rate']:
                    # Beta distribution for rates (bounded 0-1)
                    mean = params[param]
                    # Convert to beta parameters
                    alpha = mean * (mean * (1 - mean) / std**2 - 1)
                    beta = (1 - mean) * (mean * (1 - mean) / std**2 - 1)
                    if alpha > 0 and beta > 0:
                        params[param] = np.random.beta(alpha, beta)
                else:
                    # Normal distribution with truncation
                    value = np.random.normal(params[param], std)
                    # Ensure positive for price/cost parameters
                    if param in ['list_price_monthly', 'sga_launch']:
                        params[param] = max(0.1 * params[param], value)
                    else:
                        params[param] = value
        
        # Calculate cashflows for this simulation
        try:
            # Extract WACC for NPV calculation
            wacc_annual = params.pop('wacc_annual', 0.10)
            
            cashflow_result = calculate_cashflows(**params)
            net_cf = cashflow_result['net_cashflows']
            
            # Calculate NPV and IRR
            npv_sim = npv(net_cf, wacc_annual)
            irr_sim = irr(net_cf)
            
            npv_results.append(npv_sim)
            if not np.isnan(irr_sim):
                irr_results.append(irr_sim)
                
        except Exception as e:
            # Skip failed simulations
            continue
    
    # Calculate statistics
    npv_array = np.array(npv_results)
    irr_array = np.array(irr_results)
    
    # Handle empty results case
    if len(npv_results) == 0:
        results = {
            'npv': {
                'values': npv_array,
                'mean': np.nan,
                'std': np.nan,
                'p10': np.nan,
                'p50': np.nan,
                'p90': np.nan,
                'prob_positive': np.nan
            },
            'irr': {
                'values': irr_array,
                'mean': np.nan,
                'std': np.nan,
                'p10': np.nan,
                'p50': np.nan,
                'p90': np.nan,
            },
            'n_simulations': 0,
            'success_rate': 0.0,
            'random_seed': random_seed
        }
    else:
        results = {
            'npv': {
                'values': npv_array,
                'mean': np.mean(npv_array),
                'std': np.std(npv_array),
                'p10': np.percentile(npv_array, 10),
                'p50': np.percentile(npv_array, 50),
                'p90': np.percentile(npv_array, 90),
                'prob_positive': np.mean(npv_array > 0)
            },
            'irr': {
                'values': irr_array,
                'mean': np.mean(irr_array) if len(irr_array) > 0 else np.nan,
                'std': np.std(irr_array) if len(irr_array) > 0 else np.nan,
                'p10': np.percentile(irr_array, 10) if len(irr_array) > 0 else np.nan,
                'p50': np.percentile(irr_array, 50) if len(irr_array) > 0 else np.nan,
                'p90': np.percentile(irr_array, 90) if len(irr_array) > 0 else np.nan,
            },
            'n_simulations': len(npv_results),
            'success_rate': len(npv_results) / n_simulations,
            'random_seed': random_seed
        }
    
    return results

def explain_npv_drivers(base_params: Dict[str, Any],
                       parameter_ranges: Dict[str, Tuple[float, float]],
                       n_samples: int = 1000,
                       random_seed: Optional[int] = 42) -> Dict[str, Any]:
    """
    Use SHAP to explain NPV drivers and parameter importance.
    
    Args:
        base_params: Base case parameters for NPV calculation
        parameter_ranges: Min/max ranges for each parameter to vary
        n_samples: Number of samples for SHAP analysis
        random_seed: Random seed for reproducibility
    
    Returns:
        Dict with SHAP values, importance ranking, and explanations
        
    Example:
        >>> base = {'list_price_monthly': 5000, 'gtn_pct': 0.75, 'adopters': np.array([100,200,300])}
        >>> ranges = {'list_price_monthly': (3000, 7000), 'gtn_pct': (0.6, 0.9)}
        >>> explanation = explain_npv_drivers(base, ranges)
        >>> print(f"Top driver: {explanation['feature_importance'].index[0]}")
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required for explainability. Install with: pip install shap")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Extract parameters to vary
    param_names = list(parameter_ranges.keys())
    param_values = []
    
    # Generate samples within parameter ranges
    for _ in range(n_samples):
        sample_params = base_params.copy()
        sample_values = []
        
        for param_name in param_names:
            min_val, max_val = parameter_ranges[param_name]
            if param_name in ['gtn_pct', 'adherence_rate']:
                # Keep rates within [0,1] bounds
                sample_val = np.random.uniform(max(0, min_val), min(1, max_val))
            else:
                sample_val = np.random.uniform(min_val, max_val)
            
            sample_params[param_name] = sample_val
            sample_values.append(sample_val)
        
        param_values.append(sample_values)
    
    # Convert to numpy array for SHAP
    X = np.array(param_values)
    
    # Define NPV prediction function
    def predict_npv(param_matrix):
        """Predict NPV for matrix of parameter combinations"""
        npv_predictions = []
        
        for i in range(param_matrix.shape[0]):
            try:
                # Create parameter dict for this sample
                params = base_params.copy()
                for j, param_name in enumerate(param_names):
                    params[param_name] = param_matrix[i, j]
                
                # Calculate NPV (extract WACC if present)
                wacc_annual = params.pop('wacc_annual', 0.10)
                cashflow_result = calculate_cashflows(**params)
                net_cf = cashflow_result['net_cashflows']
                npv_val = npv(net_cf, wacc_annual)
                npv_predictions.append(npv_val)
                
            except Exception:
                # Return base case NPV for failed calculations
                npv_predictions.append(0.0)
        
        return np.array(npv_predictions)
    
    # Calculate baseline NPV
    try:
        baseline_params = base_params.copy()
        wacc_annual = baseline_params.pop('wacc_annual', 0.10)
        baseline_cf = calculate_cashflows(**baseline_params)
        baseline_npv = npv(baseline_cf['net_cashflows'], wacc_annual)
    except Exception:
        baseline_npv = 0.0
    
    # Create SHAP explainer (use lightweight Permutation explainer for speed)
    explainer = shap.explainers.Permutation(predict_npv, X[:20])  # Small background
    shap_values = explainer(X[:min(50, n_samples)])  # Heavily limit for performance
    
    # Calculate feature importance (mean absolute SHAP values)
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.Series(mean_shap, index=param_names).sort_values(ascending=False)
    
    # Calculate parameter impact ranges
    npv_predictions = predict_npv(X)
    parameter_impacts = {}
    
    for i, param_name in enumerate(param_names):
        param_shap = shap_values.values[:, i]
        parameter_impacts[param_name] = {
            'mean_impact': np.mean(param_shap),
            'impact_range': (np.min(param_shap), np.max(param_shap)),
            'std_impact': np.std(param_shap),
            'importance_rank': feature_importance.index.get_loc(param_name) + 1
        }
    
    # Create summary explanations
    top_driver = feature_importance.index[0]
    top_impact = parameter_impacts[top_driver]['mean_impact']
    impact_direction = "increases" if top_impact > 0 else "decreases"
    
    summary_explanation = (
        f"The top NPV driver is '{top_driver}' with average impact of ${top_impact:,.0f}. "
        f"This parameter {impact_direction} NPV when above baseline levels."
    )
    
    # Risk assessment
    npv_p10 = np.percentile(npv_predictions, 10)
    npv_p90 = np.percentile(npv_predictions, 90)
    downside_risk = baseline_npv - npv_p10
    upside_potential = npv_p90 - baseline_npv
    
    return {
        'shap_values': shap_values,
        'feature_importance': feature_importance,
        'parameter_impacts': parameter_impacts,
        'baseline_npv': baseline_npv,
        'npv_predictions': npv_predictions,
        'summary_explanation': summary_explanation,
        'risk_assessment': {
            'baseline_npv': baseline_npv,
            'p10_npv': npv_p10,
            'p90_npv': npv_p90,
            'downside_risk': downside_risk,
            'upside_potential': upside_potential,
            'risk_ratio': downside_risk / max(upside_potential, 1)  # Risk/reward ratio
        }
    }

if __name__ == "__main__":
    # Demo usage
    print("NPV Analysis Demo")
    print("=" * 40)
    
    # Sample cashflows (in millions)
    cashflows = np.array([-50, 10, 25, 40, 45, 50, 45, 40, 35, 30])  # 10 periods
    
    # Calculate NPV and IRR
    npv_result = npv(cashflows, r_annual=0.10, quarterly=False)
    irr_result = irr(cashflows, quarterly=False)
    
    print(f"Cashflows: {cashflows}")
    print(f"NPV (10% discount): ${npv_result:.1f}M")
    print(f"IRR: {irr_result:.1%}")
    
    # Test quarterly conversion
    quarterly_rate = 0.10 / 4
    npv_quarterly = npv(cashflows, r_annual=0.10, quarterly=True)
    print(f"NPV (quarterly): ${npv_quarterly:.1f}M")
    
    print("\nValidation tests:")
    # Test: NPV at 0% discount should equal sum
    npv_zero = npv(cashflows, r_annual=0.0, quarterly=False)
    print(f"NPV at 0%: ${npv_zero:.1f}M (should equal sum: ${cashflows.sum():.1f}M)")
    
    # Test: Discount factors sum
    factors = discount_factors(0.025, 4)
    print(f"Discount factors: {factors.round(4)}")
    print(f"Factors decrease: {np.all(np.diff(factors) < 0)}")
    
    # Test SHAP explainability if available
    if SHAP_AVAILABLE:
        print("\n" + "=" * 40)
        print("SHAP NPV Driver Analysis Demo")
        print("=" * 40)
        
        # Example pharmaceutical scenario
        adopters_example = np.array([500, 1500, 2500, 3000, 3500, 3200, 2800, 2400, 2000, 1600])
        
        base_scenario = {
            'adopters': adopters_example,
            'list_price_monthly': 4500,
            'gtn_pct': 0.72,
            'cogs_pct': 0.15,
            'sga_launch': 50_000_000,
            'sga_decay_to_pct': 0.3,
            'adherence_rate': 0.80,
            'wacc_annual': 0.12
        }
        
        # Define ranges for key uncertainty parameters
        param_ranges = {
            'list_price_monthly': (3500, 5500),
            'gtn_pct': (0.65, 0.80),
            'adherence_rate': (0.70, 0.90),
            'sga_launch': (30_000_000, 70_000_000)
        }
        
        try:
            explanation = explain_npv_drivers(base_scenario, param_ranges, n_samples=100)
            
            print(f"Baseline NPV: ${explanation['baseline_npv']:,.0f}")
            print(f"\nTop 3 NPV Drivers:")
            for i, (param, importance) in enumerate(explanation['feature_importance'].head(3).items()):
                rank = i + 1
                impact = explanation['parameter_impacts'][param]
                print(f"{rank}. {param}: ${importance:,.0f} avg impact (rank #{impact['importance_rank']})")
            
            print(f"\n{explanation['summary_explanation']}")
            
            risk = explanation['risk_assessment']
            print(f"\nRisk Assessment:")
            print(f"  P10 NPV: ${risk['p10_npv']:,.0f}")
            print(f"  P90 NPV: ${risk['p90_npv']:,.0f}")
            print(f"  Downside risk: ${risk['downside_risk']:,.0f}")
            print(f"  Upside potential: ${risk['upside_potential']:,.0f}")
            print(f"  Risk/reward ratio: {risk['risk_ratio']:.2f}")
            
        except Exception as e:
            print(f"SHAP demo failed: {e}")
    else:
        print("\nSHAP not available - install with: pip install shap")
