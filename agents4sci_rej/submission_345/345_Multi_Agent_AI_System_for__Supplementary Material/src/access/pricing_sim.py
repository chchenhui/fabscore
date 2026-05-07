"""
Unified Access Constraint Logic - Single Source of Truth
Following Linus principle: "Good code has no special cases"

Core concept: effective_market = TAM × access_ceiling(tier)
All access logic flows through this single module.

Access Tiers (unified):
- PREF: Preferred formulary tier - high access, moderate GTN
- NONPREF: Non-preferred tier - moderate access, higher GTN erosion  
- PA_STEP: Prior auth + step therapy - restricted access, high GTN erosion

Example usage:
    >>> eff_market, net_price, ceiling = apply_access(tier, tam, list_price)
    >>> # ↓ adoption uses eff_market only - NO other constraints applied
    >>> units = bass_new_adopters(p, q, eff_market, T)
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

# SINGLE SOURCE OF TRUTH: Access tiers with all constraints
from dataclasses import dataclass

@dataclass(frozen=True)
class AccessTier:
    """Pharmaceutical market access tier with all constraints"""
    name: str
    gross_to_net: float     # GTN ratio (e.g., 0.72 = 28% discounts)
    ceiling: float          # Max penetration among eligible patients (0-1)
    description: str = ""   # Human-readable description

# Unified tier definitions - SINGLE SOURCE OF TRUTH
TIERS = {
    "PREF": AccessTier(
        name="PREF", 
        gross_to_net=0.78, 
        ceiling=0.65,
        description="Preferred formulary tier - high access, moderate GTN"
    ),
    "NONPREF": AccessTier(
        name="NONPREF", 
        gross_to_net=0.70, 
        ceiling=0.45,
        description="Non-preferred tier - moderate access, higher GTN erosion"
    ),
    "PA_STEP": AccessTier(
        name="PA_STEP", 
        gross_to_net=0.66, 
        ceiling=0.35,
        description="Prior auth + step therapy - restricted access, high GTN erosion"
    ),
}

# Backward compatibility mappings - populate with legacy tier names
# New unified tiers
ACCESS_TIERS = {tier.name: tier.ceiling for tier in TIERS.values()}
GTN_BY_TIER = {tier.name: tier.gross_to_net for tier in TIERS.values()}

# Legacy tier name mappings for backward compatibility
# Based on original test expectations: OPEN = no constraint, PA = moderate, NICHE = high constraint
ACCESS_TIERS.update({
    "OPEN": 1.0,  # No access constraints (100% ceiling)
    "PA": 0.6,    # Moderate access constraints (60% ceiling) 
    "NICHE": 0.3  # High access constraints (30% ceiling)
})

GTN_BY_TIER.update({
    "OPEN": 0.85,  # High GTN (minimal discounts)
    "PA": 0.65,    # Moderate GTN (standard discounts)
    "NICHE": 0.50  # Low GTN (high discounts for restricted access)
})

ADOPTION_CEILING_BY_TIER = ACCESS_TIERS

def map_price_to_tier(list_price_annual: float) -> str:
    """
    Map annual list price to access tier - SINGLE SOURCE OF TRUTH
    Following industry thresholds for biologics
    
    Args:
        list_price_annual: Annual list price in USD
        
    Returns:
        str: Access tier ('PREF', 'NONPREF', or 'PA_STEP')
        
    Example:
        >>> map_price_to_tier(45000)
        'PREF'
        >>> map_price_to_tier(65000)
        'NONPREF'
        >>> map_price_to_tier(85000)
        'PA_STEP'
    """
    # Standard biologic pricing thresholds (annual)
    if list_price_annual <= 45000:
        return "PREF"
    elif list_price_annual <= 75000:
        return "NONPREF" 
    else:
        return "PA_STEP"

# Backward compatibility function - preserves original monthly thresholds
def tier_from_price(list_price_month: float, 
                   thresholds: Optional[Dict[str, float]] = None) -> str:
    """
    Map monthly price to access tier - BACKWARD COMPATIBLE
    Preserves original monthly thresholds for existing tests.
    Uses legacy monthly thresholds from src/constants.py by default.
    
    Args:
        list_price_month: Monthly list price in USD
        thresholds: Custom thresholds {'open_max': float, 'pa_max': float}
        
    Returns:
        str: Legacy tier name ('OPEN', 'PA', 'NICHE')
    """
    # Import legacy thresholds from constants for backward compatibility
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from constants import PRICE_THRESHOLDS
    
    # Use custom thresholds if provided, otherwise use legacy PRICE_THRESHOLDS
    if thresholds:
        open_threshold = thresholds.get('open_max', PRICE_THRESHOLDS['OPEN'])
        pa_threshold = thresholds.get('pa_max', PRICE_THRESHOLDS['PA'])
    else:
        # Use original monthly thresholds from constants.py - BACKWARD COMPATIBLE
        open_threshold = PRICE_THRESHOLDS['OPEN']  # $1000/month
        pa_threshold = PRICE_THRESHOLDS['PA']      # $2500/month
    
    if list_price_month < open_threshold:
        return "OPEN"
    elif list_price_month < pa_threshold:
        return "PA" 
    else:
        return "NICHE"

def apply_access(pricing_tier: str, tam_patients: float, list_price: float) -> Tuple[float, float, float]:
    """
    Apply pharmaceutical access constraints - SINGLE SOURCE OF TRUTH
    
    Args:
        pricing_tier: Access tier ("PREF", "NONPREF", "PA_STEP")
        tam_patients: Total addressable market (patient count)
        list_price: Annual list price per patient
        
    Returns:
        Tuple of (effective_market, net_price, access_ceiling)
        
    This is the ONLY function that should calculate effective market size.
    All other code should use the returned effective_market directly.
    NO additional constraints should be applied after this.
    """
    
    if pricing_tier not in TIERS:
        raise ValueError(f"Unknown pricing tier: {pricing_tier}. Valid tiers: {list(TIERS.keys())}")
    
    tier = TIERS[pricing_tier]
    
    # Core constraint calculations - SINGLE SOURCE OF TRUTH
    net_price = list_price * tier.gross_to_net
    effective_market = tam_patients * tier.ceiling
    
    return effective_market, net_price, tier.ceiling

def gtn_from_tier(access_tier: str, 
                 custom_gtn: Optional[Dict[str, float]] = None) -> float:
    """
    Get gross-to-net percentage for access tier.
    Uses unified tier system internally, supports legacy tier names.
    
    Args:
        access_tier: Access tier ('OPEN', 'PA', 'NICHE', 'PREF', 'NONPREF', 'PA_STEP')
        custom_gtn: Custom GtN mapping (optional)
    
    Returns:
        float: Gross-to-net percentage
    """
    if custom_gtn:
        return custom_gtn.get(access_tier, 0.65)
    
    # Check if tier exists in GTN_BY_TIER (includes both new and legacy)
    if access_tier in GTN_BY_TIER:
        return GTN_BY_TIER[access_tier]
    
    # Default fallback
    return 0.65

def adoption_ceiling_from_tier(access_tier: str,
                              custom_ceilings: Optional[Dict[str, float]] = None) -> float:
    """
    Get adoption ceiling (market accessibility) for access tier.
    Uses unified tier system internally, supports legacy tier names.
    
    Args:
        access_tier: Access tier ('OPEN', 'PA', 'NICHE', 'PREF', 'NONPREF', 'PA_STEP')
        custom_ceilings: Custom ceiling mapping (optional)
    
    Returns:
        float: Adoption ceiling (0.0 to 1.0)
    """
    if custom_ceilings:
        return custom_ceilings.get(access_tier, 0.6)
    
    # Check if tier exists in ACCESS_TIERS (includes both new and legacy)
    if access_tier in ACCESS_TIERS:
        return ACCESS_TIERS[access_tier]
    
    # Default fallback
    return 0.6

def apply_access_constraints(adopters: np.ndarray, 
                           access_tier: str,
                           custom_ceilings: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Apply access tier constraints to adoption forecast.
    
    Args:
        adopters: Unconstrained adoption forecast
        access_tier: Access tier constraint
        custom_ceilings: Custom ceiling mapping (optional)
    
    Returns:
        np.ndarray: Constrained adoption forecast
        
    Example:
        >>> unconstrained = np.array([100, 200, 300, 250, 200])
        >>> constrained = apply_access_constraints(unconstrained, 'PA')
        >>> (constrained <= unconstrained).all()  # Always <= unconstrained
        True
    """
    ceiling = adoption_ceiling_from_tier(access_tier, custom_ceilings)
    
    # Apply ceiling to cumulative adoption
    cumulative = np.cumsum(adopters)
    market_size = cumulative[-1] if len(cumulative) > 0 else 0
    max_cumulative = market_size * ceiling
    
    # Constrain cumulative adoption
    constrained_cumulative = np.minimum(cumulative, max_cumulative)
    
    # Convert back to period-by-period adopters
    constrained_adopters = np.diff(np.concatenate([[0], constrained_cumulative]))
    
    return constrained_adopters

class AccessClassifier:
    """
    Optional ML classifier for predicting access favorability.
    
    This is a lightweight classifier that can be trained on payer policy data
    to predict whether a given price point will receive favorable access.
    """
    
    def __init__(self, random_state: int = 42):
        self.classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=random_state
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def fit(self, features: pd.DataFrame, access_tiers: List[str]) -> None:
        """
        Train classifier on payer policy data.
        
        Args:
            features: DataFrame with pricing/policy features
            access_tiers: List of access tier labels
        """
        # Encode access tiers
        encoded_tiers = self.label_encoder.fit_transform(access_tiers)
        
        # Fit classifier
        self.classifier.fit(features, encoded_tiers)
        self.is_fitted = True
    
    def predict_tier(self, features: pd.DataFrame) -> List[str]:
        """
        Predict access tier for new price points.
        
        Args:
            features: DataFrame with pricing/policy features
        
        Returns:
            List[str]: Predicted access tiers
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        # Predict and decode
        encoded_pred = self.classifier.predict(features)
        return self.label_encoder.inverse_transform(encoded_pred).tolist()
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict access tier probabilities.
        
        Args:
            features: DataFrame with pricing/policy features
        
        Returns:
            np.ndarray: Probability matrix (n_samples x n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        return self.classifier.predict_proba(features)

def create_pricing_scenario(price_range: Tuple[float, float],
                          n_points: int = 20,
                          custom_thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Create pricing scenario analysis across price range.
    
    Args:
        price_range: (min_price, max_price) tuple
        n_points: Number of price points to analyze
        custom_thresholds: Custom access tier thresholds
    
    Returns:
        pd.DataFrame: Pricing scenario results
        
    Example:
        >>> scenario = create_pricing_scenario((500, 4000), n_points=10)
        >>> scenario.columns.tolist()
        ['price_monthly', 'access_tier', 'adoption_ceiling', 'gtn_pct', 'net_price']
    """
    min_price, max_price = price_range
    prices = np.linspace(min_price, max_price, n_points)
    
    results = []
    for price in prices:
        tier = tier_from_price(price, custom_thresholds)
        ceiling = adoption_ceiling_from_tier(tier)
        gtn = gtn_from_tier(tier)
        net_price = price * gtn
        
        results.append({
            'price_monthly': price,
            'access_tier': tier,
            'adoption_ceiling': ceiling,
            'gtn_pct': gtn,
            'net_price': net_price,
            'net_price_annual': net_price * 12
        })
    
    return pd.DataFrame(results)

def optimize_price_access_tradeoff(adopters: np.ndarray,
                                 price_range: Tuple[float, float],
                                 n_points: int = 50) -> Dict[str, any]:
    """
    Find optimal price point balancing access and revenue.
    
    Args:
        adopters: Base adoption forecast (unconstrained)
        price_range: (min_price, max_price) to evaluate
        n_points: Number of price points to test
    
    Returns:
        Dict with optimal price point and revenue analysis
    """
    scenario_df = create_pricing_scenario(price_range, n_points)
    
    results = []
    for _, row in scenario_df.iterrows():
        # Apply access constraints
        constrained_adopters = apply_access_constraints(adopters, row['access_tier'])
        total_patients = np.sum(constrained_adopters)
        
        # Calculate annual revenue (assuming persistence)
        annual_revenue = total_patients * row['net_price_annual']
        
        results.append({
            'price': row['price_monthly'],
            'tier': row['access_tier'],
            'patients': total_patients,
            'revenue': annual_revenue,
            'revenue_per_patient': row['net_price_annual']
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal price (max revenue)
    optimal_idx = results_df['revenue'].idxmax()
    optimal_result = results_df.iloc[optimal_idx]
    
    return {
        'optimal_price': optimal_result['price'],
        'optimal_tier': optimal_result['tier'],
        'optimal_patients': optimal_result['patients'],
        'optimal_revenue': optimal_result['revenue'],
        'scenario_analysis': results_df
    }

def test_single_source_of_truth_for_access():
    """
    Test to prevent regression: ensure all access logic flows through apply_access()
    Following Linus principle: "Good code has no special cases"
    """
    tam_patients = 100000
    list_price = 60000  # Annual
    
    # Test unified tier system
    eff_market, net_price, ceiling = apply_access("NONPREF", tam_patients, list_price)
    
    # Verify single source of truth calculations
    expected_ceiling = TIERS["NONPREF"].ceiling
    expected_gtn = TIERS["NONPREF"].gross_to_net
    
    assert ceiling == expected_ceiling, f"Ceiling mismatch: {ceiling} != {expected_ceiling}"
    assert abs(net_price - (list_price * expected_gtn)) < 0.01, f"Net price mismatch"
    assert abs(eff_market - (tam_patients * expected_ceiling)) < 0.01, f"Effective market mismatch"
    
    # Test backward compatibility
    legacy_ceiling = adoption_ceiling_from_tier("PA")  # Legacy name
    legacy_gtn = gtn_from_tier("PA")  # Legacy name
    
    assert abs(legacy_ceiling - TIERS["NONPREF"].ceiling) < 0.01, "Legacy mapping failed"
    assert abs(legacy_gtn - TIERS["NONPREF"].gross_to_net) < 0.01, "Legacy GTN mapping failed"
    
    print("[SUCCESS] Single source of truth test passed - no special cases")
    return True

if __name__ == "__main__":
    # Test single source of truth first
    print("Testing unified access constraint logic:")
    test_single_source_of_truth_for_access()
    print()
    
    # Demo usage
    print("Price & Access Simulator Demo")
    print("=" * 40)
    
    # Test price-to-tier mapping
    test_prices = [750, 1500, 2800, 4000]
    for price in test_prices:
        tier = tier_from_price(price)
        ceiling = adoption_ceiling_from_tier(tier)
        gtn = gtn_from_tier(tier)
        print(f"${price:,}/month -> {tier} tier (ceiling: {ceiling:.0%}, GtN: {gtn:.0%})")
    
    print("\nAccess constraint example:")
    # Example adoption curve
    base_adopters = np.array([1000, 2000, 3000, 2500, 2000, 1500, 1000])
    
    for tier in ['OPEN', 'PA', 'NICHE']:
        constrained = apply_access_constraints(base_adopters, tier)
        print(f"{tier}: {np.sum(constrained):,.0f} total patients "
              f"({100 * np.sum(constrained) / np.sum(base_adopters):.0f}% of unconstrained)")
    
    print("\nUnified tier system test:")
    for tier in ['PREF', 'NONPREF', 'PA_STEP']:
        eff_market, net_price, ceiling = apply_access(tier, 100000, 60000)
        print(f"{tier}: {eff_market:,.0f} patients, ${net_price:,.0f} net price, {ceiling:.0%} ceiling")
    
    print("\nPricing scenario analysis:")
    scenario = create_pricing_scenario((500, 3500), n_points=8)
    print(scenario[['price_monthly', 'access_tier', 'adoption_ceiling', 'net_price']].round(0))
