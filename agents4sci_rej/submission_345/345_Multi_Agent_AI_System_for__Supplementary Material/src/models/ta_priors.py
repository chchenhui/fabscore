"""
Therapeutic Area (TA) priors for intelligent defaults.
When real data is missing, use industry-standard TA-based estimates.
Based on GPT-5 recommendations and industry benchmarks.
"""

from typing import Dict, Any, Tuple
import pandas as pd


def get_ta_pricing_prior(therapeutic_area: str, route: str = None, dosage_form: str = None) -> float:
    """
    Get monthly net price prior by therapeutic area and modality.
    
    Args:
        therapeutic_area: Primary therapeutic area
        route: Administration route (e.g., 'INTRAVENOUS', 'ORAL')
        dosage_form: Dosage form (e.g., 'SOLUTION', 'TABLET')
    
    Returns:
        Monthly net price in USD
    """
    
    # Determine modality from route/dosage_form
    is_biologic = (
        route and 'INTRAVENOUS' in route.upper() or
        dosage_form and any(form in dosage_form.upper() for form in ['SOLUTION', 'INJECTION'])
    )
    
    is_inhaled = (
        route and 'INHALATION' in route.upper() or
        dosage_form and any(form in dosage_form.upper() for form in ['AEROSOL', 'INHALATION'])
    )
    
    # TA-specific pricing priors (monthly USD)
    ta_pricing = {
        'Oncology': {
            'biologic': 10500,  # $9-12k range, use midpoint
            'oral': 8000,       # Oral oncology still expensive
            'default': 9000
        },
        'Immunology': {
            'biologic': 6000,   # $5-7k range
            'oral': 4000,       # JAK inhibitors, etc.
            'default': 5500
        },
        'Cardiovascular': {
            'biologic': 2000,   # Few CV biologics
            'oral': 275,        # $150-400 range
            'default': 300
        },
        'Diabetes': {
            'biologic': 800,    # GLP-1s: $700-900
            'oral': 150,        # Metformin, etc.
            'default': 400
        },
        'Respiratory': {
            'biologic': 1500,   # Few respiratory biologics
            'inhaled': 300,     # $200-400 range
            'oral': 200,
            'default': 250
        },
        'Neurology': {
            'biologic': 5000,   # MS drugs, etc.
            'oral': 3000,
            'default': 4000
        },
        'Ophthalmology': {
            'biologic': 2000,   # Anti-VEGF
            'oral': 500,
            'default': 1200
        },
        'Endocrinology': {
            'biologic': 800,    # Similar to diabetes
            'oral': 200,
            'default': 400
        }
    }
    
    # Get TA-specific pricing
    ta_prices = ta_pricing.get(therapeutic_area, ta_pricing['Immunology'])  # Default to immunology
    
    # Select by modality
    if is_biologic:
        return ta_prices.get('biologic', ta_prices['default'])
    elif is_inhaled:
        return ta_prices.get('inhaled', ta_prices['default'])
    else:
        return ta_prices.get('oral', ta_prices['default'])


def get_ta_gtn_prior(therapeutic_area: str) -> float:
    """
    Get gross-to-net (GTN) prior by therapeutic area.
    
    Args:
        therapeutic_area: Primary therapeutic area
    
    Returns:
        GTN percentage (0-1)
    """
    
    ta_gtn = {
        'Oncology': 0.70,        # 65-75% range
        'Immunology': 0.60,      # 55-65% range
        'Cardiovascular': 0.525, # 45-60% range
        'Diabetes': 0.525,       # 45-60% range
        'Respiratory': 0.55,     # 50-60% range
        'Neurology': 0.65,       # Higher for specialty
        'Ophthalmology': 0.60,
        'Endocrinology': 0.525,
        'Rare Disease': 0.75     # Less discounting for rare
    }
    
    return ta_gtn.get(therapeutic_area, 0.60)  # Default 60%


def get_ta_compliance_prior(therapeutic_area: str, route: str = None, dosage_form: str = None) -> float:
    """
    Get compliance/adherence prior by TA and modality.
    
    Args:
        therapeutic_area: Primary therapeutic area
        route: Administration route
        dosage_form: Dosage form
    
    Returns:
        Compliance rate (0-1)
    """
    
    # Determine modality
    is_biologic = (
        route and 'INTRAVENOUS' in route.upper() or
        dosage_form and any(form in dosage_form.upper() for form in ['SOLUTION', 'INJECTION'])
    )
    
    is_inhaled = (
        route and 'INHALATION' in route.upper() or
        dosage_form and any(form in dosage_form.upper() for form in ['AEROSOL', 'INHALATION'])
    )
    
    # Base compliance by modality
    if is_biologic:
        base_compliance = 0.725  # 65-80% range for biologics
    elif is_inhaled:
        base_compliance = 0.575  # 50-65% range for inhaled
    else:
        base_compliance = 0.675  # 60-75% range for oral
    
    # TA adjustments
    ta_adjustments = {
        'Oncology': 1.1,        # Higher compliance for cancer
        'Rare Disease': 1.15,   # Highest compliance
        'Immunology': 1.0,      # Baseline
        'Cardiovascular': 0.9,  # Lower compliance for chronic
        'Diabetes': 0.85,       # Notoriously poor compliance
        'Respiratory': 0.8,     # Poor inhaler compliance
        'Neurology': 0.95,
        'Ophthalmology': 1.05,
        'Endocrinology': 0.9
    }
    
    adjustment = ta_adjustments.get(therapeutic_area, 1.0)
    
    # Apply adjustment and cap
    compliance = base_compliance * adjustment
    return min(max(compliance, 0.4), 0.9)  # Cap between 40-90%


def get_ta_market_size_prior(therapeutic_area: str) -> int:
    """
    Get eligible patient population prior by TA.
    
    Args:
        therapeutic_area: Primary therapeutic area
    
    Returns:
        Eligible patient count (addressable population)
    """
    
    ta_populations = {
        'Oncology': 500000,      # 500K cancer patients per indication
        'Immunology': 300000,    # 300K autoimmune patients
        'Cardiovascular': 2000000, # 2M CV patients
        'Respiratory': 1000000,  # 1M respiratory patients
        'Neurology': 400000,     # 400K neuro patients
        'Diabetes': 1500000,     # 1.5M diabetes patients
        'Ophthalmology': 800000, # 800K eye patients
        'Endocrinology': 600000, # 600K endocrine patients
        'Rare Disease': 50000    # 50K rare disease patients
    }
    
    return ta_populations.get(therapeutic_area, 500000)  # Default 500K


def get_ta_y2_share_prior(therapeutic_area: str) -> float:
    """
    Get Year 2 market share prior by TA.
    
    Args:
        therapeutic_area: Primary therapeutic area
    
    Returns:
        Year 2 market share (0-1)
    """
    
    ta_y2_shares = {
        'Oncology': 0.075,       # 5-10% range -> 7.5%
        'Immunology': 0.065,     # 5-8% range -> 6.5%
        'Cardiovascular': 0.02,  # 1-3% range -> 2%
        'Diabetes': 0.035,       # 2-5% range -> 3.5%
        'Respiratory': 0.03,     # 2-4% range -> 3%
        'Neurology': 0.04,       # Specialty area
        'Ophthalmology': 0.05,   # Niche markets
        'Endocrinology': 0.035,
        'Rare Disease': 0.15     # Higher share in rare disease
    }
    
    return ta_y2_shares.get(therapeutic_area, 0.05)  # Default 5%


def apply_ta_priors(drug_row: pd.Series, imputation_log: Dict[str, Any] = None) -> pd.Series:
    """
    Apply TA priors to fill missing drug characteristics.
    
    Args:
        drug_row: Drug data row (will be modified in place)
        imputation_log: Optional dict to log imputation details
    
    Returns:
        Updated drug row with priors applied
    """
    
    if imputation_log is None:
        imputation_log = {}
    
    drug_row = drug_row.copy()
    therapeutic_area = drug_row.get('therapeutic_area', 'Immunology')
    route = drug_row.get('route', '')
    dosage_form = drug_row.get('dosage_form', '')
    
    # Apply pricing prior if missing
    if drug_row.get('list_price_month_usd_launch', 0) <= 0:
        prior_price = get_ta_pricing_prior(therapeutic_area, route, dosage_form)
        drug_row['list_price_month_usd_launch'] = prior_price
        imputation_log['price_imputed'] = True
        imputation_log['price_prior_value'] = prior_price
        imputation_log['price_prior_basis'] = f"TA={therapeutic_area}, route={route}"
    
    # Apply GTN prior if missing
    if drug_row.get('net_gtn_pct_launch', 0) <= 0:
        prior_gtn = get_ta_gtn_prior(therapeutic_area)
        drug_row['net_gtn_pct_launch'] = prior_gtn
        imputation_log['gtn_imputed'] = True
        imputation_log['gtn_prior_value'] = prior_gtn
        imputation_log['gtn_prior_basis'] = f"TA={therapeutic_area}"
    
    # Apply market size prior if missing
    if drug_row.get('eligible_patients_at_launch', 0) <= 0:
        prior_market = get_ta_market_size_prior(therapeutic_area)
        drug_row['eligible_patients_at_launch'] = prior_market
        imputation_log['market_size_imputed'] = True
        imputation_log['market_size_prior_value'] = prior_market
        imputation_log['market_size_prior_basis'] = f"TA={therapeutic_area}"
    
    # Apply efficacy prior if missing or zero (CRITICAL: never allow 0)
    if drug_row.get('clinical_efficacy_proxy', 0) <= 0:
        ta_efficacy = {
            'Oncology': 0.70,      # Good efficacy for approved oncology drugs
            'Immunology': 0.65,    # Moderate-good efficacy
            'Cardiovascular': 0.55, # Moderate efficacy
            'Diabetes': 0.60,      # Moderate-good efficacy  
            'Respiratory': 0.55,   # Moderate efficacy
            'Neurology': 0.65,     # Moderate-good efficacy
            'Ophthalmology': 0.70, # Good efficacy typical
            'Endocrinology': 0.60,
            'Rare Disease': 0.75   # High efficacy for rare disease
        }
        prior_efficacy = ta_efficacy.get(therapeutic_area, 0.65)
        drug_row['clinical_efficacy_proxy'] = prior_efficacy
        imputation_log['efficacy_imputed'] = True
        imputation_log['efficacy_prior_value'] = prior_efficacy
        imputation_log['efficacy_prior_basis'] = f"TA={therapeutic_area}"
    
    # Apply access tier prior if missing (CRITICAL: never allow empty)
    if not drug_row.get('access_tier_at_launch') or str(drug_row.get('access_tier_at_launch', '')).strip() == '':
        ta_access = {
            'Oncology': 'PA',      # Usually prior authorization
            'Immunology': 'PA',    # Usually prior authorization  
            'Cardiovascular': 'OPEN', # Often open access
            'Diabetes': 'OPEN',    # Often open access
            'Respiratory': 'PA',   # Mixed, default PA
            'Neurology': 'PA',     # Usually specialty
            'Ophthalmology': 'PA', # Specialty access
            'Endocrinology': 'PA', # Usually specialty
            'Rare Disease': 'NICHE' # Highly restricted
        }
        prior_access = ta_access.get(therapeutic_area, 'PA')
        drug_row['access_tier_at_launch'] = prior_access
        imputation_log['access_imputed'] = True
        imputation_log['access_prior_value'] = prior_access
        imputation_log['access_prior_basis'] = f"TA={therapeutic_area}"
    
    # Add compliance (CRITICAL: never 0, always add even if exists)
    compliance = get_ta_compliance_prior(therapeutic_area, route, dosage_form)
    drug_row['compliance_prior'] = compliance
    imputation_log['compliance_prior_value'] = compliance
    imputation_log['compliance_prior_basis'] = f"TA={therapeutic_area}, modality derived"
    
    # Add Y2 share prior (CRITICAL: never 0, always add)
    y2_share = get_ta_y2_share_prior(therapeutic_area)
    drug_row['y2_share_prior'] = y2_share
    imputation_log['y2_share_prior_value'] = y2_share
    imputation_log['y2_share_prior_basis'] = f"TA={therapeutic_area}"
    
    return drug_row


def get_ta_analog_weights(therapeutic_area: str) -> Dict[str, float]:
    """
    Get analog similarity weights by therapeutic area.
    Helps analog matching when direct analogs aren't available.
    
    Args:
        therapeutic_area: Primary therapeutic area
    
    Returns:
        Dict of TA -> similarity weight
    """
    
    # Define TA similarity clusters
    ta_clusters = {
        'Oncology': {
            'Oncology': 1.0,
            'Immunology': 0.7,    # Some overlap (checkpoint inhibitors)
            'Rare Disease': 0.6,  # Often orphan oncology
            'Neurology': 0.3,
            'Cardiovascular': 0.1,
            'Diabetes': 0.1,
            'Respiratory': 0.1,
            'Ophthalmology': 0.2,
            'Endocrinology': 0.1
        },
        'Immunology': {
            'Immunology': 1.0,
            'Oncology': 0.7,      # Checkpoint inhibitors
            'Neurology': 0.6,     # MS drugs
            'Rare Disease': 0.5,
            'Cardiovascular': 0.3,
            'Respiratory': 0.4,   # Asthma overlap
            'Diabetes': 0.2,
            'Ophthalmology': 0.3,
            'Endocrinology': 0.3
        },
        'Cardiovascular': {
            'Cardiovascular': 1.0,
            'Diabetes': 0.8,      # Strong overlap
            'Endocrinology': 0.6,
            'Neurology': 0.4,
            'Respiratory': 0.3,
            'Immunology': 0.3,
            'Oncology': 0.1,
            'Ophthalmology': 0.2,
            'Rare Disease': 0.2
        },
        'Diabetes': {
            'Diabetes': 1.0,
            'Cardiovascular': 0.8,
            'Endocrinology': 0.7,
            'Immunology': 0.2,
            'Respiratory': 0.2,
            'Neurology': 0.3,
            'Oncology': 0.1,
            'Ophthalmology': 0.2,
            'Rare Disease': 0.3
        }
    }
    
    # Add default patterns for any TA not explicitly defined
    default_weights = {
        'same_ta': 1.0,
        'related_specialty': 0.6,
        'different_specialty': 0.3,
        'unrelated': 0.1
    }
    
    return ta_clusters.get(therapeutic_area, {therapeutic_area: 1.0})


def get_ta_peak_share_prior(therapeutic_area: str) -> float:
    """
    Get peak market share prior by TA.
    
    Args:
        therapeutic_area: Primary therapeutic area
    
    Returns:
        Peak market share (0-1)
    """
    
    ta_peak_shares = {
        'Oncology': 0.20,        # 15-25% range -> 20%
        'Immunology': 0.18,      # 15-20% range -> 18%
        'Cardiovascular': 0.08,  # 5-10% range -> 8%
        'Diabetes': 0.12,        # 8-15% range -> 12%
        'Respiratory': 0.10,     # 8-12% range -> 10%
        'Neurology': 0.15,       # Specialty area
        'Ophthalmology': 0.25,   # Niche markets, higher share
        'Endocrinology': 0.12,
        'Rare Disease': 0.40     # Much higher share in rare disease
    }
    
    return ta_peak_shares.get(therapeutic_area, 0.15)  # Default 15%


def get_ta_access_tier_multiplier(therapeutic_area: str) -> Dict[str, float]:
    """
    Get access tier impact multipliers by TA.
    
    Args:
        therapeutic_area: Primary therapeutic area
    
    Returns:
        Dict mapping access tier to revenue multiplier
    """
    
    # Base multipliers (can vary by TA in future)
    base_multipliers = {
        'OPEN': 1.0,    # Full access
        'PA': 0.65,     # Prior authorization reduces uptake
        'NICHE': 0.35   # Highly restricted access
    }
    
    # TA-specific adjustments could go here
    ta_adjustments = {
        'Oncology': {'PA': 0.70, 'NICHE': 0.40},  # Less impact in oncology
        'Rare Disease': {'PA': 0.75, 'NICHE': 0.50},  # Less impact in rare
        'Cardiovascular': {'PA': 0.55, 'NICHE': 0.25},  # More impact in CV
    }
    
    if therapeutic_area in ta_adjustments:
        multipliers = base_multipliers.copy()
        multipliers.update(ta_adjustments[therapeutic_area])
        return multipliers
    
    return base_multipliers


def get_ta_growth_rate(therapeutic_area: str) -> float:
    """
    Get annual growth rate for simple extrapolation by TA.
    
    Args:
        therapeutic_area: Primary therapeutic area
    
    Returns:
        Annual growth rate (e.g., 1.15 = 15% growth)
    """
    
    ta_growth_rates = {
        'Oncology': 1.25,        # High growth in oncology
        'Immunology': 1.20,      # High growth
        'Cardiovascular': 1.08,  # Slower growth, mature market
        'Diabetes': 1.10,        # Moderate growth
        'Respiratory': 1.12,     # Moderate growth
        'Neurology': 1.18,       # Good growth for specialty
        'Ophthalmology': 1.15,   # Moderate growth
        'Endocrinology': 1.12,
        'Rare Disease': 1.30     # Highest growth in rare disease
    }
    
    return ta_growth_rates.get(therapeutic_area, 1.15)  # Default 15%


def get_ta_factor_caps(therapeutic_area: str) -> Dict[str, Tuple[float, float]]:
    """
    Get min/max caps for adjustment factors by TA.
    
    Args:
        therapeutic_area: Primary therapeutic area
    
    Returns:
        Dict with (min, max) tuples for each factor type
    """
    
    # Conservative caps - prevent extreme adjustments
    base_caps = {
        'size_factor': (0.1, 10.0),      # 10x range for market size
        'price_factor': (0.2, 5.0),      # 5x range for price  
        'gtn_factor': (0.4, 2.0),        # 2x range for GTN
        'access_factor': (0.3, 3.0),     # 3x range for access
        'efficacy_factor': (0.5, 2.0)    # 2x range for efficacy
    }
    
    # Could customize by TA in future
    return base_caps


def get_ta_prediction_interval_multipliers(therapeutic_area: str) -> Tuple[float, float]:
    """
    Get prediction interval multipliers by TA.
    
    Args:
        therapeutic_area: Primary therapeutic area
    
    Returns:
        Tuple of (lower_mult, upper_mult) for prediction intervals
    """
    
    ta_uncertainty = {
        'Oncology': (0.60, 1.50),        # Moderate uncertainty
        'Immunology': (0.65, 1.45),      # Moderate uncertainty
        'Cardiovascular': (0.75, 1.25),  # Lower uncertainty, mature market
        'Diabetes': (0.70, 1.35),        # Moderate uncertainty
        'Respiratory': (0.70, 1.40),     # Moderate uncertainty
        'Neurology': (0.55, 1.60),       # Higher uncertainty, complex market
        'Ophthalmology': (0.65, 1.45),   # Moderate uncertainty
        'Endocrinology': (0.70, 1.35),
        'Rare Disease': (0.40, 2.00)     # High uncertainty
    }
    
    return ta_uncertainty.get(therapeutic_area, (0.65, 1.45))  # Default


def get_ta_minimum_revenue_threshold(therapeutic_area: str) -> float:
    """
    Get minimum revenue threshold for analog normalization by TA.
    
    Args:
        therapeutic_area: Primary therapeutic area
    
    Returns:
        Minimum revenue in USD for valid normalization base
    """
    
    ta_thresholds = {
        'Oncology': 10e6,         # $10M minimum 
        'Immunology': 8e6,        # $8M minimum
        'Cardiovascular': 15e6,   # $15M minimum (large market)
        'Diabetes': 12e6,         # $12M minimum
        'Respiratory': 10e6,      # $10M minimum
        'Neurology': 5e6,         # $5M minimum (specialty)
        'Ophthalmology': 3e6,     # $3M minimum (niche)
        'Endocrinology': 8e6,     # $8M minimum
        'Rare Disease': 2e6       # $2M minimum (small markets)
    }
    
    return ta_thresholds.get(therapeutic_area, 5e6)  # Default $5M


__all__ = [
    'get_ta_pricing_prior',
    'get_ta_gtn_prior', 
    'get_ta_compliance_prior',
    'get_ta_market_size_prior',
    'get_ta_y2_share_prior',
    'get_ta_peak_share_prior',
    'get_ta_access_tier_multiplier',
    'get_ta_growth_rate',
    'get_ta_factor_caps',
    'get_ta_prediction_interval_multipliers',
    'get_ta_minimum_revenue_threshold',
    'apply_ta_priors',
    'get_ta_analog_weights'
]