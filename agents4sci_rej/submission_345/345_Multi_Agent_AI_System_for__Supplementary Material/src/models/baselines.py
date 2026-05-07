"""
Industry-standard baseline forecasting methods.
Not toy baselines - these are what consultants actually use.
Following Linus principle: Simple, proven methods that work.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def peak_sales_heuristic(drug_row: pd.Series) -> float:
    """
    Industry standard peak sales heuristic.
    Peak = Market_Size × Peak_Share × Annual_Price × Compliance × GTN
    
    This is literally what McKinsey/Bain use as their first-pass estimate.
    
    Args:
        drug_row: Row from launches.parquet
    
    Returns:
        float: Estimated peak sales in USD
    """
    
    # Extract parameters with sane defaults for missing data
    market_size = drug_row.get('eligible_patients_at_launch', 0)
    monthly_price = drug_row.get('list_price_month_usd_launch', 0)
    gtn = drug_row.get('net_gtn_pct_launch', 0)
    
    # Apply intelligent defaults based on therapeutic area if data missing
    therapeutic_area = drug_row.get('therapeutic_area', 'Unknown')
    
    # Default market sizes by therapeutic area (conservative estimates)
    if market_size <= 0:
        market_size_defaults = {
            'Oncology': 500000,      # 500K cancer patients
            'Immunology': 300000,    # 300K autoimmune patients  
            'Cardiovascular': 2000000, # 2M CV patients
            'Respiratory': 1000000,  # 1M respiratory patients
            'Neurology': 400000,     # 400K neuro patients
            'Rare Disease': 50000,   # 50K rare disease patients
            'Diabetes': 1500000      # 1.5M diabetes patients
        }
        market_size = market_size_defaults.get(therapeutic_area, 500000)
    
    # Default pricing by therapeutic area (monthly USD)
    if monthly_price <= 0:
        price_defaults = {
            'Oncology': 12000,       # $12K/month for cancer drugs
            'Immunology': 8000,      # $8K/month for autoimmune
            'Cardiovascular': 400,   # $400/month for CV drugs
            'Respiratory': 300,      # $300/month for respiratory
            'Neurology': 6000,       # $6K/month for CNS drugs
            'Rare Disease': 15000,   # $15K/month for rare disease
            'Diabetes': 200          # $200/month for diabetes
        }
        monthly_price = price_defaults.get(therapeutic_area, 5000)
    
    # Default GTN (gross-to-net) if missing
    if gtn <= 0:
        gtn = 0.7  # 70% GTN is industry standard
    
    annual_price = monthly_price * 12
    
    # Estimate peak share based on competition and access
    peak_share = estimate_peak_share(drug_row)
    
    # Industry standard compliance rate
    compliance = 0.70  # 70% adherence/persistence
    
    # Calculate peak sales
    peak_sales = market_size * peak_share * annual_price * compliance * gtn
    
    return peak_sales


def estimate_peak_share(drug_row: pd.Series) -> float:
    """
    Estimate peak market share based on drug characteristics.
    Uses industry heuristics from analog analysis.
    
    Args:
        drug_row: Row from launches.parquet
    
    Returns:
        float: Estimated peak market share (0-1)
    """
    
    # Base share by therapeutic area (industry benchmarks)
    ta_base_shares = {
        'Oncology': 0.25,  # Higher share due to limited options
        'Immunology': 0.20,
        'Cardiovascular': 0.15,
        'Endocrinology': 0.12,
        'Respiratory': 0.18,
        'Neurology': 0.15,
        'Rare Disease': 0.35,  # Higher share in rare disease
    }
    
    base_share = ta_base_shares.get(drug_row.get('therapeutic_area', ''), 0.15)
    
    # Adjust for competition
    competitors = drug_row.get('competitor_count_at_launch', 3)
    if competitors == 0:
        competition_mult = 1.5  # First in class
    elif competitors <= 2:
        competition_mult = 1.0  # Limited competition
    elif competitors <= 5:
        competition_mult = 0.7  # Moderate competition
    else:
        competition_mult = 0.4  # Crowded market
    
    # Adjust for access tier
    access_mult = {
        'OPEN': 1.2,  # Better access = higher share
        'PA': 0.8,    # Prior auth limits uptake
        'NICHE': 0.5  # Highly restricted
    }.get(drug_row.get('access_tier_at_launch', 'PA'), 0.8)
    
    # Adjust for efficacy
    efficacy = drug_row.get('clinical_efficacy_proxy', 0.7)
    efficacy_mult = 0.5 + efficacy  # 0.5-1.5x based on efficacy
    
    # Adjust for safety
    if drug_row.get('safety_black_box', False):
        safety_mult = 0.6  # Black box warning reduces share
    else:
        safety_mult = 1.0
    
    # Calculate final peak share
    peak_share = base_share * competition_mult * access_mult * efficacy_mult * safety_mult
    
    # Cap at realistic levels
    peak_share = min(peak_share, 0.6)  # No drug gets >60% share
    peak_share = max(peak_share, 0.01)  # Minimum 1% share
    
    return peak_share


def year2_naive(drug_row: pd.Series) -> float:
    """
    Simple year 2 revenue forecast.
    Industry rule of thumb: Year 2 = 25-35% of peak.
    
    Args:
        drug_row: Row from launches.parquet
    
    Returns:
        float: Estimated year 2 revenue in USD
    """
    
    peak = peak_sales_heuristic(drug_row)
    
    # Year 2 percentage of peak varies by therapeutic area
    ta_year2_pct = {
        'Oncology': 0.35,  # Faster uptake in oncology
        'Immunology': 0.30,
        'Cardiovascular': 0.25,  # Slower uptake
        'Endocrinology': 0.28,
        'Respiratory': 0.32,
        'Neurology': 0.22,  # Very slow uptake
        'Rare Disease': 0.40,  # Fast uptake in rare disease
    }
    
    year2_pct = ta_year2_pct.get(drug_row.get('therapeutic_area', ''), 0.30)
    
    # Adjust for access
    if drug_row.get('access_tier_at_launch') == 'OPEN':
        year2_pct *= 1.2
    elif drug_row.get('access_tier_at_launch') == 'NICHE':
        year2_pct *= 0.7
    
    return peak * year2_pct


def linear_trend_forecast(drug_row: pd.Series, years: int = 5) -> np.ndarray:
    """
    Simple linear growth to peak forecast.
    Baseline sanity check - drugs grow linearly to peak.
    
    Args:
        drug_row: Row from launches.parquet
        years: Number of years to forecast
    
    Returns:
        np.ndarray: Revenue forecast by year
    """
    
    peak = peak_sales_heuristic(drug_row)
    peak_year = 3  # Assume peak at year 3 (index 3, which is year 4)
    
    forecast = np.zeros(years)
    
    for i in range(years):
        if i == 0:
            # Launch year
            forecast[i] = peak * 0.05
        elif i <= peak_year:
            # Linear growth to peak
            forecast[i] = peak * (i / peak_year)
        else:
            # Flat after peak
            forecast[i] = peak
    
    return forecast


def market_share_evolution(drug_row: pd.Series, years: int = 5) -> np.ndarray:
    """
    Market share evolution model.
    Models share capture over time based on competitive dynamics.
    
    Args:
        drug_row: Row from launches.parquet
        years: Number of years to forecast
    
    Returns:
        np.ndarray: Market share by year (0-1)
    """
    
    peak_share = estimate_peak_share(drug_row)
    
    # Share evolution depends on competition
    competitors = drug_row.get('competitor_count_at_launch', 3)
    
    if competitors <= 1:
        # Fast share capture with limited competition
        share_curve = np.array([0.15, 0.45, 0.70, 0.85, 0.90])
    elif competitors <= 3:
        # Moderate share capture
        share_curve = np.array([0.10, 0.30, 0.55, 0.75, 0.85])
    else:
        # Slow share capture in crowded market
        share_curve = np.array([0.05, 0.20, 0.40, 0.60, 0.75])
    
    # Scale to peak share
    share_curve = share_curve * peak_share
    
    # Extend or truncate to requested years
    if len(share_curve) < years:
        # Pad with peak share
        share_curve = np.pad(share_curve, (0, years - len(share_curve)), 
                             constant_values=peak_share)
    else:
        share_curve = share_curve[:years]
    
    return share_curve


def simple_bass_forecast(drug_row: pd.Series, years: int = 5) -> np.ndarray:
    """
    Simplified Bass diffusion forecast.
    Uses industry-standard p and q values.
    
    Args:
        drug_row: Row from launches.parquet
        years: Number of years to forecast
    
    Returns:
        np.ndarray: Revenue forecast by year
    """
    
    # Market parameters
    market_size = drug_row['eligible_patients_at_launch']
    annual_price = drug_row['list_price_month_usd_launch'] * 12
    gtn = drug_row['net_gtn_pct_launch']
    compliance = 0.70
    
    # Bass parameters by therapeutic area
    ta_bass_params = {
        'Oncology': (0.05, 0.45),  # Higher p, q for oncology
        'Immunology': (0.04, 0.40),
        'Cardiovascular': (0.02, 0.35),  # Lower p for CV
        'Endocrinology': (0.03, 0.38),
        'Respiratory': (0.035, 0.42),
        'Neurology': (0.02, 0.30),  # Slow diffusion
        'Rare Disease': (0.08, 0.50),  # Fast diffusion in rare
    }
    
    p, q = ta_bass_params.get(drug_row.get('therapeutic_area', ''), (0.03, 0.40))
    
    # Adjust for access
    access_mult = {
        'OPEN': 1.0,
        'PA': 0.6,
        'NICHE': 0.3
    }.get(drug_row.get('access_tier_at_launch', 'PA'), 0.6)
    
    # Bass diffusion
    forecast = np.zeros(years)
    cumulative = 0
    
    # Prevent division by zero
    total_addressable = market_size * access_mult
    if total_addressable <= 0:
        return forecast  # Return zeros if no addressable market
    
    for t in range(years):
        if t == 0:
            # Launch year
            adopters = market_size * p * access_mult
        else:
            # Bass formula
            remaining = total_addressable - cumulative
            if remaining <= 0:
                adopters = 0
            else:
                adopters = remaining * (p + q * cumulative / total_addressable)
        
        cumulative += adopters
        revenue = adopters * annual_price * compliance * gtn
        forecast[t] = revenue
    
    return forecast


def ensemble_baseline(drug_row: pd.Series, years: int = 5) -> Dict[str, np.ndarray]:
    """
    Ensemble of all baseline methods.
    Returns individual forecasts and weighted average.
    
    Args:
        drug_row: Row from launches.parquet
        years: Number of years to forecast
    
    Returns:
        Dict with individual forecasts and ensemble
    """
    
    # Generate all baseline forecasts
    forecasts = {
        'peak_sales_linear': linear_trend_forecast(drug_row, years),
        'simple_bass': simple_bass_forecast(drug_row, years),
        'market_share': market_share_evolution(drug_row, years) * 
                        drug_row['eligible_patients_at_launch'] *
                        drug_row['list_price_month_usd_launch'] * 12 *
                        drug_row['net_gtn_pct_launch'] * 0.70
    }
    
    # Year 2 point estimate
    y2_estimate = year2_naive(drug_row)
    
    # Create year 2 anchored forecast
    y2_anchored = np.zeros(years)
    if years >= 2:
        y2_anchored[1] = y2_estimate
        peak = peak_sales_heuristic(drug_row)
        
        # Interpolate to peak
        for i in range(years):
            if i == 0:
                y2_anchored[i] = peak * 0.05
            elif i == 1:
                y2_anchored[i] = y2_estimate
            elif i < 4:
                # Interpolate to peak
                y2_anchored[i] = y2_estimate + (peak - y2_estimate) * (i - 1) / 2
            else:
                y2_anchored[i] = peak
        
        forecasts['year2_anchored'] = y2_anchored
    
    # Weighted ensemble (equal weights for simplicity)
    ensemble = np.mean(list(forecasts.values()), axis=0)
    forecasts['ensemble'] = ensemble
    
    return forecasts