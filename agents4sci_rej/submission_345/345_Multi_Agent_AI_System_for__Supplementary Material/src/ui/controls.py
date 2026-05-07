"""
UI controls and parameter widgets for Streamlit app.

Handles all sidebar controls and parameter input widgets.
Follows Linus principle: one thing well.
"""
import streamlit as st
from typing import Dict, Any
try:
    from constants import (
        DEFAULT_WACC, DEFAULT_COGS_PCT, DEFAULT_BASS_P, DEFAULT_BASS_Q,
        DEFAULT_MARKET_SIZE, DEFAULT_TIME_HORIZON_YEARS
    )
except ImportError:
    # Fallback values
    DEFAULT_WACC = 0.10
    DEFAULT_COGS_PCT = 0.15
    DEFAULT_BASS_P = 0.03
    DEFAULT_BASS_Q = 0.40
    DEFAULT_MARKET_SIZE = 1_200_000
    DEFAULT_TIME_HORIZON_YEARS = 10

def render_sidebar_controls(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render all sidebar parameter controls.
    
    Args:
        config: Default configuration values
        
    Returns:
        Dict with all parameter values
    """
    st.sidebar.header("📋 Model Parameters")
    
    # Market parameters
    st.sidebar.subheader("Market Sizing")
    market_size = st.sidebar.number_input(
        "Eligible Patients (TAM)", 
        value=config.get('market', {}).get('eligible_patients', DEFAULT_MARKET_SIZE),
        min_value=100_000,
        max_value=10_000_000,
        step=100_000,
        help="Total addressable market size"
    )
    
    # Bass model parameters
    st.sidebar.subheader("Adoption Dynamics")
    bass_p = st.sidebar.slider(
        "Innovation coefficient (p)", 
        min_value=0.005, 
        max_value=0.100, 
        value=config.get('bass', {}).get('p', DEFAULT_BASS_P),
        step=0.005,
        help="External influence (early adopters)"
    )
    
    bass_q = st.sidebar.slider(
        "Imitation coefficient (q)", 
        min_value=0.10, 
        max_value=0.80, 
        value=config.get('bass', {}).get('q', DEFAULT_BASS_Q),
        step=0.05,
        help="Internal influence (word-of-mouth)"
    )
    
    # Pricing parameters
    st.sidebar.subheader("Pricing & Access")
    list_price = st.sidebar.number_input(
        "Monthly List Price ($)", 
        value=config.get('price_access', {}).get('list_price_month_usd', 2950),
        min_value=500,
        max_value=10_000,
        step=250,
        help="Monthly list price before discounts"
    )
    
    # Financial parameters
    st.sidebar.subheader("Economics")
    wacc = st.sidebar.slider(
        "WACC (annual)", 
        min_value=0.05, 
        max_value=0.20, 
        value=config.get('economics', {}).get('wacc_annual', DEFAULT_WACC),
        step=0.01,
        format="%.2f",
        help="Weighted average cost of capital"
    )
    
    cogs_pct = st.sidebar.slider(
        "COGS (% of revenue)", 
        min_value=0.05, 
        max_value=0.30, 
        value=config.get('economics', {}).get('cogs_pct', DEFAULT_COGS_PCT),
        step=0.01,
        format="%.2f"
    )
    
    # Simulation parameters
    st.sidebar.subheader("Simulation")
    time_horizon = st.sidebar.selectbox(
        "Time Horizon (years)", 
        options=[5, 7, 10, 15],
        index=2,  # Default to 10 years
        help="Forecast horizon in years"
    )
    
    n_simulations = st.sidebar.selectbox(
        "Monte Carlo Runs",
        options=[1000, 5000, 10000],
        index=1,  # Default to 5000
        help="Number of Monte Carlo simulations"
    )
    
    # Return all parameters
    return {
        'market_size': market_size,
        'bass_p': bass_p,
        'bass_q': bass_q,
        'list_price': list_price,
        'wacc': wacc,
        'cogs_pct': cogs_pct,
        'time_horizon': time_horizon,
        'n_simulations': n_simulations
    }

def display_access_info(access_tier: str, gtn_pct: float, adoption_ceiling: float):
    """Display access tier information in sidebar."""
    st.sidebar.markdown(f"**Access Tier:** {access_tier}")
    st.sidebar.markdown(f"**Gross-to-Net:** {gtn_pct:.0%}")
    st.sidebar.markdown(f"**Adoption Ceiling:** {adoption_ceiling:.0%}")