"""
Chart generation for Streamlit app.

All matplotlib/plotting code isolated here.
Follows Linus principle: do one thing well.
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from typing import Dict, Any

def create_adoption_charts(adopters: np.ndarray, cumulative: np.ndarray, 
                         effective_market: float, quarters: np.ndarray):
    """
    Create adoption forecasting charts.
    
    Args:
        adopters: New adopters per quarter
        cumulative: Cumulative adoption
        effective_market: Market size with access constraints
        quarters: Quarter numbers
        
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # New adopters per quarter
    ax1.bar(quarters, adopters, alpha=0.7, color='steelblue')
    ax1.set_title('New Adopters per Quarter')
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('New Patients')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative adoption
    ax2.plot(quarters, cumulative, linewidth=3, color='darkgreen')
    ax2.axhline(y=effective_market, color='red', linestyle='--', alpha=0.7, label='Market Cap')
    ax2.set_title('Cumulative Adoption')
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Total Patients')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_revenue_chart(quarters: np.ndarray, revenue: np.ndarray, net_cashflows: np.ndarray):
    """
    Create revenue and cashflow chart.
    
    Args:
        quarters: Quarter numbers
        revenue: Quarterly revenue
        net_cashflows: Net cashflows
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(quarters, revenue/1e6, linewidth=2, color='green', label='Revenue')
    ax.plot(quarters, net_cashflows/1e6, linewidth=2, color='blue', label='Net Cash Flow')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('Quarterly Financials')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('$M')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def create_npv_histogram(npv_values: np.ndarray, npv_stats: Dict[str, float]):
    """
    Create NPV distribution histogram.
    
    Args:
        npv_values: Array of NPV values from Monte Carlo
        npv_stats: Statistics dict with p10, p50, p90
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Filter extreme values for better visualization
    npv_millions = npv_values / 1e6  # Convert to millions
    p1, p99 = np.percentile(npv_millions, [1, 99])
    filtered_npv = npv_millions[(npv_millions >= p1) & (npv_millions <= p99)]
    
    ax.hist(filtered_npv, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(npv_stats['p10']/1e6, color='red', linestyle='--', label='P10')
    ax.axvline(npv_stats['p50']/1e6, color='orange', linestyle='--', label='P50')
    ax.axvline(npv_stats['p90']/1e6, color='green', linestyle='--', label='P90')
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_title('NPV Distribution')
    ax.set_xlabel('NPV ($M)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def display_key_metrics(peak_quarter: int, total_adoption: float, 
                       penetration_rate: float, access_tier: str):
    """Display key adoption metrics in columns."""
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Peak Quarter", f"Q{peak_quarter}")
    with metric_cols[1]:
        st.metric("Total Adoption", f"{total_adoption:,.0f}")
    with metric_cols[2]:
        st.metric("Market Penetration", f"{penetration_rate:.1%}")
    with metric_cols[3]:
        st.metric("Access Tier", access_tier)

def display_financial_metrics(npv_value: float, total_revenue: float, peak_revenue: float):
    """Display financial summary metrics."""
    st.metric("NPV ($M)", f"${npv_value/1e6:.1f}")
    st.metric("Total Revenue ($M)", f"${total_revenue/1e6:.1f}")
    st.metric("Peak Quarterly Revenue ($M)", f"${peak_revenue/1e6:.1f}")

def display_monte_carlo_results(npv_stats: Dict[str, float], n_simulations: int, success_rate: float):
    """Display Monte Carlo results in columns."""
    col_mc1, col_mc2 = st.columns([1, 1])
    
    with col_mc1:
        st.markdown("#### NPV Distribution")
        st.metric("P10 ($M)", f"${npv_stats['p10']/1e6:.1f}")
        st.metric("P50 ($M)", f"${npv_stats['p50']/1e6:.1f}")
        st.metric("P90 ($M)", f"${npv_stats['p90']/1e6:.1f}")
        st.metric("Prob(NPV > 0)", f"{npv_stats['prob_positive']:.1%}")
    
    # Success metrics
    st.markdown(f"""
    **Simulation Summary:**
    - Successful simulations: {n_simulations:,}
    - Success rate: {success_rate:.1%}
    """)