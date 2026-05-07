"""
Simplified Streamlit app for AI-Powered Commercial Forecast Agent.

Clean, modular structure following Linus principles.
Each module does one thing well.
"""
import streamlit as st
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any
from datetime import datetime

# Add src to path for imports  
src_path = str(Path(__file__).parent)
sys.path.insert(0, src_path)

# Import our modules
try:
    from models.bass import bass_adopters, bass_cumulative
    from econ.npv import calculate_cashflows, npv, monte_carlo_npv  
    from access.pricing_sim import tier_from_price, gtn_from_tier, adoption_ceiling_from_tier
    from data.etl import load_config
    from ui.controls import render_sidebar_controls, display_access_info
    from ui.charts import (
        create_adoption_charts, create_revenue_chart, create_npv_histogram,
        display_key_metrics, display_financial_metrics, display_monte_carlo_results
    )
    from ui.ai_panel import render_ai_demo_panel
    from export.pptx_generator import create_commercial_forecast_slide, PPTX_AVAILABLE
    
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure Streamlit
st.set_page_config(
    page_title="Commercial Forecast Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    .metric-container { 
        background-color: #f0f2f6; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin: 0.5rem 0; 
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_default_config() -> Dict[str, Any]:
    """Load default configuration."""
    try:
        return load_config()
    except Exception as e:
        st.error(f"Error loading config: {e}")
        # Return minimal defaults
        return {
            'market': {'eligible_patients': 1200000},
            'bass': {'p': 0.03, 'q': 0.40},
            'price_access': {'list_price_month_usd': 2950},
            'economics': {'wacc_annual': 0.10, 'cogs_pct': 0.15, 'sga_launch_annual': 350000000}
        }

def main() -> None:
    """Main application entry point."""
    
    # Title and introduction
    st.title("🎯 Commercial Forecast Agent")
    st.markdown("""
    **AI-powered commercial forecasting for pharmaceutical products**  
    Interactive demo using Bass diffusion model + NPV analysis with Monte Carlo uncertainty.
    """)
    
    # Add AI Agent Interface
    st.markdown("---")
    st.markdown("## 🤖 AI Agent Interface")
    st.markdown("*Ask natural language questions and watch the AI reason through the analysis*")
    
    # Add AI panel in an expander
    with st.expander("🚀 **Try the AI Agent!** - Natural language pharmaceutical analysis", expanded=False):
        try:
            render_ai_demo_panel()
        except Exception as e:
            st.error(f"AI Agent currently unavailable: {e}")
            st.info("Continuing with parameter-based interface below...")
    
    st.markdown("---")
    st.markdown("## 📊 Manual Parameter Interface")
    st.markdown("*Traditional parameter-based analysis:*")
    
    # Load configuration and render controls
    config = load_default_config()
    params = render_sidebar_controls(config)
    
    # Calculate access tier info
    access_tier = tier_from_price(params['list_price'])
    gtn_pct = gtn_from_tier(access_tier)
    adoption_ceiling = adoption_ceiling_from_tier(access_tier)
    display_access_info(access_tier, gtn_pct, adoption_ceiling)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Adoption Forecast")
        
        # Calculate Bass adoption
        T = params['time_horizon'] * 4  # Convert to quarters
        effective_market = params['market_size'] * adoption_ceiling
        
        adopters = bass_adopters(T, effective_market, params['bass_p'], params['bass_q'])
        cumulative = bass_cumulative(T, effective_market, params['bass_p'], params['bass_q'])
        quarters = np.arange(1, T + 1)
        
        # Display adoption charts
        fig = create_adoption_charts(adopters, cumulative, effective_market, quarters)
        st.pyplot(fig)
        
        # Key metrics
        peak_quarter = np.argmax(adopters) + 1
        total_adoption = cumulative[-1]
        penetration_rate = total_adoption / params['market_size']
        
        st.markdown("### 📊 Key Adoption Metrics")
        display_key_metrics(peak_quarter, total_adoption, penetration_rate, access_tier)
    
    with col2:
        st.header("💰 Financial Summary")
        
        # Calculate cashflows
        try:
            cashflow_params = {
                'adopters': adopters,
                'list_price_monthly': params['list_price'],
                'gtn_pct': gtn_pct,
                'cogs_pct': params['cogs_pct'],
                'sga_launch': config['economics']['sga_launch_annual'] / 4,  # Quarterly
                'sga_decay_to_pct': 0.5,
                'adherence_rate': 0.85
            }
            
            cashflows = calculate_cashflows(**cashflow_params)
            net_cf = cashflows['net_cashflows']
            npv_value = npv(net_cf, params['wacc'])
            
            st.markdown("#### Base Case Results")
            display_financial_metrics(npv_value, cashflows['revenue'].sum(), cashflows['revenue'].max())
            
            # Revenue chart
            fig_rev = create_revenue_chart(quarters, cashflows['revenue'], net_cf)
            st.pyplot(fig_rev)
            
        except Exception as e:
            st.error(f"Error calculating financials: {e}")
    
    # Monte Carlo Analysis
    st.header("🎲 Monte Carlo Uncertainty Analysis")
    
    if st.button("Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Running {params['n_simulations']:,} simulations..."):
            try:
                # Set up Monte Carlo
                base_params = cashflow_params.copy()
                base_params['wacc_annual'] = params['wacc']
                
                uncertainty_params = {
                    'gtn_pct': 0.05,
                    'list_price_monthly': params['list_price'] * 0.1,
                    'adherence_rate': 0.1,
                }
                
                # Run simulation
                mc_results = monte_carlo_npv(
                    base_params=base_params,
                    uncertainty_params=uncertainty_params,
                    n_simulations=params['n_simulations'],
                    random_seed=42
                )
                
                # Display results
                col_mc1, col_mc2 = st.columns([1, 1])
                
                with col_mc1:
                    display_monte_carlo_results(
                        mc_results['npv'], 
                        mc_results['n_simulations'],
                        mc_results['success_rate']
                    )
                
                with col_mc2:
                    fig_hist = create_npv_histogram(mc_results['npv']['values'], mc_results['npv'])
                    st.pyplot(fig_hist)
                
            except Exception as e:
                st.error(f"Monte Carlo simulation failed: {e}")
    
    # Export section
    st.header("📤 Export Results")
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📊 Export Charts", type="secondary"):
            st.info("Chart export functionality coming soon!")
    
    with col_export2:
        if PPTX_AVAILABLE and st.button("🎯 Generate PowerPoint", type="secondary"):
            try:
                # Prepare analysis results for PPTX
                analysis_result = {
                    "query": f"Analyze {params.get('product_name', 'pharmaceutical product')} at ${params['list_price']}/month",
                    "recommendation": {
                        "decision": "GO" if npv_value > 0 else "NO GO",
                        "rationale": f"NPV of ${npv_value/1e9:.1f}B with {penetration_rate:.0%} market penetration",
                        "confidence": "High" if npv_value > 1e9 else "Medium"
                    },
                    "parameters": {
                        "list_price_monthly": params['list_price'],
                        "market_size": params['market_size'],
                        "bass_p": params['bass_p'],
                        "bass_q": params['bass_q'],
                        "wacc": params['wacc']
                    },
                    "characteristics": {
                        "access_tier": access_tier,
                        "peak_quarter": peak_quarter,
                        "total_adoption": total_adoption,
                        "peak_revenue": cashflows['revenue'].max() / 1e6,
                        "npv_billions": npv_value / 1e9
                    },
                    "npv_billions": npv_value / 1e9,
                    "peak_adoption_quarter": peak_quarter
                }
                
                # Generate PPTX
                pptx_buffer = create_commercial_forecast_slide(analysis_result)
                
                # Download button
                st.download_button(
                    label="📥 Download PowerPoint",
                    data=pptx_buffer.getvalue(),
                    file_name=f"commercial_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
                st.success("PowerPoint generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating PowerPoint: {e}")
        elif not PPTX_AVAILABLE:
            st.warning("PowerPoint export requires python-pptx. Install with: pip install python-pptx")
    
    with col_export3:
        if st.button("📝 Generate Report", type="secondary"):
            st.info("LaTeX report generation coming soon!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>AI-Powered Commercial Forecast Agent | Academic MVP Demo</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()