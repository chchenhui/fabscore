"""
AI Reasoning Panel for Streamlit UI.

This module provides the AI agent interface and reasoning display
for the Commercial Forecast System.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add agent to path
agent_path = str(Path(__file__).parent.parent.parent / "agent")
ai_scientist_path = str(Path(__file__).parent.parent.parent / "ai_scientist")
export_path = str(Path(__file__).parent.parent / "export")
sys.path.insert(0, agent_path)
sys.path.insert(0, ai_scientist_path)
sys.path.insert(0, export_path)

from tools import CommercialForecastTools, AnalysisState
from real_ai_parser import RealAIParser
from pptx_generator import create_commercial_forecast_slide, generate_pptx_filename

# Import real pipeline components
import sys
models_path = str(Path(__file__).parent.parent / "models")
econ_path = str(Path(__file__).parent.parent / "econ")
access_path = str(Path(__file__).parent.parent / "access")
sys.path.insert(0, models_path)
sys.path.insert(0, econ_path)
sys.path.insert(0, access_path)

from bass import bass_adopters
from npv import calculate_cashflows, npv, monte_carlo_npv, explain_npv_drivers
from pricing_sim import apply_access


def render_ai_query_interface():
    """Render the AI query input interface."""
    
    st.markdown('<div class="step-header">🤖 AI-Powered Analysis</div>', unsafe_allow_html=True)
    st.markdown("*Ask the AI agent to analyze pharmaceutical investment opportunities*")
    
    # Predefined example queries
    example_queries = [
        "Should we develop a Tezspire competitor for pediatric severe asthma?",
        "What's the commercial potential for a Dupixent biosimilar?", 
        "Analyze launching a severe asthma biologic in the US market",
        "Is there opportunity for an oral asthma biologic?",
        "Evaluate a me-too respiratory biologic investment"
    ]
    
    # Query input options
    input_method = st.radio(
        "How would you like to ask your question?",
        ["Use example query", "Write custom query"]
    )
    
    if input_method == "Use example query":
        query = st.selectbox(
            "Select an example analysis:",
            example_queries
        )
    else:
        query = st.text_area(
            "Enter your pharmaceutical investment question:",
            placeholder="e.g., Should we develop a competitor to [drug] for [indication]?",
            height=100
        )
    
    # Analysis button
    analyze_button = st.button("🚀 Run AI Analysis", type="primary")
    
    return query, analyze_button


def display_ai_reasoning_trace(reasoning_trace: List[str]):
    """Display the AI reasoning process step by step."""
    
    st.markdown("### 🧠 AI Reasoning Process")
    st.markdown("*See how the AI thinks through the analysis:*")
    
    with st.expander("View AI Reasoning Steps", expanded=True):
        for i, step in enumerate(reasoning_trace, 1):
            if step.startswith("**") and step.endswith("**"):
                # Section headers
                st.markdown(f"**{step}**")
            else:
                # Reasoning details
                st.markdown(f"{step}")
    
    st.markdown("---")


def display_ai_recommendation(recommendation: Dict[str, Any]):
    """Display the AI's final investment recommendation."""
    
    decision = recommendation.get("decision", "No decision")
    rationale = recommendation.get("rationale", "No rationale provided")
    confidence = recommendation.get("confidence", "unknown")
    
    st.markdown("### 🎯 AI Investment Recommendation")
    
    # Color-code the decision
    if decision in ["STRONG GO", "GO"]:
        st.success(f"**{decision}** - Recommended Investment")
    elif decision == "CONDITIONAL GO":
        st.warning(f"**{decision}** - Conditional Investment")
    else:
        st.error(f"**{decision}** - Not Recommended")
    
    st.markdown(f"**Rationale:** {rationale}")
    st.markdown(f"**Confidence Level:** {confidence.upper()}")
    
    # Key metrics
    if "key_metrics" in recommendation:
        metrics = recommendation["key_metrics"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "NPV (P50)",
                f"${metrics.get('npv_p50', metrics.get('npv_billions', 0)):.1f}B",
                delta=None
            )
            # Show P10/P90 range
            p10 = metrics.get('npv_p10', 0)
            p90 = metrics.get('npv_p90', 0)
            st.caption(f"Range: ${p10:.1f}B to ${p90:.1f}B")
        
        with col2:
            prob_positive = metrics.get('prob_positive', metrics.get('success_rate', 0))
            st.metric(
                "Prob(NPV>0)", 
                f"{prob_positive:.0%}",
                delta=None
            )
            st.caption("Monte Carlo probability")
        
        with col3:
            market_size = metrics.get('market_size', 0)
            st.metric(
                "Market Size",
                f"{market_size:,}",
                delta=None
            )


def run_ai_analysis(query: str) -> Dict[str, Any]:
    """Run AI analysis on the given query - NOW WITH REAL AI!"""
    
    # Initialize AI tools and parser
    tools = CommercialForecastTools()
    ai_parser = RealAIParser()
    
    try:
        # REAL AI PARSING - No more keyword matching!
        parsed_query = ai_parser.parse_query(query)
        
        # Convert to characteristics format
        characteristics = {
            "name": parsed_query.drug_name,
            "drug_type": parsed_query.drug_type,
            "indication_area": parsed_query.indication_area,
            "severity": parsed_query.severity,
            "patient_population": parsed_query.patient_population,
            "competitive_position": parsed_query.competitive_position
        }
        
        # Add AI reasoning to state
        tools.state.log_reasoning(
            "AI Query Parsing",
            f"Confidence: {parsed_query.confidence:.1%}. {parsed_query.reasoning}",
            "high" if parsed_query.confidence > 0.8 else "medium"
        )
        
        # Run AI analysis steps
        market_size = tools.intelligent_market_sizing(characteristics, tools.state)
        p, q = tools.intelligent_adoption_parameters(characteristics, tools.state)
        pricing_info = tools.intelligent_pricing(characteristics, tools.state)
        
        # REAL PIPELINE: Bass → cashflow → NPV calculation
        tools.state.log_reasoning(
            "Bass Diffusion Modeling",
            f"Generating adoption curve with m={market_size:,}, p={p:.3f}, q={q:.3f}",
            "high"
        )
        
        # Generate Bass adoption curve (40 quarters = 10 years)
        T = 40
        list_price_monthly = pricing_info["list_price"]
        access_tier = pricing_info["access_tier"]
        
        # Map access tier to unified system
        access_mapping = {"OPEN": "PREF", "PA": "NONPREF", "NICHE": "PA_STEP"}
        unified_tier = access_mapping.get(access_tier, "NONPREF")
        
        # Apply access constraints using single source of truth
        effective_market, net_price_annual, ceiling = apply_access(unified_tier, market_size, list_price_monthly * 12)
        
        # Generate Bass adopters with constrained market
        bass_adopters_raw = bass_adopters(T, effective_market, p, q)
        
        tools.state.log_reasoning(
            "Access Constraints Applied",
            f"Effective market: {effective_market:,.0f} patients ({ceiling:.0%} of TAM). Net price: ${net_price_annual:,.0f}/year",
            "high"
        )
        
        # Calculate cashflows using real pipeline
        cashflow_params = {
            'adopters': bass_adopters_raw,
            'list_price_monthly': list_price_monthly,
            'gtn_pct': pricing_info.get('gtn_pct', 0.72),
            'cogs_pct': 0.15,  # Standard pharma COGS
            'sga_launch': 50_000_000,  # $50M quarterly launch spend
            'sga_decay_to_pct': 0.3,
            'adherence_rate': 0.80,
            'price_erosion_annual': 0.02
        }
        
        cashflow_result = calculate_cashflows(**cashflow_params)
        net_cashflows = cashflow_result['net_cashflows']
        
        # Calculate base NPV with pharmaceutical WACC
        wacc_annual = 0.12
        npv_result = npv(net_cashflows, wacc_annual)
        npv_billions = npv_result / 1e9  # Convert to billions
        
        # Run Monte Carlo NPV analysis for uncertainty quantification
        mc_params = {
            'adopters': bass_adopters_raw,
            'list_price_monthly': list_price_monthly,
            'gtn_pct': pricing_info.get('gtn_pct', 0.72),
            'cogs_pct': 0.15,
            'sga_launch': 50_000_000,
            'sga_decay_to_pct': 0.3,
            'adherence_rate': 0.80,
            'price_erosion_annual': 0.02,
            'wacc_annual': wacc_annual
        }
        
        # Define uncertainty parameters for Monte Carlo
        uncertainty_params = {
            'gtn_pct': 0.05,  # ±5% std dev
            'adherence_rate': 0.10,  # ±10% std dev  
            'list_price_monthly': list_price_monthly * 0.15,  # ±15% std dev
            'sga_launch': 15_000_000  # ±15M std dev
        }
        
        # Run Monte Carlo simulation
        mc_results = monte_carlo_npv(mc_params, uncertainty_params, n_simulations=1000, random_seed=42)
        
        # Extract Monte Carlo statistics
        npv_p10 = mc_results['npv']['p10'] / 1e9 if mc_results['npv']['p10'] else npv_billions
        npv_p50 = mc_results['npv']['p50'] / 1e9 if mc_results['npv']['p50'] else npv_billions
        npv_p90 = mc_results['npv']['p90'] / 1e9 if mc_results['npv']['p90'] else npv_billions
        prob_positive = mc_results['npv']['prob_positive'] if mc_results['npv']['prob_positive'] else (1.0 if npv_billions > 0 else 0.0)
        
        tools.state.log_reasoning(
            "Monte Carlo Analysis Complete",
            f"NPV: ${npv_billions:.2f}B (base case), P10/P50/P90: ${npv_p10:.2f}B/${npv_p50:.2f}B/${npv_p90:.2f}B, Prob(NPV>0): {prob_positive:.1%}",
            "high"
        )
        
        # Use Monte Carlo probability instead of heuristic
        success_rate = prob_positive
        
        # Generate recommendation
        recommendation = tools.generate_recommendation(characteristics)
        recommendation["key_metrics"] = {
            "npv_billions": npv_billions,
            "npv_p10": npv_p10,
            "npv_p50": npv_p50,
            "npv_p90": npv_p90,
            "success_rate": success_rate,
            "prob_positive": prob_positive,
            "market_size": market_size
        }
        
        return {
            "success": True,
            "query": query,
            "characteristics": characteristics,
            "recommendation": recommendation,
            "reasoning_trace": tools.state.get_reasoning_summary(),
            "parameters": {
                "market_size": market_size,
                "effective_market": effective_market,
                "bass_p": p,
                "bass_q": q,
                "pricing": pricing_info,
                "access_ceiling": ceiling,
                "cashflow_result": cashflow_result,
                "net_cashflows": net_cashflows.tolist(),  # For JSON serialization
                "wacc": wacc_annual
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


def display_ai_parameters(parameters: Dict[str, Any]):
    """Display the AI-estimated parameters from real pipeline."""
    
    st.markdown("### 🔧 Real Pipeline Parameters")
    st.markdown("*Parameters used in Bass diffusion → cashflow → NPV pipeline:*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Market Model:**")
        st.markdown(f"• TAM: {parameters['market_size']:,} patients")
        st.markdown(f"• Effective Market: {parameters.get('effective_market', 0):,.0f}")
        st.markdown(f"• Access Ceiling: {parameters.get('access_ceiling', 0):.0%}")
        st.markdown(f"• Bass p: {parameters['bass_p']:.3f}")
        st.markdown(f"• Bass q: {parameters['bass_q']:.3f}")
    
    with col2:
        st.markdown("**Pricing Model:**")
        pricing = parameters['pricing']
        st.markdown(f"• List Price: ${pricing['list_price']:,}/month")
        st.markdown(f"• Access Tier: {pricing['access_tier']}")
        st.markdown(f"• GTN: {pricing['gtn_pct']:.0%}")
        st.markdown(f"• WACC: {parameters.get('wacc', 0.12):.0%}")
        
    with col3:
        st.markdown("**Cashflow Model:**")
        cf_result = parameters.get('cashflow_result', {})
        if cf_result:
            total_revenue = sum(cf_result.get('revenue', []))
            total_costs = sum(cf_result.get('total_costs', []))
            st.markdown(f"• Total Revenue: ${total_revenue/1e6:.0f}M")
            st.markdown(f"• Total Costs: ${total_costs/1e6:.0f}M")
            st.markdown(f"• Peak Revenue: ${max(cf_result.get('revenue', [0]))/1e6:.0f}M")
        
        # Show adoption metrics
        net_cf = parameters.get('net_cashflows', [])
        if net_cf:
            peak_cf = max(net_cf) if net_cf else 0
            st.markdown(f"• Peak Cashflow: ${peak_cf/1e6:.0f}M")


def render_ai_demo_panel():
    """Render the complete AI demo panel."""
    
    # Query interface
    query, run_analysis = render_ai_query_interface()
    
    # Run analysis if button clicked
    if run_analysis and query.strip():
        with st.spinner("🤖 AI Agent analyzing your query..."):
            result = run_ai_analysis(query)
        
        if result["success"]:
            st.success("✅ AI Analysis Complete!")
            
            # Display reasoning trace
            display_ai_reasoning_trace(result["reasoning_trace"])
            
            # Display parameters
            display_ai_parameters(result["parameters"])
            
            # Display recommendation
            display_ai_recommendation(result["recommendation"])
            
            # PPTX Export Section
            st.markdown("---")
            st.markdown("### 📊 Export Results")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                export_pptx = st.button("📋 Download PPTX One-Pager", type="secondary")
            
            with col2:
                st.markdown("*Get a professional PowerPoint summary of this analysis*")
            
            if export_pptx:
                try:
                    with st.spinner("🔄 Generating PowerPoint..."):
                        pptx_buffer = create_commercial_forecast_slide(result)
                        filename = generate_pptx_filename(result["query"])
                    
                    st.download_button(
                        label="💾 Download PPTX File",
                        data=pptx_buffer.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
                    st.success("✅ PPTX ready for download!")
                    
                except Exception as e:
                    st.error(f"❌ PPTX generation failed: {str(e)}")
            
        else:
            st.error(f"❌ Analysis failed: {result['error']}")
    
    elif run_analysis and not query.strip():
        st.warning("⚠️ Please enter a query to analyze")


# Demo function for testing
if __name__ == "__main__":
    st.set_page_config(page_title="AI Demo", layout="wide")
    st.title("🤖 AI Commercial Forecast Agent Demo")
    
    render_ai_demo_panel()