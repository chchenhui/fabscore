"""
Simple test of AI Agent functionality without external API dependencies.

This tests the core AI reasoning logic and parameter estimation
without requiring Anthropic API keys.
"""

import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from tools import CommercialForecastTools, AnalysisState


def test_ai_agent_core():
    """Test core AI agent functionality."""
    
    print("=== AI AGENT CORE FUNCTIONALITY TEST ===")
    print()
    
    # Initialize tools
    tools = CommercialForecastTools()
    
    # Test drug characteristics (pediatric severe asthma biologic)
    test_characteristics = {
        "name": "Tezspire Pediatric Competitor",
        "drug_type": "biologic",
        "indication_area": "respiratory", 
        "severity": "severe",
        "patient_population": "pediatric",
        "competitive_position": "me_too"
    }
    
    print("**TEST CHARACTERISTICS**")
    for key, value in test_characteristics.items():
        print(f"   {key}: {value}")
    print()
    
    # Test AI reasoning components
    print("🤖 **AI REASONING TESTS**")
    print()
    
    # 1. Market Sizing
    print("📊 **Step 1: Intelligent Market Sizing**")
    market_size = tools.intelligent_market_sizing(test_characteristics, tools.state)
    print(f"   Result: {market_size:,} patients")
    print()
    
    # 2. Adoption Parameters
    print("📈 **Step 2: Intelligent Adoption Parameters**")
    bass_p, bass_q = tools.intelligent_adoption_parameters(test_characteristics, tools.state)
    print(f"   Bass p (early adopters): {bass_p:.3f}")
    print(f"   Bass q (word-of-mouth): {bass_q:.3f}")
    print()
    
    # 3. Pricing Strategy
    print("💰 **Step 3: Intelligent Pricing**")
    pricing_info = tools.intelligent_pricing(test_characteristics, tools.state)
    print(f"   List Price: ${pricing_info['list_price']:,}/month")
    print(f"   Access Tier: {pricing_info['access_tier']}")
    print(f"   GTN: {pricing_info['gtn_pct']:.0%}")
    print(f"   Adoption Ceiling: {pricing_info['adoption_ceiling']:.0%}")
    print()
    
    # 4. Run Bass Analysis
    print("🔄 **Step 4: Bass Diffusion Analysis**")
    bass_results = tools.run_bass_analysis(
        market_size, bass_p, bass_q, pricing_info["adoption_ceiling"]
    )
    print(f"   Peak Quarter: Q{bass_results['peak_quarter']}")
    print(f"   Peak New Patients: {bass_results['peak_patients']:,.0f}")
    print(f"   Total Penetration: {bass_results['penetration_rate']:.1%}")
    print()
    
    # 5. Financial Analysis
    print("💹 **Step 5: Financial Analysis**")
    financial_results = tools.run_financial_analysis(bass_results["adopters"], pricing_info)
    print(f"   NPV: ${financial_results['npv']/1e9:.2f}B")
    print(f"   Total Revenue: ${financial_results['total_revenue']/1e6:.0f}M")
    print(f"   Peak Revenue: ${financial_results['peak_revenue']/1e6:.0f}M/quarter")
    print()
    
    # 6. Generate Recommendation
    print("⚖️ **Step 6: AI Investment Recommendation**")
    recommendation = tools.generate_recommendation(test_characteristics)
    print(f"   Decision: {recommendation['decision']}")
    print(f"   Rationale: {recommendation['rationale']}")
    print(f"   Confidence: {recommendation['confidence']}")
    print()
    
    # 7. Show AI Reasoning Trace
    print("🧠 **AI REASONING TRACE**")
    print("*This shows how AI 'thinks' about parameter choices:*")
    print()
    
    for i, reasoning in enumerate(tools.state.get_reasoning_summary(), 1):
        print(f"   {i}. {reasoning}")
    print()
    
    # 8. Success Summary
    npv_billions = recommendation["key_metrics"]["npv_billions"]
    success_rate = recommendation.get("key_metrics", {}).get("success_rate", "N/A")
    
    print("🎯 **DEMO SUMMARY**")
    print(f"   Query: Should we develop a pediatric severe asthma biologic?")
    print(f"   AI Decision: {recommendation['decision']}")
    print(f"   NPV: ${npv_billions:.1f}B")
    if success_rate != "N/A":
        print(f"   Risk: {success_rate:.0%} success rate")
    print(f"   Market: {market_size:,} patients")
    print()
    
    if recommendation['decision'] in ['STRONG GO', 'GO']:
        print("✅ **AI RECOMMENDS INVESTMENT**")
    elif recommendation['decision'] == 'CONDITIONAL GO':
        print("⚠️ **AI RECOMMENDS CONDITIONAL INVESTMENT**") 
    else:
        print("❌ **AI RECOMMENDS NO INVESTMENT**")
        
    print()
    print("🎉 **AI AGENT CORE TEST COMPLETED SUCCESSFULLY!**")
    
    return {
        "success": True,
        "recommendation": recommendation,
        "market_size": market_size,
        "npv_billions": npv_billions,
        "reasoning_trace": tools.state.get_reasoning_summary()
    }


if __name__ == "__main__":
    result = test_ai_agent_core()