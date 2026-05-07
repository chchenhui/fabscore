"""
Simple test of AI Agent functionality without external API dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Set environment variable to use default config
os.environ['USE_DEFAULT_CONFIG'] = '1'

try:
    from tools import CommercialForecastTools, AnalysisState
    print("AI Tools imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_ai_reasoning():
    """Test core AI reasoning functionality."""
    
    print("\n=== AI AGENT REASONING TEST ===")
    
    # Initialize tools
    tools = CommercialForecastTools()
    
    # Test characteristics
    characteristics = {
        "name": "Pediatric Asthma Biologic",
        "drug_type": "biologic",
        "indication_area": "respiratory", 
        "severity": "severe",
        "patient_population": "pediatric"
    }
    
    print(f"\nTesting drug: {characteristics['name']}")
    print(f"Type: {characteristics['drug_type']}")
    print(f"Population: {characteristics['patient_population']} {characteristics['severity']} {characteristics['indication_area']}")
    
    # Test AI market sizing
    print("\n--- Step 1: AI Market Sizing ---")
    market_size = tools.intelligent_market_sizing(characteristics, tools.state)
    print(f"AI estimated market size: {market_size:,} patients")
    
    # Test AI adoption parameters
    print("\n--- Step 2: AI Adoption Parameters ---")
    p, q = tools.intelligent_adoption_parameters(characteristics, tools.state)
    print(f"AI estimated Bass p: {p:.3f}")
    print(f"AI estimated Bass q: {q:.3f}")
    
    # Test AI pricing
    print("\n--- Step 3: AI Pricing ---")
    pricing = tools.intelligent_pricing(characteristics, tools.state)
    print(f"AI estimated price: ${pricing['list_price']:,}/month")
    print(f"Access tier: {pricing['access_tier']}")
    
    # Show AI reasoning
    print("\n--- AI REASONING TRACE ---")
    for i, reasoning in enumerate(tools.state.get_reasoning_summary(), 1):
        print(f"{i}. {reasoning}")
    
    print("\n=== TEST COMPLETED SUCCESSFULLY ===")
    
    return {
        "market_size": market_size,
        "bass_p": p,
        "bass_q": q,
        "pricing": pricing,
        "reasoning": tools.state.get_reasoning_summary()
    }


if __name__ == "__main__":
    result = test_ai_reasoning()