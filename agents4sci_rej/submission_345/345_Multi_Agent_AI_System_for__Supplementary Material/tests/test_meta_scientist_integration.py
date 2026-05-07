"""
Test integrated Meta-Scientist with multi-LLM ModelRouter
Verifies real AI hypothesis generation works
"""

import sys
from pathlib import Path

# Add ai_scientist to path
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
sys.path.insert(0, ai_scientist_path)

from meta_scientist import MetaScientist

def test_meta_scientist_integration():
    """Test Meta-Scientist with real multi-LLM integration"""
    
    print("=== META-SCIENTIST INTEGRATION TEST ===")
    
    # Initialize Meta-Scientist
    meta_scientist = MetaScientist()
    
    # Test hypothesis generation (this should use GPT-5 via router)
    print("\n[TEST 1] AI Hypothesis Generation...")
    hypotheses = meta_scientist.generate_research_hypotheses()
    
    print(f"\nGenerated {len(hypotheses)} hypotheses:")
    for i, h in enumerate(hypotheses, 1):
        print(f"\n{i}. {h.question}")
        print(f"   Method A: {h.method_a}")
        print(f"   Method B: {h.method_b}")
        print(f"   Expected: {h.expected_outcome}")
        print(f"   Confidence: {h.confidence}")
        print(f"   Type: {h.type.value}")
    
    # Verify AI was used (not fallback)
    if meta_scientist.api_calls > 0:
        print(f"\n[SUCCESS] Real AI used!")
        print(f"   API calls: {meta_scientist.api_calls}")
        print(f"   Tokens processed: {meta_scientist.token_count}")
        
        if meta_scientist.router:
            # Show usage report
            report = meta_scientist.router.get_usage_report()
            print(f"   Total cost: ${report['total_cost_cents']/100:.4f}")
            print(f"   Providers used: {list(report['by_provider'].keys()) if report['by_provider'] else 'None'}")
        
        return True
    else:
        print(f"\n[WARNING] Fallback mode used (no real AI)")
        return False

if __name__ == "__main__":
    success = test_meta_scientist_integration()
    
    if success:
        print("\n[READY] Meta-Scientist with real AI integration is working!")
        print("Ready for autonomous hypothesis generation and experimentation.")
    else:
        print("\n[ISSUES] Meta-Scientist needs API configuration.")
        print("Check .env file and API keys.")