#!/usr/bin/env python3
"""
Test each pipeline step systematically to find where it hangs
Following Linus: tackle the problem head-on
"""

import asyncio
import sys
from pathlib import Path

# Add ai_scientist to path
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)

async def test_each_pipeline_step():
    """Test each step of the pipeline to find where it hangs"""
    
    print("=== SYSTEMATIC PIPELINE TEST ===")
    
    try:
        from gpt5_orchestrator import GPT5Orchestrator
        
        orchestrator = GPT5Orchestrator()
        query = "Should we develop a Tezspire competitor for pediatric severe asthma?"
        
        print("\n[STEP 1] Testing query parsing...")
        drug_info = await orchestrator._parse_query(query)
        print(f"[STEP 1] SUCCESS: {drug_info}")
        
        print("\n[STEP 2] Testing data collection orchestration...")
        print("[STEP 2] About to call _orchestrate_data_collection...")
        drug_data = await orchestrator._orchestrate_data_collection(drug_info)
        print(f"[STEP 2] SUCCESS: {len(str(drug_data))} chars")
        
        print("\n[STEP 3] Testing data review...")
        print("[STEP 3] About to call _orchestrate_data_review...")
        data_review = await orchestrator._orchestrate_data_review(drug_data)
        print(f"[STEP 3] SUCCESS: {data_review}")
        
        print("\n[STEP 4] Testing market analysis...")
        print("[STEP 4] About to call _orchestrate_market_analysis...")
        market_analysis = await orchestrator._orchestrate_market_analysis(drug_data)
        print(f"[STEP 4] SUCCESS: {len(str(market_analysis))} chars")
        
        print("\n[STEP 5] Testing forecast generation...")
        print("[STEP 5] About to call _orchestrate_multi_method_forecast...")
        forecasts = await orchestrator._orchestrate_multi_method_forecast(drug_data, market_analysis)
        print(f"[STEP 5] SUCCESS: {len(str(forecasts))} chars")
        
        print("\n[STEP 6] Testing harsh review...")
        print("[STEP 6] About to call _orchestrate_harsh_review...")
        review = await orchestrator._orchestrate_harsh_review(forecasts)
        print(f"[STEP 6] SUCCESS: {review}")
        
        print("\n[SUCCESS] All pipeline steps completed!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed at current step: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Testing each pipeline step systematically...")
    try:
        result = asyncio.run(test_each_pipeline_step())
        print(f"\nFinal result: {'SUCCESS' if result else 'FAILED'}")
    except Exception as e:
        print(f"MAIN ERROR: {e}")

if __name__ == "__main__":
    main()