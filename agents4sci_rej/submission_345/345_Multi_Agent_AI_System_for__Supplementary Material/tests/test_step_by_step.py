#!/usr/bin/env python3
"""
Test pipeline steps one by one to find exact hanging point
"""

import asyncio
import sys
from pathlib import Path

# Add ai_scientist to path
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)

async def test_step_by_step():
    """Test each step sequentially"""
    
    print("=== STEP-BY-STEP PIPELINE TEST ===")
    
    try:
        from gpt5_orchestrator import GPT5Orchestrator
        
        orchestrator = GPT5Orchestrator()
        query = "Should we develop a Tezspire competitor for pediatric severe asthma?"
        
        print(f"\n[STEP 1] Query parsing...")
        drug_info = await orchestrator._parse_query(query)
        print(f"[STEP 1] SUCCESS: {drug_info}")
        
        print(f"\n[STEP 2] Data collection orchestration...")
        print(f"[STEP 2] About to call _orchestrate_data_collection...")
        data_start_time = asyncio.get_event_loop().time()
        drug_data = await orchestrator._orchestrate_data_collection(drug_info)
        data_end_time = asyncio.get_event_loop().time()
        print(f"[STEP 2] SUCCESS: {len(str(drug_data))} chars in {data_end_time - data_start_time:.1f}s")
        
        print(f"\n[STEP 3] Data quality review...")
        print(f"[STEP 3] About to call _orchestrate_data_review...")
        review_start_time = asyncio.get_event_loop().time()
        data_review = await orchestrator._orchestrate_data_review(drug_data)
        review_end_time = asyncio.get_event_loop().time()
        print(f"[STEP 3] SUCCESS: {len(str(data_review))} chars in {review_end_time - review_start_time:.1f}s")
        
        print(f"\n[STEP 4] Market analysis...")
        print(f"[STEP 4] About to call _orchestrate_market_analysis...")
        market_start_time = asyncio.get_event_loop().time()
        market_analysis = await orchestrator._orchestrate_market_analysis(drug_data)
        market_end_time = asyncio.get_event_loop().time()
        print(f"[STEP 4] SUCCESS: {len(str(market_analysis))} chars in {market_end_time - market_start_time:.1f}s")
        
        print(f"\n[STEP 5] Multi-method forecasting...")
        print(f"[STEP 5] About to call _orchestrate_multi_method_forecast...")
        forecast_start_time = asyncio.get_event_loop().time()
        forecasts = await orchestrator._orchestrate_multi_method_forecast(drug_data, market_analysis)
        forecast_end_time = asyncio.get_event_loop().time()
        print(f"[STEP 5] SUCCESS: {len(str(forecasts))} chars in {forecast_end_time - forecast_start_time:.1f}s")
        
        print(f"\n[STEP 6] Harsh review...")
        print(f"[STEP 6] About to call _orchestrate_harsh_review...")
        harsh_start_time = asyncio.get_event_loop().time()
        review = await orchestrator._orchestrate_harsh_review(forecasts)
        harsh_end_time = asyncio.get_event_loop().time()
        print(f"[STEP 6] SUCCESS: {len(str(review))} chars in {harsh_end_time - harsh_start_time:.1f}s")
        
        print(f"\n*** ALL STEPS COMPLETED SUCCESSFULLY! ***")
        return True
        
    except Exception as e:
        print(f"\n*** PIPELINE FAILED: {e} ***")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Starting step-by-step pipeline test...")
    start_time = asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else 0
    
    try:
        result = asyncio.run(test_step_by_step())
        end_time = asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else 0
        duration = end_time - start_time if start_time else 0
        print(f"\nFinal result: {'SUCCESS' if result else 'FAILED'} in {duration:.1f}s")
    except Exception as e:
        print(f"MAIN ERROR: {e}")

if __name__ == "__main__":
    main()