#!/usr/bin/env python3
"""
Quick test to verify forecast extraction fix
"""

import asyncio
import sys
from pathlib import Path

# Add paths
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)

async def test_forecast_extraction():
    """Test forecast extraction from orchestrator"""
    
    print("Testing forecast extraction...")
    
    try:
        from gpt5_orchestrator import GPT5Orchestrator
        from system_monitor import reset_system_monitor
        
        # Reset for clean test
        reset_system_monitor()
        
        # Initialize orchestrator
        orchestrator = GPT5Orchestrator()
        
        # Run a simple query
        query = "Commercial forecast for Keytruda in Oncology for cancer treatment"
        
        print(f"Running orchestrator with query: {query}")
        result = await orchestrator.process_drug_forecast(query)
        
        print("\nResult structure:")
        print(f"  Top-level keys: {list(result.keys())}")
        
        forecast_data = result.get('forecast', {})
        print(f"  Forecast type: {type(forecast_data)}")
        
        if isinstance(forecast_data, dict):
            print(f"  Forecast keys: {list(forecast_data.keys())}")
            peak_sales = forecast_data.get('peak_sales_forecast', 0)
            print(f"  Peak sales forecast: ${peak_sales:,.0f}")
        else:
            print(f"  Forecast content: {forecast_data}")
        
        confidence = result.get('confidence', 0)
        print(f"  Confidence: {confidence}")
        
        # Test conversion to trajectory
        if isinstance(forecast_data, dict):
            peak_sales = forecast_data.get('peak_sales_forecast', 0)
            if peak_sales > 0:
                trajectory = [
                    peak_sales * 0.1,   # Y0
                    peak_sales * 0.4,   # Y1  
                    peak_sales * 0.8,   # Y2
                    peak_sales * 1.0,   # Y3
                    peak_sales * 0.9    # Y4
                ]
                print(f"  5-year trajectory: {[f'${x:,.0f}' for x in trajectory]}")
            else:
                print("  No valid peak sales found")
        
        return peak_sales > 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    try:
        success = asyncio.run(test_forecast_extraction())
        if success:
            print("\nForecast extraction test: SUCCESS")
        else:
            print("\nForecast extraction test: FAILED")
        return success
    except Exception as e:
        print(f"Main error: {e}")
        return False

if __name__ == "__main__":
    main()