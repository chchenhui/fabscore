#!/usr/bin/env python3
"""
Quick test to verify calibration is working
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_scientist.gpt5_orchestrator import GPT5Orchestrator

async def test_calibration():
    print("Testing calibrated forecast methods...")
    
    orchestrator = GPT5Orchestrator()
    
    # Test Keytruda (Oncology) - should target ~$25B
    keytruda_data = {
        "drug_name": "Keytruda",
        "therapeutic_area": "Oncology",
        "indication": "cancer treatment"
    }
    
    market_analysis = {
        "market_sizing": {"addressable_market": 2_000_000_000},
        "competitive_landscape": {},
        "regulatory_pathway": {}
    }
    
    # Test patient flow method
    pf_result = await orchestrator._patient_flow_method(keytruda_data)
    print(f"Keytruda Patient Flow Forecast: ${pf_result['peak_sales_forecast']/1e9:.1f}B")
    
    # Test ML ensemble method  
    ml_result = await orchestrator._ml_ensemble_method(keytruda_data, market_analysis)
    print(f"Keytruda ML Ensemble Forecast: ${ml_result['peak_sales_forecast']/1e9:.1f}B")
    
    # Test Repatha (Cardiovascular) - should target ~$1.5B
    repatha_data = {
        "drug_name": "Repatha",
        "therapeutic_area": "Cardiovascular", 
        "indication": "cholesterol management"
    }
    
    # Test patient flow method
    pf_result2 = await orchestrator._patient_flow_method(repatha_data)
    print(f"Repatha Patient Flow Forecast: ${pf_result2['peak_sales_forecast']/1e9:.1f}B")
    
    # Test ML ensemble method
    ml_result2 = await orchestrator._ml_ensemble_method(repatha_data, market_analysis)
    print(f"Repatha ML Ensemble Forecast: ${ml_result2['peak_sales_forecast']/1e9:.1f}B")
    
    print("\nCalibration targets:")
    print("- Keytruda: $25B actual")
    print("- Repatha: $1.5B actual")

if __name__ == "__main__":
    asyncio.run(test_calibration())