#!/usr/bin/env python3
"""
Simple H2 test: Our multi-agent system vs single baseline
Following Linus principles: Test what actually works
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import asyncio

# Add ai_scientist to path
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)

# Add src to path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from gpt5_orchestrator import GPT5Orchestrator
from models.baselines import peak_sales_heuristic

async def test_h2_simple():
    """
    Simple H2 test: Multi-agent vs single baseline
    """
    
    print("=== H2 SIMPLE TEST ===")
    print("Multi-Agent vs Peak Sales Heuristic")
    
    # Load data with revenues
    launches = pd.read_parquet('data_proc/launches.parquet')
    revenues = pd.read_parquet('data_proc/launch_revenues.parquet')
    
    # Find drugs with revenue data
    drugs_with_revenue = set(revenues['launch_id'].unique())
    test_drugs = launches[launches['launch_id'].isin(drugs_with_revenue)].head(3)
    
    print(f"Testing on {len(test_drugs)} drugs with actual revenue data")
    
    # Initialize multi-agent system
    orchestrator = GPT5Orchestrator()
    
    results = []
    
    for _, drug in test_drugs.iterrows():
        drug_name = drug['drug_name']
        launch_id = drug['launch_id']
        
        print(f"\n--- Testing {drug_name} ---")
        
        # Get actual revenues
        actual_revs = revenues[revenues['launch_id'] == launch_id].sort_values('year_since_launch')
        if len(actual_revs) == 0:
            continue
        
        # Test 1: Peak Sales Heuristic (Baseline)
        try:
            baseline_peak = peak_sales_heuristic(drug)
            print(f"Baseline peak forecast: ${baseline_peak:,.0f}")
        except Exception as e:
            print(f"Baseline failed: {e}")
            baseline_peak = 0
        
        # Test 2: Multi-Agent System
        try:
            query = f"Commercial forecast for {drug_name}"
            result = await orchestrator.process_drug_forecast(query)
            multi_agent_forecast = result.get('forecast', {})
            peak_forecast = multi_agent_forecast.get('peak_sales', 0)
            print(f"Multi-agent peak forecast: ${peak_forecast:,.0f}")
            print(f"Multi-agent confidence: {result.get('confidence', 0):.1f}%")
        except Exception as e:
            print(f"Multi-agent failed: {e}")
            peak_forecast = 0
        
        # Get actual peak (max revenue in available years)
        actual_peak = actual_revs['revenue_usd'].max()
        print(f"Actual peak revenue: ${actual_peak:,.0f}")
        
        # Calculate errors
        baseline_error = abs(baseline_peak - actual_peak) / actual_peak if actual_peak > 0 else float('inf')
        multi_error = abs(peak_forecast - actual_peak) / actual_peak if actual_peak > 0 else float('inf')
        
        print(f"Baseline error: {baseline_error:.1%}")
        print(f"Multi-agent error: {multi_error:.1%}")
        
        results.append({
            'drug': drug_name,
            'actual_peak': float(actual_peak),
            'baseline_forecast': float(baseline_peak),
            'multi_agent_forecast': float(peak_forecast),
            'baseline_error': float(baseline_error) if not np.isinf(baseline_error) else 999.0,
            'multi_agent_error': float(multi_error) if not np.isinf(multi_error) else 999.0,
            'multi_agent_better': bool(multi_error < baseline_error)
        })
    
    # Summary
    print(f"\n=== H2 RESULTS SUMMARY ===")
    if results:
        baseline_avg_error = np.mean([r['baseline_error'] for r in results if not np.isinf(r['baseline_error'])])
        multi_avg_error = np.mean([r['multi_agent_error'] for r in results if not np.isinf(r['multi_agent_error'])])
        multi_wins = sum(1 for r in results if r['multi_agent_better'])
        
        print(f"Average baseline error: {baseline_avg_error:.1%}")
        print(f"Average multi-agent error: {multi_avg_error:.1%}")
        print(f"Multi-agent wins: {multi_wins}/{len(results)}")
        print(f"Multi-agent success rate: {multi_wins/len(results)*100:.0f}%")
        
        # Save results
        with open('results/h2_simple_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'baseline_avg_error': baseline_avg_error,
                    'multi_agent_avg_error': multi_avg_error,
                    'multi_agent_wins': multi_wins,
                    'total_tests': len(results),
                    'multi_agent_success_rate': multi_wins/len(results)
                },
                'detailed_results': results
            }, f, indent=2)
        
        print("Results saved to results/h2_simple_results.json")
        
        return multi_avg_error < baseline_avg_error
    else:
        print("No valid results!")
        return False

def main():
    """Main function"""
    try:
        success = asyncio.run(test_h2_simple())
        print(f"\nH2 Test Result: {'MULTI-AGENT WINS' if success else 'BASELINE WINS'}")
    except Exception as e:
        print(f"H2 test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()