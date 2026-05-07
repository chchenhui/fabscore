#!/usr/bin/env python3
"""
Quick test to verify baseline fixes work
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_baseline_fixes():
    """Test that baselines now return reasonable values instead of 0"""
    
    print("Testing baseline fixes...")
    
    # Load real data
    launches = pd.read_parquet('data_proc/launches.parquet')
    
    # Test with a drug that had missing data (Keytruda)
    keytruda = launches[launches['drug_name'] == 'Keytruda'].iloc[0]
    
    print(f"\nTesting Keytruda:")
    print(f"  Therapeutic area: {keytruda.get('therapeutic_area', 'Unknown')}")
    print(f"  Original eligible_patients: {keytruda.get('eligible_patients_at_launch', 'Missing')}")
    print(f"  Original price_month: {keytruda.get('list_price_month_usd_launch', 'Missing')}")
    print(f"  Original GTN: {keytruda.get('net_gtn_pct_launch', 'Missing')}")
    
    try:
        from models.baselines import peak_sales_heuristic, ensemble_baseline
        
        # Test peak sales heuristic
        peak_forecast = peak_sales_heuristic(keytruda)
        print(f"\nPeak Sales Heuristic: ${peak_forecast:,.0f}")
        
        if peak_forecast > 0:
            print("SUCCESS: Baseline now returns positive value!")
        else:
            print("ISSUE: Still returning 0")
            
        # Test ensemble
        ensemble_result = ensemble_baseline(keytruda, years=5)
        if isinstance(ensemble_result, dict) and 'ensemble' in ensemble_result:
            ensemble_peak = max(ensemble_result['ensemble'])
            print(f"Ensemble Baseline Peak: ${ensemble_peak:,.0f}")
        else:
            print("Ensemble format issue")
            
    except Exception as e:
        print(f"Baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return peak_forecast > 0

def main():
    success = test_baseline_fixes()
    if success:
        print("\nBaseline fixes: SUCCESS")
    else:
        print("\nBaseline fixes: FAILED")
    return success

if __name__ == "__main__":
    main()