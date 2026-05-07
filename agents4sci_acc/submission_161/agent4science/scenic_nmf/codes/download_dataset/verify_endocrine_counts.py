#!/usr/bin/env python3
"""
Verify endocrine cell counts in downloaded h5ad files against the summary CSV
"""

import pandas as pd
import scanpy as sc
import os
import warnings
warnings.filterwarnings('ignore')

def verify_endocrine_counts():
    """
    Read each h5ad file and verify endocrine cell counts
    """
    # Read the summary CSV
    summary_df = pd.read_csv("endocrine_datasets_summary.csv")
    
    results = []
    
    print("Verifying endocrine cell counts in downloaded datasets...")
    print("="*80)
    
    for idx, row in summary_df.iterrows():
        dataset_id = row['dataset_id']
        expected_count = row['endocrine_cell_count']
        expected_cell_types = row['endocrine_cell_types']
        dataset_title = row['dataset_title']
        
        h5ad_path = f"endocrine_datasets/{dataset_id}.h5ad"
        
        if not os.path.exists(h5ad_path):
            print(f"{idx+1:2d}. {dataset_id}")
            print(f"    MISSING FILE: {h5ad_path}")
            results.append({
                'dataset_id': dataset_id,
                'expected_count': expected_count,
                'actual_count': 'MISSING',
                'match': False
            })
            continue
        
        try:
            # Read the h5ad file
            adata = sc.read_h5ad(h5ad_path)
            
            # Count cells with "endocrine" in cell_type
            if 'cell_type' in adata.obs.columns:
                endocrine_mask = adata.obs['cell_type'].str.contains('endocrine', case=False, na=False)
                actual_count = endocrine_mask.sum()
                
                # Get unique endocrine cell types
                endocrine_types = adata.obs[endocrine_mask]['cell_type'].unique()
                actual_cell_types = '; '.join(sorted(endocrine_types))
                
                match = (actual_count == expected_count)
                status = "✓ MATCH" if match else "✗ MISMATCH"
                
                print(f"{idx+1:2d}. {dataset_id}")
                print(f"    Title: {dataset_title[:60]}...")
                print(f"    Expected: {expected_count:,} endocrine cells")
                print(f"    Actual:   {actual_count:,} endocrine cells")
                print(f"    Status:   {status}")
                
                if not match:
                    print(f"    Expected types: {expected_cell_types[:100]}...")
                    print(f"    Actual types:   {actual_cell_types[:100]}...")
                
                print()
                
                results.append({
                    'dataset_id': dataset_id,
                    'expected_count': expected_count,
                    'actual_count': actual_count,
                    'expected_types': expected_cell_types,
                    'actual_types': actual_cell_types,
                    'match': match
                })
                
            else:
                print(f"{idx+1:2d}. {dataset_id}")
                print(f"    ERROR: No 'cell_type' column found")
                print()
                results.append({
                    'dataset_id': dataset_id,
                    'expected_count': expected_count,
                    'actual_count': 'NO_CELL_TYPE_COLUMN',
                    'match': False
                })
                
        except Exception as e:
            print(f"{idx+1:2d}. {dataset_id}")
            print(f"    ERROR reading file: {e}")
            print()
            results.append({
                'dataset_id': dataset_id,
                'expected_count': expected_count,
                'actual_count': f'ERROR: {e}',
                'match': False
            })
    
    # Summary
    results_df = pd.DataFrame(results)
    matches = results_df['match'].sum()
    total = len(results_df)
    
    print("="*80)
    print("VERIFICATION SUMMARY:")
    print(f"  Total datasets: {total}")
    print(f"  Matches: {matches}")
    print(f"  Mismatches: {total - matches}")
    print(f"  Success rate: {matches/total*100:.1f}%")
    
    # Save detailed results
    results_df.to_csv("endocrine_verification_results.csv", index=False)
    print(f"\nDetailed results saved to: endocrine_verification_results.csv")
    
    return results_df

if __name__ == "__main__":
    results = verify_endocrine_counts()