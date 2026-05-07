#!/usr/bin/env python
"""
Test cNMF pipeline with k=10 only
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from cnmf import cNMF

def test_cnmf_k10():
    """
    Test cNMF analysis with k=10 only
    """
    
    # Load data
    print("="*60)
    print("Loading data...")
    print("="*60)
    input_file = '../sub_adata/strict_endocrine.h5ad'
    adata = sc.read_h5ad(input_file)
    print(f"Original data shape: {adata.shape}")
    
    # Filter cells with very low counts
    print("\nFiltering cells...")
    if 'n_counts' not in adata.obs.columns:
        if hasattr(adata.X, 'toarray'):
            adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        else:
            adata.obs['n_counts'] = adata.X.sum(axis=1)
    
    min_counts = 100
    cells_before = adata.n_obs
    adata = adata[adata.obs['n_counts'] > min_counts, :].copy()
    cells_after = adata.n_obs
    print(f"Filtered {cells_before - cells_after} cells with counts <= {min_counts}")
    print(f"Filtered data shape: {adata.shape}")
    
    # Setup for test run
    base_name = "test_k10"
    output_dir = "cnmf_test_k10"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save counts
    counts_file = os.path.join(output_dir, f"{base_name}_counts.h5ad")
    adata.write_h5ad(counts_file)
    print(f"\nSaved counts to: {counts_file}")
    
    # Initialize cNMF object
    print("\n" + "="*60)
    print("STEP 1: Initializing cNMF...")
    print("="*60)
    
    cnmf_obj = cNMF(
        output_dir=output_dir,
        name=base_name
    )
    
    # Test parameters
    k_value = 10
    n_iter = 10  # Just 10 iterations for testing
    seed = 14
    
    print(f"Testing with k={k_value}, n_iter={n_iter}")
    
    # Step 1: Prepare
    print("\n" + "="*60)
    print("STEP 2: Preparing data...")
    print("="*60)
    
    try:
        cnmf_obj.prepare(
            counts_fn=counts_file,
            components=[k_value],
            n_iter=n_iter,
            seed=seed,
            num_highvar_genes=2000
        )
        print("✓ Prepare step complete!")
    except Exception as e:
        print(f"✗ Error in prepare: {e}")
        return
    
    # Check what files were created
    print("\nFiles created after prepare:")
    tmp_dir = os.path.join(output_dir, base_name, "cnmf_tmp")
    if os.path.exists(tmp_dir):
        files = os.listdir(tmp_dir)
        for f in sorted(files)[:5]:
            print(f"  - {f}")
        if len(files) > 5:
            print(f"  ... and {len(files)-5} more files")
    
    # Step 2: Factorize
    print("\n" + "="*60)
    print("STEP 3: Running factorization...")
    print("="*60)
    
    try:
        cnmf_obj.factorize()
        print("✓ Factorize step complete!")
    except Exception as e:
        print(f"✗ Error in factorize: {e}")
        return
    
    # Check factorization results
    print("\nChecking factorization output...")
    import glob
    iter_files = glob.glob(os.path.join(tmp_dir, f"*.k_{k_value}.iter_*.df.npz"))
    print(f"Found {len(iter_files)} iteration files")
    
    # Step 3: Combine
    print("\n" + "="*60)
    print("STEP 4: Combining results...")
    print("="*60)
    
    try:
        cnmf_obj.combine(components=[k_value])
        print("✓ Combine step complete!")
    except Exception as e:
        print(f"✗ Error in combine: {e}")
        print(f"  Details: {str(e)}")
        return
    
    # Check if merged file was created
    merged_file = os.path.join(tmp_dir, f"{base_name}.spectra.k_{k_value}.merged.df.npz")
    if os.path.exists(merged_file):
        print(f"✓ Merged file created: {merged_file}")
    else:
        print(f"✗ Merged file NOT found: {merged_file}")
        print("\nFiles in temp directory after combine:")
        files = os.listdir(tmp_dir)
        for f in sorted(files):
            if f"k_{k_value}" in f:
                print(f"  - {f}")
    
    # Step 4: K selection plot (optional for single k)
    print("\n" + "="*60)
    print("STEP 5: K selection plot...")
    print("="*60)
    
    try:
        cnmf_obj.k_selection_plot()
        print("✓ K selection plot created!")
    except Exception as e:
        print(f"  Note: K selection plot not created (expected for single k): {e}")
    
    # Step 5: Consensus
    print("\n" + "="*60)
    print("STEP 6: Computing consensus...")
    print("="*60)
    
    try:
        result = cnmf_obj.consensus(
            k=k_value,
            density_threshold=0.1,
            show_clustering=False,
            close_clustergram_fig=True
        )
        
        if result is not None:
            print("✓ Consensus step complete!")
            
            # Save results
            k_output_dir = os.path.join(output_dir, f"k_{k_value}")
            os.makedirs(k_output_dir, exist_ok=True)
            
            # Try to unpack and save results
            if isinstance(result, tuple):
                print(f"  Consensus returned {len(result)} objects")
                if len(result) >= 2:
                    usage_norm, gep_scores = result[0], result[1]
                    
                    # Save usage matrix
                    if usage_norm is not None:
                        usage_file = os.path.join(k_output_dir, f"usage_k{k_value}.csv")
                        usage_norm.to_csv(usage_file)
                        print(f"  ✓ Usage matrix saved: shape {usage_norm.shape}")
                    
                    # Save gene scores
                    if gep_scores is not None:
                        gep_file = os.path.join(k_output_dir, f"gene_scores_k{k_value}.csv")
                        gep_scores.to_csv(gep_file)
                        print(f"  ✓ Gene scores saved: shape {gep_scores.shape}")
            else:
                print(f"  Warning: Unexpected result type: {type(result)}")
        else:
            print("✗ Consensus returned None")
            
    except Exception as e:
        print(f"✗ Error in consensus: {e}")
        import traceback
        print("Full error trace:")
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}/")
    
    # List all output files
    print("\nFinal output structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Limit to first 10 files per directory
            print(f"{subindent}{file}")
        if len(files) > 10:
            print(f"{subindent}... and {len(files)-10} more files")

if __name__ == "__main__":
    test_cnmf_k10()