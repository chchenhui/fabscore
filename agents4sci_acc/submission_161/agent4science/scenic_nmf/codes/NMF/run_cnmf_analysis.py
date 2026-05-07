#!/usr/bin/env python
"""
Run cNMF analysis on strict_endocrine.h5ad with k values from 10 to 50
Each k value will have its own output folder
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from cnmf import cNMF
import argparse

def run_cnmf_analysis(input_file, k_min=10, k_max=50, n_iter=100, seed=14):
    """
    Run complete cNMF analysis for a range of k values
    """
    
    # Load data
    print(f"Loading data from {input_file}...")
    adata = sc.read_h5ad(input_file)
    print(f"Data shape: {adata.shape}")
    
    # Get base name for output
    base_name = "strict_endocrine_cnmf"
    
    # Create main results directory
    main_output_dir = "cnmf_results"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Prepare k values list
    k_values = list(range(k_min, k_max + 1))
    k_str = ','.join(map(str, k_values))
    
    print(f"\nRunning cNMF with k values: {k_values}")
    print(f"Number of iterations per k: {n_iter}")
    print(f"Random seed: {seed}")
    
    # Step 1: Prepare the data (this is done once for all k values)
    print("\n" + "="*60)
    print("STEP 1: Preparing data...")
    print("="*60)
    
    cnmf_obj = cNMF(
        output_dir=main_output_dir,
        name=base_name
    )
    
    # Prepare the data - using counts data
    # cNMF expects raw counts, but since our data is log-transformed, we need to handle this
    print("Preparing count matrix...")
    
    # Save the expression data to a file that cNMF can read
    counts_file = os.path.join(main_output_dir, f"{base_name}_counts.h5ad")
    
    # If data is log-transformed, we should ideally use raw counts
    # For now, we'll use the data as is but note this in the analysis
    adata.write_h5ad(counts_file)
    
    # Run prepare step
    cnmf_obj.prepare(
        counts_fn=counts_file,
        components=k_values,
        n_iter=n_iter,
        seed=seed,
        num_highvar_genes=2000  # Use top 2000 highly variable genes
    )
    
    print("Data preparation complete!")
    
    # Step 2: Factorize for all k values
    print("\n" + "="*60)
    print("STEP 2: Running factorization...")
    print("="*60)
    
    # Run factorization in parallel for multiple k values
    # We'll use worker_index=0 and total_workers=1 to run everything in one process
    cnmf_obj.factorize(
        worker_index=0,
        total_workers=1
    )
    
    print("Factorization complete for all k values!")
    
    # Step 3: Combine results for each k
    print("\n" + "="*60)
    print("STEP 3: Combining results...")
    print("="*60)
    
    for k in k_values:
        print(f"\nCombining results for k={k}...")
        cnmf_obj.combine(k=k)
    
    print("Combination complete for all k values!")
    
    # Step 4: Create k selection plot
    print("\n" + "="*60)
    print("STEP 4: Creating k selection plot...")
    print("="*60)
    
    cnmf_obj.k_selection_plot()
    print(f"K selection plot saved to: {main_output_dir}/{base_name}.k_selection.png")
    
    # Step 5: Consensus and save results for each k
    print("\n" + "="*60)
    print("STEP 5: Computing consensus and saving results for each k...")
    print("="*60)
    
    for k in k_values:
        print(f"\n Processing k={k}...")
        
        # Create directory for this k value
        k_output_dir = os.path.join(main_output_dir, f"k_{k}")
        os.makedirs(k_output_dir, exist_ok=True)
        
        try:
            # Compute consensus
            density_threshold = 0.1  # Can be adjusted based on stability plots
            
            usage_norm, gep_scores, gep_tpm, topgenes = cnmf_obj.consensus(
                k=k,
                density_threshold=density_threshold,
                show_clustering=True,
                close_clustergram_fig=False
            )
            
            # Save results for this k
            # Usage matrix (cells x programs)
            usage_file = os.path.join(k_output_dir, f"usage_k{k}.csv")
            if usage_norm is not None:
                usage_norm.to_csv(usage_file)
                print(f"  - Usage matrix saved to: {usage_file}")
            
            # Gene expression programs (genes x programs)
            gep_file = os.path.join(k_output_dir, f"gene_scores_k{k}.csv")
            if gep_scores is not None:
                gep_scores.to_csv(gep_file)
                print(f"  - Gene scores saved to: {gep_file}")
            
            # Top genes per program
            topgenes_file = os.path.join(k_output_dir, f"top_genes_k{k}.csv")
            if topgenes is not None:
                topgenes.to_csv(topgenes_file)
                print(f"  - Top genes saved to: {topgenes_file}")
            
            print(f"  ✓ Results for k={k} saved successfully!")
            
        except Exception as e:
            print(f"  ✗ Error processing k={k}: {str(e)}")
            continue
    
    print("\n" + "="*60)
    print("cNMF ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved in: {main_output_dir}/")
    print(f"Individual k results in: {main_output_dir}/k_*/")
    
    return main_output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cNMF analysis')
    parser.add_argument('--input', type=str, 
                       default='../sub_adata/strict_endocrine.h5ad',
                       help='Input h5ad file path')
    parser.add_argument('--k_min', type=int, default=10,
                       help='Minimum k value')
    parser.add_argument('--k_max', type=int, default=50,
                       help='Maximum k value')
    parser.add_argument('--n_iter', type=int, default=100,
                       help='Number of iterations per k')
    parser.add_argument('--seed', type=int, default=14,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Run analysis
    run_cnmf_analysis(
        input_file=args.input,
        k_min=args.k_min,
        k_max=args.k_max,
        n_iter=args.n_iter,
        seed=args.seed
    )