#!/usr/bin/env python
"""
Run cNMF analysis on strict_endocrine.h5ad with k values from 10 to 50
Each k value will have its own output folder
Fixed version that handles cells with zero counts
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
    print(f"Original data shape: {adata.shape}")
    
    # Filter cells with very low counts
    print("\nFiltering cells with low counts...")
    # Calculate total counts per cell if not already present
    if 'n_counts' not in adata.obs.columns:
        if hasattr(adata.X, 'toarray'):
            adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        else:
            adata.obs['n_counts'] = adata.X.sum(axis=1)
    
    # Filter cells with very low total counts (potential empty droplets)
    min_counts = 100  # Minimum total counts per cell
    cells_before = adata.n_obs
    adata = adata[adata.obs['n_counts'] > min_counts, :].copy()
    cells_after = adata.n_obs
    print(f"Filtered {cells_before - cells_after} cells with counts <= {min_counts}")
    print(f"Filtered data shape: {adata.shape}")
    
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
    
    # Prepare the data
    print("Preparing count matrix...")
    
    # Save the filtered expression data
    counts_file = os.path.join(main_output_dir, f"{base_name}_counts_filtered.h5ad")
    adata.write_h5ad(counts_file)
    print(f"Saved filtered data to: {counts_file}")
    
    # Run prepare step with adjusted parameters
    try:
        cnmf_obj.prepare(
            counts_fn=counts_file,
            components=k_values,
            n_iter=n_iter,
            seed=seed,
            num_highvar_genes=2000  # Use top 2000 highly variable genes
        )
        print("Data preparation complete!")
    except Exception as e:
        print(f"Error in prepare step: {e}")
        # Try with fewer highly variable genes
        print("Retrying with 1500 highly variable genes...")
        cnmf_obj.prepare(
            counts_fn=counts_file,
            components=k_values,
            n_iter=n_iter,
            seed=seed,
            num_highvar_genes=1500  # Reduced number
        )
        print("Data preparation complete with adjusted parameters!")
    
    # Step 2: Factorize for all k values
    print("\n" + "="*60)
    print("STEP 2: Running factorization...")
    print("="*60)
    print("This will take a while as we're running 100 iterations for each k value from 10 to 50...")
    
    # Run factorization
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
        print(f"Combining results for k={k}...")
        try:
            cnmf_obj.combine(k=k)
        except Exception as e:
            print(f"  Warning: Could not combine results for k={k}: {e}")
    
    print("Combination complete!")
    
    # Step 4: Create k selection plot
    print("\n" + "="*60)
    print("STEP 4: Creating k selection plot...")
    print("="*60)
    
    try:
        cnmf_obj.k_selection_plot()
        print(f"K selection plot saved to: {main_output_dir}/{base_name}.k_selection.png")
    except Exception as e:
        print(f"Warning: Could not create k selection plot: {e}")
    
    # Step 5: Consensus and save results for each k
    print("\n" + "="*60)
    print("STEP 5: Computing consensus and saving results for each k...")
    print("="*60)
    
    successful_k = []
    failed_k = []
    
    for k in k_values:
        print(f"\nProcessing k={k}...")
        
        # Create directory for this k value
        k_output_dir = os.path.join(main_output_dir, f"k_{k}")
        os.makedirs(k_output_dir, exist_ok=True)
        
        try:
            # Compute consensus
            density_threshold = 0.1  # Can be adjusted based on stability plots
            
            result = cnmf_obj.consensus(
                k=k,
                density_threshold=density_threshold,
                show_clustering=False,  # Don't show plots during batch processing
                close_clustergram_fig=True
            )
            
            # Unpack results based on what consensus returns
            if isinstance(result, tuple) and len(result) == 4:
                usage_norm, gep_scores, gep_tpm, topgenes = result
            else:
                print(f"  Warning: Unexpected return format from consensus for k={k}")
                usage_norm = gep_scores = gep_tpm = topgenes = None
            
            # Save results for this k
            # Usage matrix (cells x programs)
            if usage_norm is not None:
                usage_file = os.path.join(k_output_dir, f"usage_k{k}.csv")
                usage_norm.to_csv(usage_file)
                print(f"  - Usage matrix saved: {usage_norm.shape}")
            
            # Gene expression programs (genes x programs)
            if gep_scores is not None:
                gep_file = os.path.join(k_output_dir, f"gene_scores_k{k}.csv")
                gep_scores.to_csv(gep_file)
                print(f"  - Gene scores saved: {gep_scores.shape}")
            
            # Top genes per program
            if topgenes is not None:
                topgenes_file = os.path.join(k_output_dir, f"top_genes_k{k}.csv")
                topgenes.to_csv(topgenes_file)
                print(f"  - Top genes saved")
            
            # Also save the TPM normalized gene expression programs
            if gep_tpm is not None:
                gep_tpm_file = os.path.join(k_output_dir, f"gene_tpm_k{k}.csv")
                gep_tpm.to_csv(gep_tpm_file)
                print(f"  - Gene TPM saved: {gep_tpm.shape}")
            
            successful_k.append(k)
            print(f"  ✓ Results for k={k} saved successfully!")
            
        except Exception as e:
            failed_k.append(k)
            print(f"  ✗ Error processing k={k}: {str(e)}")
            # Save error log
            error_file = os.path.join(k_output_dir, "error.txt")
            with open(error_file, 'w') as f:
                f.write(f"Error processing k={k}:\n{str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("cNMF ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved in: {main_output_dir}/")
    print(f"Individual k results in: {main_output_dir}/k_*/")
    print(f"\nSuccessfully processed k values: {successful_k}")
    if failed_k:
        print(f"Failed k values: {failed_k}")
    
    # Save summary
    summary_file = os.path.join(main_output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"cNMF Analysis Summary\n")
        f.write(f"====================\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Data shape (after filtering): {adata.shape}\n")
        f.write(f"K range: {k_min} to {k_max}\n")
        f.write(f"Iterations per k: {n_iter}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"\nSuccessful k values: {successful_k}\n")
        if failed_k:
            f.write(f"Failed k values: {failed_k}\n")
    
    print(f"\nAnalysis summary saved to: {summary_file}")
    
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