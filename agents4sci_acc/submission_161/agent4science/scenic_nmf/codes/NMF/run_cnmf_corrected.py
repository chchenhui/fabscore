#!/usr/bin/env python
"""
Corrected cNMF analysis script with proper API usage
Run with k values from 10 to 50
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from cnmf import cNMF
import argparse
import shutil

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
    
    # Setup
    base_name = "strict_endocrine_cnmf"
    main_output_dir = "cnmf_results_corrected"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Prepare k values list
    k_values = list(range(k_min, k_max + 1))
    
    print(f"\nRunning cNMF with k values: {k_values}")
    print(f"Number of iterations per k: {n_iter}")
    print(f"Random seed: {seed}")
    
    # Step 1: Prepare
    print("\n" + "="*60)
    print("STEP 1: Preparing data...")
    print("="*60)
    
    cnmf_obj = cNMF(
        output_dir=main_output_dir,
        name=base_name
    )
    
    # Save filtered data
    counts_file = os.path.join(main_output_dir, f"{base_name}_counts_filtered.h5ad")
    adata.write_h5ad(counts_file)
    print(f"Saved filtered data to: {counts_file}")
    
    # Run prepare
    try:
        cnmf_obj.prepare(
            counts_fn=counts_file,
            components=k_values,
            n_iter=n_iter,
            seed=seed,
            num_highvar_genes=2000
        )
        print("Data preparation complete!")
    except Exception as e:
        if "zero counts" in str(e):
            print(f"Some cells have zero counts, retrying with 1500 HVGs...")
            cnmf_obj.prepare(
                counts_fn=counts_file,
                components=k_values,
                n_iter=n_iter,
                seed=seed,
                num_highvar_genes=1500
            )
            print("Data preparation complete with adjusted parameters!")
        else:
            raise e
    
    # Step 2: Factorize
    print("\n" + "="*60)
    print("STEP 2: Running factorization...")
    print("="*60)
    print(f"This will run {len(k_values) * n_iter} total factorizations...")
    
    cnmf_obj.factorize()
    
    print("Factorization complete!")
    
    # Step 3: Combine results for each k
    print("\n" + "="*60)
    print("STEP 3: Combining results...")
    print("="*60)
    
    # Combine all k values at once
    cnmf_obj.combine(components=k_values)
    print(f"Combined results for all k values")
    
    # Step 4: Create k selection plot
    print("\n" + "="*60)
    print("STEP 4: Creating k selection plot...")
    print("="*60)
    
    try:
        cnmf_obj.k_selection_plot()
        print(f"K selection plot saved")
    except Exception as e:
        print(f"Warning: Could not create k selection plot: {e}")
    
    # Step 5: Consensus for each k and organize results
    print("\n" + "="*60)
    print("STEP 5: Computing consensus and organizing results...")
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
            density_threshold = 0.1
            
            # Run consensus - it saves files but doesn't return them
            cnmf_obj.consensus(
                k=k,
                density_threshold=density_threshold,
                show_clustering=False,
                close_clustergram_fig=True
            )
            
            # Now copy/move the generated files to the k-specific directory
            cnmf_output_dir = os.path.join(main_output_dir, base_name)
            
            # Files to copy
            files_to_copy = [
                (f"{base_name}.usages.k_{k}.dt_0_1.consensus.txt", f"usage_k{k}.txt"),
                (f"{base_name}.gene_spectra_score.k_{k}.dt_0_1.txt", f"gene_scores_k{k}.txt"),
                (f"{base_name}.gene_spectra_tpm.k_{k}.dt_0_1.txt", f"gene_tpm_k{k}.txt"),
                (f"{base_name}.spectra.k_{k}.dt_0_1.consensus.txt", f"spectra_consensus_k{k}.txt")
            ]
            
            files_copied = []
            for src_file, dest_file in files_to_copy:
                src_path = os.path.join(cnmf_output_dir, src_file)
                dest_path = os.path.join(k_output_dir, dest_file)
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dest_path)
                    files_copied.append(dest_file)
                    
                    # Also convert to CSV for easier access
                    if dest_file.startswith("usage") or dest_file.startswith("gene"):
                        df = pd.read_csv(src_path, sep='\t', index_col=0)
                        csv_path = dest_path.replace('.txt', '.csv')
                        df.to_csv(csv_path)
                        files_copied.append(os.path.basename(csv_path))
            
            if files_copied:
                successful_k.append(k)
                print(f"  ✓ Results for k={k} saved successfully!")
                print(f"    Files: {', '.join(files_copied[:3])}...")
            else:
                failed_k.append(k)
                print(f"  ✗ No consensus files found for k={k}")
                
        except Exception as e:
            failed_k.append(k)
            print(f"  ✗ Error processing k={k}: {str(e)}")
            # Save error log
            error_file = os.path.join(k_output_dir, "error.txt")
            with open(error_file, 'w') as f:
                f.write(f"Error processing k={k}:\n{str(e)}\n")
    
    # Create summary
    print("\n" + "="*60)
    print("cNMF ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved in: {main_output_dir}/")
    print(f"Individual k results in: {main_output_dir}/k_*/")
    print(f"\nSuccessfully processed k values ({len(successful_k)}): {successful_k[:10]}...")
    if failed_k:
        print(f"Failed k values ({len(failed_k)}): {failed_k}")
    
    # Save summary
    summary_file = os.path.join(main_output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"cNMF Analysis Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Original data: {cells_before} cells × {adata.n_vars} genes\n")
        f.write(f"Filtered data: {cells_after} cells × {adata.n_vars} genes\n")
        f.write(f"Cells filtered: {cells_before - cells_after}\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  K range: {k_min} to {k_max}\n")
        f.write(f"  Iterations per k: {n_iter}\n")
        f.write(f"  Random seed: {seed}\n")
        f.write(f"  Highly variable genes: 2000\n")
        f.write(f"  Density threshold: 0.1\n\n")
        f.write(f"Results:\n")
        f.write(f"  Successful: {len(successful_k)} k values\n")
        if failed_k:
            f.write(f"  Failed: {len(failed_k)} k values\n")
        f.write(f"\nOutput files per k:\n")
        f.write(f"  - usage_k{{k}}.csv: Cell × program usage matrix\n")
        f.write(f"  - gene_scores_k{{k}}.csv: Gene × program scores\n")
        f.write(f"  - gene_tpm_k{{k}}.csv: TPM normalized gene scores\n")
        f.write(f"  - spectra_consensus_k{{k}}.txt: Consensus spectra\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")
    
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
    
    run_cnmf_analysis(
        input_file=args.input,
        k_min=args.k_min,
        k_max=args.k_max,
        n_iter=args.n_iter,
        seed=args.seed
    )