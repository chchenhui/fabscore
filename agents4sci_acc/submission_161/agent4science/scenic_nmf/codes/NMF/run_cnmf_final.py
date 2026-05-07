#!/usr/bin/env python
"""
Run cNMF analysis on strict_endocrine.h5ad with k values from 10 to 50
Each k value will have its own output folder
Final version with correct API calls
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
    print(f"Total factorizations to run: {len(k_values) * n_iter}")
    
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
    print(f"This will take a while - running {n_iter} iterations for each k value from {k_min} to {k_max}...")
    print("Progress will be shown below...")
    
    # Run factorization (correct API call without worker parameters)
    cnmf_obj.factorize()
    
    print("\nFactorization complete for all k values!")
    
    # Step 3: Combine results for each k
    print("\n" + "="*60)
    print("STEP 3: Combining results...")
    print("="*60)
    
    for k in k_values:
        print(f"Combining results for k={k}...")
        try:
            cnmf_obj.combine(k=k)
            print(f"  ✓ Combined successfully for k={k}")
        except Exception as e:
            print(f"  ✗ Warning: Could not combine results for k={k}: {e}")
    
    print("\nCombination step complete!")
    
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
        print(f"\n{'='*40}")
        print(f"Processing k={k}...")
        print(f"{'='*40}")
        
        # Create directory for this k value
        k_output_dir = os.path.join(main_output_dir, f"k_{k}")
        os.makedirs(k_output_dir, exist_ok=True)
        
        try:
            # Compute consensus
            density_threshold = 0.1  # Can be adjusted based on stability plots
            
            print(f"Computing consensus for k={k} with density_threshold={density_threshold}...")
            
            result = cnmf_obj.consensus(
                k=k,
                density_threshold=density_threshold,
                show_clustering=False,  # Don't show plots during batch processing
                close_clustergram_fig=True
            )
            
            # Handle different possible return formats from consensus
            if result is None:
                print(f"  Warning: No result returned from consensus for k={k}")
                failed_k.append(k)
                continue
                
            # Try to unpack results
            try:
                if isinstance(result, tuple):
                    if len(result) == 4:
                        usage_norm, gep_scores, gep_tpm, topgenes = result
                    elif len(result) == 3:
                        usage_norm, gep_scores, topgenes = result
                        gep_tpm = None
                    elif len(result) == 2:
                        usage_norm, gep_scores = result
                        gep_tpm = None
                        topgenes = None
                    else:
                        print(f"  Warning: Unexpected number of returns ({len(result)}) from consensus for k={k}")
                        usage_norm = result[0] if len(result) > 0 else None
                        gep_scores = result[1] if len(result) > 1 else None
                        gep_tpm = None
                        topgenes = None
                else:
                    # Single return value, assume it's usage
                    usage_norm = result
                    gep_scores = None
                    gep_tpm = None
                    topgenes = None
            except:
                print(f"  Warning: Could not unpack consensus results for k={k}")
                usage_norm = None
                gep_scores = None
                gep_tpm = None
                topgenes = None
            
            # Save results for this k
            files_saved = []
            
            # Usage matrix (cells x programs)
            if usage_norm is not None:
                usage_file = os.path.join(k_output_dir, f"usage_k{k}.csv")
                usage_norm.to_csv(usage_file)
                files_saved.append(f"usage matrix ({usage_norm.shape})")
                print(f"  ✓ Usage matrix saved: {usage_norm.shape}")
            
            # Gene expression programs (genes x programs)
            if gep_scores is not None:
                gep_file = os.path.join(k_output_dir, f"gene_scores_k{k}.csv")
                gep_scores.to_csv(gep_file)
                files_saved.append(f"gene scores ({gep_scores.shape})")
                print(f"  ✓ Gene scores saved: {gep_scores.shape}")
            
            # Top genes per program
            if topgenes is not None:
                topgenes_file = os.path.join(k_output_dir, f"top_genes_k{k}.csv")
                topgenes.to_csv(topgenes_file)
                files_saved.append("top genes")
                print(f"  ✓ Top genes saved")
            
            # TPM normalized gene expression programs
            if gep_tpm is not None:
                gep_tpm_file = os.path.join(k_output_dir, f"gene_tpm_k{k}.csv")
                gep_tpm.to_csv(gep_tpm_file)
                files_saved.append(f"gene TPM ({gep_tpm.shape})")
                print(f"  ✓ Gene TPM saved: {gep_tpm.shape}")
            
            if files_saved:
                successful_k.append(k)
                print(f"\n  ✓✓✓ Results for k={k} saved successfully!")
                print(f"      Files saved: {', '.join(files_saved)}")
            else:
                failed_k.append(k)
                print(f"\n  ✗ No results saved for k={k}")
            
        except Exception as e:
            failed_k.append(k)
            print(f"\n  ✗✗✗ Error processing k={k}: {str(e)}")
            # Save error log
            error_file = os.path.join(k_output_dir, "error.txt")
            with open(error_file, 'w') as f:
                f.write(f"Error processing k={k}:\n{str(e)}\n")
                import traceback
                f.write(f"\nFull traceback:\n{traceback.format_exc()}")
    
    # Summary
    print("\n" + "="*60)
    print("cNMF ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved in: {main_output_dir}/")
    print(f"Individual k results in: {main_output_dir}/k_*/")
    print(f"\nSuccessfully processed k values ({len(successful_k)}): {successful_k}")
    if failed_k:
        print(f"Failed k values ({len(failed_k)}): {failed_k}")
    
    # Save summary
    summary_file = os.path.join(main_output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"cNMF Analysis Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Original data shape: {cells_before} cells × {adata.n_vars} genes\n")
        f.write(f"Filtered data shape: {cells_after} cells × {adata.n_vars} genes\n")
        f.write(f"Cells filtered: {cells_before - cells_after}\n")
        f.write(f"\nParameters:\n")
        f.write(f"  K range: {k_min} to {k_max}\n")
        f.write(f"  Iterations per k: {n_iter}\n")
        f.write(f"  Random seed: {seed}\n")
        f.write(f"  Highly variable genes: 2000\n")
        f.write(f"\nResults:\n")
        f.write(f"  Successful k values ({len(successful_k)}): {successful_k}\n")
        if failed_k:
            f.write(f"  Failed k values ({len(failed_k)}): {failed_k}\n")
        f.write(f"\nOutput structure:\n")
        f.write(f"  {main_output_dir}/\n")
        f.write(f"    ├── {base_name}.k_selection.png (k selection plot)\n")
        f.write(f"    ├── analysis_summary.txt (this file)\n")
        f.write(f"    └── k_{{k}}/  (for each k value)\n")
        f.write(f"        ├── usage_k{{k}}.csv (cell × program usage matrix)\n")
        f.write(f"        ├── gene_scores_k{{k}}.csv (gene × program scores)\n")
        f.write(f"        ├── gene_tpm_k{{k}}.csv (TPM normalized gene scores)\n")
        f.write(f"        └── top_genes_k{{k}}.csv (top genes per program)\n")
    
    print(f"\n✓ Analysis summary saved to: {summary_file}")
    
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