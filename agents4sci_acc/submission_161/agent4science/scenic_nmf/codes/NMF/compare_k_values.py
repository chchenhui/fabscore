#!/usr/bin/env python
"""
Compare results across different k values
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compare_k_values(k_values=[10, 15, 20, 25, 30], base_dir='nmf_results'):
    """
    Compare NMF results across different k values
    """
    
    print("="*60)
    print("COMPARING NMF RESULTS ACROSS K VALUES")
    print("="*60)
    
    # Create comparison directory
    comp_dir = f"{base_dir}_comparison"
    os.makedirs(comp_dir, exist_ok=True)
    
    comparison_data = {}
    
    for k in k_values:
        result_dir = f"{base_dir}_k{k}"
        
        if os.path.exists(result_dir):
            print(f"\nAnalyzing k={k}...")
            
            # Load enrichment summary
            enrich_file = os.path.join(result_dir, 'enrichment', 'enrichment_summary.csv')
            if os.path.exists(enrich_file):
                enrich_df = pd.read_csv(enrich_file)
                n_enriched = len(enrich_df)
                print(f"  Programs with enrichment: {n_enriched}/{k}")
            else:
                n_enriched = 0
            
            # Load activity data
            activity_files = {
                'cell_type': os.path.join(result_dir, 'activity_by_cell_type.csv'),
                'disease': os.path.join(result_dir, 'activity_by_disease.csv'),
                'tissue': os.path.join(result_dir, 'activity_by_tissue.csv')
            }
            
            activity_variance = {}
            for name, file_path in activity_files.items():
                if os.path.exists(file_path):
                    act_df = pd.read_csv(file_path, index_col=0)
                    # Calculate variance across groups for each program
                    var = act_df.var(axis=0).mean()
                    activity_variance[name] = var
                    print(f"  Mean variance in {name}: {var:.4f}")
            
            comparison_data[k] = {
                'n_enriched': n_enriched,
                'enrichment_ratio': n_enriched / k,
                'activity_variance': activity_variance
            }
        else:
            print(f"\nDirectory not found: {result_dir}")
    
    # Create comparison plots
    if comparison_data:
        create_comparison_plots(comparison_data, comp_dir, k_values)
    
    # Save comparison summary
    summary_df = pd.DataFrame(comparison_data).T
    summary_df.to_csv(os.path.join(comp_dir, 'k_comparison_summary.csv'))
    print(f"\nComparison summary saved to: {comp_dir}/k_comparison_summary.csv")
    
    return comparison_data

def create_comparison_plots(comparison_data, comp_dir, k_values):
    """
    Create plots comparing different k values
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Enrichment ratio
    ax1 = axes[0, 0]
    k_vals = []
    enrich_ratios = []
    for k, data in comparison_data.items():
        k_vals.append(k)
        enrich_ratios.append(data['enrichment_ratio'])
    
    ax1.plot(k_vals, enrich_ratios, 'o-', color='steelblue', markersize=10)
    ax1.set_xlabel('k (number of programs)', fontsize=12)
    ax1.set_ylabel('Fraction of programs with enrichment', fontsize=12)
    ax1.set_title('Biological Interpretability vs k', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Activity variance by cell type
    ax2 = axes[0, 1]
    cell_type_vars = []
    for k in k_vals:
        if k in comparison_data and 'cell_type' in comparison_data[k]['activity_variance']:
            cell_type_vars.append(comparison_data[k]['activity_variance']['cell_type'])
        else:
            cell_type_vars.append(0)
    
    ax2.plot(k_vals, cell_type_vars, 'o-', color='coral', markersize=10)
    ax2.set_xlabel('k (number of programs)', fontsize=12)
    ax2.set_ylabel('Mean variance across cell types', fontsize=12)
    ax2.set_title('Cell Type Specificity vs k', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Activity variance by disease
    ax3 = axes[1, 0]
    disease_vars = []
    for k in k_vals:
        if k in comparison_data and 'disease' in comparison_data[k]['activity_variance']:
            disease_vars.append(comparison_data[k]['activity_variance']['disease'])
        else:
            disease_vars.append(0)
    
    ax3.plot(k_vals, disease_vars, 'o-', color='green', markersize=10)
    ax3.set_xlabel('k (number of programs)', fontsize=12)
    ax3.set_ylabel('Mean variance across diseases', fontsize=12)
    ax3.set_title('Disease Specificity vs k', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary metrics
    ax4 = axes[1, 1]
    
    # Normalize metrics for comparison
    norm_enrich = np.array(enrich_ratios) / max(enrich_ratios) if max(enrich_ratios) > 0 else enrich_ratios
    norm_cell = np.array(cell_type_vars) / max(cell_type_vars) if max(cell_type_vars) > 0 else cell_type_vars
    norm_disease = np.array(disease_vars) / max(disease_vars) if max(disease_vars) > 0 else disease_vars
    
    x = np.arange(len(k_vals))
    width = 0.25
    
    ax4.bar(x - width, norm_enrich, width, label='Enrichment', color='steelblue', alpha=0.7)
    ax4.bar(x, norm_cell, width, label='Cell specificity', color='coral', alpha=0.7)
    ax4.bar(x + width, norm_disease, width, label='Disease specificity', color='green', alpha=0.7)
    
    ax4.set_xlabel('k value', fontsize=12)
    ax4.set_ylabel('Normalized score', fontsize=12)
    ax4.set_title('Normalized Metrics Comparison', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(k_vals)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Comparison of NMF Results Across Different k Values', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, 'k_comparison_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plots saved to: {comp_dir}/k_comparison_plots.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare NMF results across k values')
    parser.add_argument('--k-values', nargs='+', type=int, default=[10, 15, 20, 25, 30],
                       help='List of k values to compare')
    parser.add_argument('--base-dir', type=str, default='nmf_results',
                       help='Base directory name for results')
    
    args = parser.parse_args()
    
    compare_k_values(args.k_values, args.base_dir)