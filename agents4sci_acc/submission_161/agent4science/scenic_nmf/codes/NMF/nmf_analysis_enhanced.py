#!/usr/bin/env python
"""
Enhanced NMF analysis with gene program analysis and visualization
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from cnmf import cNMF
import warnings
warnings.filterwarnings('ignore')

def run_enhanced_nmf_analysis(input_file, k_value=10, n_iter=100, seed=14):
    """
    Run enhanced NMF analysis with gene program analysis and visualization
    """
    
    print("="*60)
    print("ENHANCED NMF ANALYSIS")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    adata = sc.read_h5ad(input_file)
    print(f"Original data shape: {adata.shape}")
    print(f"Cells: {adata.n_obs:,}")
    print(f"Genes: {adata.n_vars:,}")
    
    # Less aggressive cell filtering - only remove truly empty cells
    print("\n2. Minimal cell filtering...")
    if 'n_counts' not in adata.obs.columns:
        if hasattr(adata.X, 'toarray'):
            adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        else:
            adata.obs['n_counts'] = adata.X.sum(axis=1)
    
    # Much lower threshold - only filter out truly empty/dead cells
    min_counts = 10  # Very minimal filtering
    min_genes = 5    # Cell must express at least 5 genes
    
    # Calculate number of genes per cell
    if hasattr(adata.X, 'toarray'):
        adata.obs['n_genes'] = np.array((adata.X > 0).sum(axis=1)).flatten()
    else:
        adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
    
    cells_before = adata.n_obs
    adata = adata[(adata.obs['n_counts'] > min_counts) & (adata.obs['n_genes'] > min_genes), :].copy()
    cells_after = adata.n_obs
    
    print(f"Removed only {cells_before - cells_after} cells (truly empty cells)")
    print(f"Retained {cells_after:,} cells ({100*cells_after/cells_before:.1f}% of original)")
    print(f"Filtered data shape: {adata.shape}")
    
    # Setup output directories
    base_name = "enhanced_nmf_k10"
    output_dir = "nmf_enhanced_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the minimally filtered counts
    counts_file = os.path.join(output_dir, f"{base_name}_counts.h5ad")
    adata.write_h5ad(counts_file)
    print(f"\nSaved filtered counts to: {counts_file}")
    
    # Initialize cNMF
    print("\n3. Initializing cNMF...")
    cnmf_obj = cNMF(
        output_dir=output_dir,
        name=base_name
    )
    
    # Prepare with more genes for better resolution
    print("\n4. Preparing data with high variable genes...")
    cnmf_obj.prepare(
        counts_fn=counts_file,
        components=[k_value],
        n_iter=n_iter,
        seed=seed,
        num_highvar_genes=3000  # Use more genes for better program definition
    )
    
    # Factorize
    print("\n5. Running factorization...")
    cnmf_obj.factorize(worker_i=0, total_workers=1)
    
    # Combine
    print("\n6. Combining results...")
    cnmf_obj.combine(components=[k_value])
    
    # K selection plot
    print("\n7. Creating K selection plot...")
    cnmf_obj.k_selection_plot()
    
    # Consensus
    print("\n8. Computing consensus...")
    density_threshold = 0.1
    cnmf_obj.consensus(
        k=k_value,
        density_threshold=density_threshold,
        show_clustering=False,
        close_clustergram_fig=True
    )
    
    print("\n9. Loading and analyzing results...")
    
    # Load the gene scores
    gene_scores_file = os.path.join(output_dir, base_name, 
                                    f"{base_name}.gene_spectra_score.k_{k_value}.dt_{str(density_threshold).replace('.', '_')}.txt")
    
    if os.path.exists(gene_scores_file):
        # Gene scores file has genes as rows and programs as columns
        gene_scores = pd.read_csv(gene_scores_file, sep='\t', index_col=0)
        print(f"Loaded gene scores: {gene_scores.shape}")
        print(f"Number of genes: {gene_scores.shape[0]}")
        print(f"Number of programs: {gene_scores.shape[1]}")
        print(f"Program names: {gene_scores.columns.tolist()}")
        
        # Get gene names from adata
        gene_names = adata.var_names.tolist()
        
        # Ensure we have the right number of genes
        if len(gene_names) == gene_scores.shape[0]:
            gene_scores.index = gene_names
        
        # Get top genes for each program
        n_top_genes = 50  # Top 50 genes per program
        top_genes_per_program = {}
        
        print(f"\n10. Identifying top {n_top_genes} genes for each program...")
        for idx, program in enumerate(gene_scores.columns):
            # Get top scoring genes for this program
            top_genes = gene_scores[program].nlargest(n_top_genes).index.tolist()
            top_genes_per_program[f"Program_{idx+1}"] = top_genes
            print(f"   Program {idx+1}: Found {len(top_genes)} top genes")
        
        # Save top genes
        top_genes_file = os.path.join(output_dir, "top_genes_per_program.txt")
        with open(top_genes_file, 'w') as f:
            for program, genes in top_genes_per_program.items():
                f.write(f"{program}:\n")
                for gene in genes[:20]:  # Show top 20 in file
                    f.write(f"  {gene}\n")
                f.write("\n")
        print(f"Saved top genes to: {top_genes_file}")
        
        # Perform gene program analysis
        perform_gene_program_analysis(adata, top_genes_per_program, gene_scores, output_dir)
        
        # Load usage matrix for visualization
        usage_file = os.path.join(output_dir, base_name, 
                                  f"{base_name}.usages.k_{k_value}.dt_{str(density_threshold).replace('.', '_')}.consensus.txt")
        
        if os.path.exists(usage_file):
            usage_matrix = pd.read_csv(usage_file, sep='\t', index_col=0)
            print(f"\nLoaded usage matrix: {usage_matrix.shape}")
            
            # Create visualizations
            create_program_visualizations(adata, usage_matrix, top_genes_per_program, 
                                        gene_scores, output_dir)
        else:
            print(f"Usage matrix file not found: {usage_file}")
    else:
        print(f"Gene scores file not found: {gene_scores_file}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    
    return output_dir

def perform_gene_program_analysis(adata, top_genes_per_program, gene_scores, output_dir):
    """
    Perform functional enrichment and characterization of gene programs
    """
    print("\n11. Performing gene program analysis...")
    
    # Create program characterization summary
    program_summary = []
    
    for program, genes in top_genes_per_program.items():
        # Get genes that are actually in the dataset
        genes_in_data = [g for g in genes if g in adata.var_names]
        
        # Calculate some basic statistics
        program_info = {
            'Program': program,
            'N_top_genes': len(genes),
            'N_genes_in_data': len(genes_in_data),
            'Top_5_genes': ', '.join(genes[:5])
        }
        
        # Add marker gene categories (you can expand this based on known markers)
        marker_categories = categorize_genes(genes[:20])
        program_info.update(marker_categories)
        
        program_summary.append(program_info)
    
    # Save program summary
    summary_df = pd.DataFrame(program_summary)
    summary_file = os.path.join(output_dir, "program_characterization.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved program characterization to: {summary_file}")
    
    return summary_df

def categorize_genes(gene_list):
    """
    Categorize genes based on known markers
    This is a simplified version - expand with your specific markers
    """
    # Define marker gene sets (expand this based on your knowledge)
    marker_sets = {
        'Beta_cells': ['INS', 'IAPP', 'MAFA', 'PDX1', 'NKX6-1', 'UCN3', 'G6PC2'],
        'Alpha_cells': ['GCG', 'ARX', 'IRX1', 'IRX2', 'LOXL4'],
        'Delta_cells': ['SST', 'HHEX', 'LEPR', 'RBP4'],
        'PP_cells': ['PPY', 'SERTM1', 'CARTPT'],
        'Ductal': ['KRT19', 'KRT7', 'CFTR', 'SOX9', 'MUC1'],
        'Acinar': ['CPA1', 'PRSS1', 'CTRB1', 'AMY2A', 'CEL'],
        'Endothelial': ['PECAM1', 'CDH5', 'VWF', 'CD34'],
        'Immune': ['PTPRC', 'CD3E', 'CD68', 'CD14', 'CD19'],
        'Stellate': ['RGS5', 'PDGFRB', 'COL1A1', 'COL3A1'],
        'Stress_response': ['HSPA1A', 'HSPA1B', 'HSP90AA1', 'DNAJB1'],
        'Cell_cycle': ['MKI67', 'TOP2A', 'PCNA', 'CDK1', 'CCNB1'],
        'Apoptosis': ['CASP3', 'CASP8', 'BAX', 'BCL2', 'TP53']
    }
    
    categories = {}
    for category, markers in marker_sets.items():
        overlap = [g for g in gene_list if g in markers]
        if overlap:
            categories[f'{category}_markers'] = ', '.join(overlap)
    
    return categories

def create_program_visualizations(adata, usage_matrix, top_genes_per_program, 
                                 gene_scores, output_dir):
    """
    Create comprehensive visualizations for gene programs
    """
    print("\n12. Creating visualizations...")
    
    # Ensure cell IDs match between adata and usage matrix
    common_cells = list(set(adata.obs_names) & set(usage_matrix.index))
    if len(common_cells) == 0:
        print("Warning: No matching cells between adata and usage matrix")
        return
    
    print(f"Found {len(common_cells)} cells in both datasets")
    
    # Subset to common cells
    adata_vis = adata[common_cells, :].copy()
    usage_vis = usage_matrix.loc[common_cells, :]
    
    # Add usage scores to adata
    for col in usage_vis.columns:
        adata_vis.obs[f'NMF_{col}'] = usage_vis[col].values
    
    # Create figure directory
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Heatmap of usage matrix by metadata
    create_usage_heatmap(adata_vis, usage_vis, fig_dir)
    
    # 2. Top genes expression by metadata
    create_top_genes_plots(adata_vis, top_genes_per_program, fig_dir)
    
    # 3. Program activity by cell type/condition
    create_program_activity_plots(adata_vis, usage_vis, fig_dir)
    
    print(f"Saved all figures to: {fig_dir}/")

def create_usage_heatmap(adata, usage_matrix, fig_dir):
    """
    Create heatmap of NMF usage matrix organized by metadata
    """
    print("   Creating usage heatmap...")
    
    # Check available metadata columns
    metadata_cols = []
    for col in ['cell_type', 'celltype', 'CellType', 'disease', 'Disease', 
                'tissue', 'Tissue', 'condition', 'sample']:
        if col in adata.obs.columns:
            metadata_cols.append(col)
    
    if not metadata_cols:
        print("   Warning: No metadata columns found for visualization")
        return
    
    # Create clustered heatmap
    plt.figure(figsize=(12, 8))
    
    # Normalize usage matrix
    usage_norm = usage_matrix.div(usage_matrix.sum(axis=1), axis=0)
    
    # Create heatmap
    sns.clustermap(usage_norm.T, 
                   cmap='RdBu_r', 
                   center=0,
                   figsize=(15, 10),
                   cbar_kws={'label': 'Normalized Usage'},
                   yticklabels=True,
                   xticklabels=False)
    
    plt.savefig(os.path.join(fig_dir, 'usage_heatmap_clustered.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create heatmap for each metadata category
    for meta_col in metadata_cols[:3]:  # Limit to first 3 metadata columns
        print(f"   Creating heatmap for {meta_col}...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort cells by metadata
        sorted_cells = adata.obs.sort_values(meta_col).index
        usage_sorted = usage_norm.loc[sorted_cells, :]
        
        # Create annotation dataframe
        annot_df = adata.obs.loc[sorted_cells, [meta_col]]
        
        # Plot
        sns.heatmap(usage_sorted.T, 
                   cmap='RdBu_r',
                   center=0,
                   cbar_kws={'label': 'Normalized Usage'},
                   yticklabels=True,
                   xticklabels=False,
                   ax=ax)
        
        ax.set_xlabel('Cells')
        ax.set_ylabel('NMF Programs')
        ax.set_title(f'NMF Program Usage by {meta_col}')
        
        plt.savefig(os.path.join(fig_dir, f'usage_heatmap_by_{meta_col}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

def create_top_genes_plots(adata, top_genes_per_program, fig_dir):
    """
    Create plots showing top gene expression by metadata
    """
    print("   Creating top genes expression plots...")
    
    # Check available metadata
    metadata_cols = []
    for col in ['cell_type', 'celltype', 'CellType', 'disease', 'Disease', 
                'tissue', 'Tissue', 'condition']:
        if col in adata.obs.columns:
            metadata_cols.append(col)
    
    if not metadata_cols:
        print("   Warning: No metadata columns found")
        return
    
    # For each program, plot top genes
    for program_idx, (program, genes) in enumerate(top_genes_per_program.items()):
        if program_idx >= 3:  # Limit to first 3 programs for brevity
            break
            
        print(f"   Plotting {program} top genes...")
        
        # Get top 10 genes that exist in the data
        genes_to_plot = [g for g in genes[:10] if g in adata.var_names]
        
        if len(genes_to_plot) == 0:
            continue
        
        # Create expression matrix for these genes
        gene_exp = adata[:, genes_to_plot].X
        if hasattr(gene_exp, 'toarray'):
            gene_exp = gene_exp.toarray()
        
        # Create plot for each metadata column
        for meta_col in metadata_cols[:2]:  # Limit to first 2 metadata columns
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            for idx, gene in enumerate(genes_to_plot[:10]):
                ax = axes[idx]
                
                # Get gene expression
                gene_idx = adata.var_names.get_loc(gene)
                exp_values = gene_exp[:, genes_to_plot.index(gene)]
                
                # Create violin plot by metadata
                plot_df = pd.DataFrame({
                    'Expression': exp_values,
                    meta_col: adata.obs[meta_col].values
                })
                
                # Plot
                sns.violinplot(data=plot_df, x=meta_col, y='Expression', ax=ax)
                ax.set_title(gene)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Reduce y-label for space
                if idx % 5 != 0:
                    ax.set_ylabel('')
            
            plt.suptitle(f'{program} Top Genes Expression by {meta_col}')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'program_{program}_genes_by_{meta_col}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()

def create_program_activity_plots(adata, usage_matrix, fig_dir):
    """
    Create plots showing program activity by cell type and condition
    """
    print("   Creating program activity plots...")
    
    # Check available metadata
    metadata_cols = []
    for col in ['cell_type', 'celltype', 'CellType', 'disease', 'Disease', 
                'tissue', 'Tissue', 'condition', 'sample']:
        if col in adata.obs.columns:
            metadata_cols.append(col)
    
    if not metadata_cols:
        print("   Warning: No metadata columns found")
        return
    
    # Normalize usage matrix
    usage_norm = usage_matrix.div(usage_matrix.sum(axis=1), axis=0)
    
    # Plot program activity by each metadata category
    for meta_col in metadata_cols[:3]:  # Limit to first 3 metadata columns
        print(f"   Plotting program activity by {meta_col}...")
        
        # Calculate mean activity per group
        activity_df = pd.DataFrame(index=usage_norm.columns)
        
        for group in adata.obs[meta_col].unique():
            cells_in_group = adata.obs[adata.obs[meta_col] == group].index
            cells_in_group = [c for c in cells_in_group if c in usage_norm.index]
            
            if len(cells_in_group) > 0:
                mean_activity = usage_norm.loc[cells_in_group, :].mean()
                activity_df[group] = mean_activity
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(activity_df, 
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Mean Program Activity'})
        
        plt.title(f'Mean NMF Program Activity by {meta_col}')
        plt.xlabel(meta_col)
        plt.ylabel('NMF Program')
        plt.tight_layout()
        
        plt.savefig(os.path.join(fig_dir, f'program_activity_by_{meta_col}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create a grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 6))
        activity_df.T.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xlabel(meta_col)
        ax.set_ylabel('Mean Program Activity')
        ax.set_title(f'NMF Program Activity Distribution by {meta_col}')
        ax.legend(title='Program', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(fig_dir, f'program_activity_barplot_{meta_col}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Input file
    input_file = '../sub_adata/strict_endocrine.h5ad'
    
    # Run enhanced analysis
    output_dir = run_enhanced_nmf_analysis(
        input_file=input_file,
        k_value=10,
        n_iter=100,  # Use 100 iterations for better results
        seed=14
    )
    
    print(f"\nAll results saved to: {output_dir}/")
    print("\nKey outputs:")
    print("  - Top genes per program: top_genes_per_program.txt")
    print("  - Program characterization: program_characterization.csv")
    print("  - Visualizations: figures/")
    print("    - Usage heatmaps by metadata")
    print("    - Top gene expression by cell type/disease")
    print("    - Program activity summaries")