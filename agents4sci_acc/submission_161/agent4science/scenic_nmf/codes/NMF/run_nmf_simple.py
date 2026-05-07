#!/usr/bin/env python
"""
Simple NMF analysis with k=10, minimal filtering, and visualization
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')

def run_simple_nmf(input_file='../sub_adata/strict_endocrine.h5ad', k=10):
    """
    Run simple NMF analysis with visualization
    """
    
    print("="*60)
    print("SIMPLE NMF ANALYSIS (k=10)")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    adata = sc.read_h5ad(input_file)
    print(f"Original shape: {adata.shape}")
    
    # Minimal filtering - only remove truly empty cells
    print("\n2. Minimal cell filtering...")
    sc.pp.filter_cells(adata, min_genes=10)  # Cells must express at least 10 genes
    sc.pp.filter_genes(adata, min_cells=3)   # Genes must be in at least 3 cells
    print(f"After minimal filtering: {adata.shape}")
    
    # Store raw counts
    adata.raw = adata
    
    # Normalization
    print("\n3. Normalizing data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    print("\n4. Finding highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=False)
    hvg = adata.var_names[adata.var.highly_variable].tolist()
    print(f"Found {len(hvg)} highly variable genes")
    
    # Subset to HVG for NMF
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    
    # Convert to dense array if sparse
    X = adata_hvg.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Ensure non-negative values for NMF
    X = np.maximum(X, 0)
    
    # Run NMF
    print(f"\n5. Running NMF with k={k}...")
    nmf_model = NMF(n_components=k, init='nndsvda', random_state=42, max_iter=500)
    
    # W: cell x program matrix (usage)
    # H: program x gene matrix (gene scores)
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_
    
    print(f"NMF completed!")
    print(f"Usage matrix (W): {W.shape} - cells x programs")
    print(f"Gene scores (H): {H.shape} - programs x genes")
    
    # Create output directory
    output_dir = "nmf_simple_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    print("\n6. Saving NMF results...")
    
    # Create DataFrames
    usage_df = pd.DataFrame(W, 
                           index=adata_hvg.obs_names, 
                           columns=[f'Program_{i+1}' for i in range(k)])
    
    gene_scores_df = pd.DataFrame(H.T, 
                                 columns=[f'Program_{i+1}' for i in range(k)],
                                 index=adata_hvg.var_names)
    
    # Save matrices
    usage_df.to_csv(os.path.join(output_dir, 'usage_matrix.csv'))
    gene_scores_df.to_csv(os.path.join(output_dir, 'gene_scores.csv'))
    
    # Add usage to adata
    for col in usage_df.columns:
        adata.obs[col] = usage_df[col].values
    
    # Get top genes per program
    print("\n7. Identifying top genes per program...")
    n_top_genes = 50
    top_genes_dict = {}
    
    with open(os.path.join(output_dir, 'top_genes_per_program.txt'), 'w') as f:
        for prog_idx in range(k):
            prog_name = f'Program_{prog_idx+1}'
            # Get top genes for this program
            top_genes = gene_scores_df[prog_name].nlargest(n_top_genes)
            top_genes_dict[prog_name] = top_genes.index.tolist()
            
            # Write to file
            f.write(f"\n{prog_name}:\n")
            f.write("-" * 40 + "\n")
            for i, (gene, score) in enumerate(top_genes.items(), 1):
                f.write(f"{i:3d}. {gene:15s} (score: {score:.4f})\n")
            
            # Print summary
            print(f"{prog_name}: Top genes include {', '.join(top_genes.index[:5])}")
    
    # Characterize programs based on known markers
    print("\n8. Characterizing programs based on marker genes...")
    characterize_programs(gene_scores_df, output_dir)
    
    # Create visualizations
    print("\n9. Creating visualizations...")
    create_visualizations(adata, usage_df, gene_scores_df, top_genes_dict, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("  - usage_matrix.csv: Cell x Program usage matrix")
    print("  - gene_scores.csv: Gene x Program scores")
    print("  - top_genes_per_program.txt: Top 50 genes per program")
    print("  - program_characterization.txt: Program annotation")
    print("  - figures/: Visualization plots")
    
    return adata, usage_df, gene_scores_df

def characterize_programs(gene_scores_df, output_dir):
    """
    Characterize programs based on known marker genes
    """
    
    # Define marker gene sets
    markers = {
        'Beta_cells': ['INS', 'IAPP', 'MAFA', 'PDX1', 'NKX6-1', 'UCN3', 'G6PC2', 'SLC30A8'],
        'Alpha_cells': ['GCG', 'ARX', 'IRX1', 'IRX2', 'LOXL4', 'TM4SF4'],
        'Delta_cells': ['SST', 'HHEX', 'LEPR', 'RBP4', 'GABRB3'],
        'PP_cells': ['PPY', 'SERTM1', 'CARTPT', 'ETV1', 'FXYD2'],
        'Ductal': ['KRT19', 'KRT7', 'CFTR', 'SOX9', 'MUC1', 'SPP1'],
        'Acinar': ['CPA1', 'PRSS1', 'CTRB1', 'AMY2A', 'CEL', 'PNLIP'],
        'Endothelial': ['PECAM1', 'CDH5', 'VWF', 'CD34', 'PLVAP', 'ESM1'],
        'Immune': ['PTPRC', 'CD3E', 'CD68', 'CD14', 'CD19', 'MS4A1'],
        'Stellate': ['RGS5', 'PDGFRB', 'COL1A1', 'COL3A1', 'ACTA2'],
        'Cell_cycle': ['MKI67', 'TOP2A', 'PCNA', 'CDK1', 'CCNB1', 'CCNA2'],
        'Stress': ['HSPA1A', 'HSPA1B', 'HSP90AA1', 'DNAJB1', 'HSPB1'],
        'Metabolism': ['LDHA', 'PKM', 'G6PD', 'ENO1', 'GAPDH'],
        'Secretory': ['INS', 'GCG', 'SST', 'PPY', 'CHGA', 'CHGB']
    }
    
    # Check each program for marker enrichment
    results = []
    for prog in gene_scores_df.columns:
        # Get top 100 genes for this program
        top_genes = gene_scores_df[prog].nlargest(100).index.tolist()
        
        prog_markers = {}
        for cell_type, marker_list in markers.items():
            # Count how many markers are in top genes
            overlap = [g for g in marker_list if g in top_genes]
            if overlap:
                prog_markers[cell_type] = overlap
        
        results.append({
            'Program': prog,
            'Likely_type': max(prog_markers.keys(), key=lambda k: len(prog_markers[k])) if prog_markers else 'Unknown',
            'Markers_found': prog_markers
        })
    
    # Save characterization
    with open(os.path.join(output_dir, 'program_characterization.txt'), 'w') as f:
        f.write("PROGRAM CHARACTERIZATION BASED ON MARKER GENES\n")
        f.write("="*60 + "\n\n")
        
        for res in results:
            f.write(f"{res['Program']}:\n")
            f.write(f"  Likely cell type: {res['Likely_type']}\n")
            if res['Markers_found']:
                f.write("  Markers found:\n")
                for ct, genes in res['Markers_found'].items():
                    f.write(f"    - {ct}: {', '.join(genes)}\n")
            f.write("\n")
    
    return results

def create_visualizations(adata, usage_df, gene_scores_df, top_genes_dict, output_dir):
    """
    Create visualization plots
    """
    
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Program usage heatmap
    print("   Creating usage heatmap...")
    plt.figure(figsize=(12, 8))
    
    # Normalize usage by row (cell)
    usage_norm = usage_df.div(usage_df.sum(axis=1), axis=0)
    
    # Create clustered heatmap
    sns.clustermap(usage_norm.T, 
                   cmap='YlOrRd',
                   figsize=(15, 10),
                   cbar_kws={'label': 'Normalized Usage'},
                   yticklabels=True,
                   xticklabels=False)
    plt.savefig(os.path.join(fig_dir, 'usage_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Top genes heatmap for each program
    print("   Creating top genes heatmaps...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (prog, genes) in enumerate(top_genes_dict.items()):
        if idx >= 10:
            break
        
        ax = axes[idx]
        
        # Get top 20 genes for visualization
        top20_genes = genes[:20]
        scores = gene_scores_df.loc[top20_genes, prog].values
        
        # Create bar plot
        y_pos = np.arange(len(top20_genes))
        ax.barh(y_pos, scores, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top20_genes, fontsize=8)
        ax.set_xlabel('Score')
        ax.set_title(prog)
        ax.invert_yaxis()
    
    plt.suptitle('Top 20 Genes per Program', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'top_genes_per_program.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Program activity by metadata (if available)
    print("   Checking for metadata to plot...")
    metadata_cols = []
    for col in ['cell_type', 'celltype', 'CellType', 'disease', 'Disease', 
                'tissue', 'Tissue', 'condition', 'sample', 'batch']:
        if col in adata.obs.columns:
            metadata_cols.append(col)
    
    if metadata_cols:
        meta_col = metadata_cols[0]
        print(f"   Creating activity plot by {meta_col}...")
        
        # Calculate mean activity per group
        activity_data = []
        for group in adata.obs[meta_col].unique():
            cells_in_group = adata.obs[adata.obs[meta_col] == group].index
            cells_in_group = [c for c in cells_in_group if c in usage_df.index]
            
            if cells_in_group:
                mean_usage = usage_df.loc[cells_in_group].mean()
                activity_data.append(mean_usage.values)
            else:
                activity_data.append(np.zeros(usage_df.shape[1]))
        
        activity_df = pd.DataFrame(activity_data, 
                                  columns=usage_df.columns,
                                  index=adata.obs[meta_col].unique())
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(activity_df.T, 
                   annot=True, 
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   cbar_kws={'label': 'Mean Activity'})
        plt.title(f'Program Activity by {meta_col}')
        plt.xlabel(meta_col)
        plt.ylabel('Program')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'activity_by_{meta_col}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Expression of marker genes
        print("   Creating marker gene expression plots...")
        
        # Define key marker genes to plot
        key_markers = {
            'Beta': ['INS', 'IAPP', 'MAFA'],
            'Alpha': ['GCG', 'ARX', 'IRX1'],
            'Delta': ['SST', 'HHEX'],
            'Ductal': ['KRT19', 'CFTR', 'SOX9'],
            'Immune': ['PTPRC', 'CD68', 'CD3E']
        }
        
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for cell_type, markers in key_markers.items():
            for marker in markers:
                if marker in adata.var_names and plot_idx < 15:
                    ax = axes[plot_idx]
                    
                    # Get expression values
                    expr = adata[:, marker].X
                    if hasattr(expr, 'toarray'):
                        expr = expr.toarray().flatten()
                    
                    # Create violin plot
                    plot_df = pd.DataFrame({
                        'Expression': expr,
                        meta_col: adata.obs[meta_col].values
                    })
                    
                    sns.violinplot(data=plot_df, x=meta_col, y='Expression', ax=ax)
                    ax.set_title(f'{marker} ({cell_type})')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    
                    plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Marker Gene Expression by {meta_col}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'markers_by_{meta_col}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        print("   No metadata columns found for grouping")
    
    print(f"   All figures saved to {fig_dir}/")

if __name__ == "__main__":
    # Run the analysis
    adata, usage, gene_scores = run_simple_nmf()