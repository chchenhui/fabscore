#!/usr/bin/env python
"""
Enhanced NMF analysis with pathway enrichment and multiple metadata visualizations
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

# For enrichment analysis
try:
    import gseapy as gp
except ImportError:
    print("Installing gseapy for enrichment analysis...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'gseapy'])
    import gseapy as gp

def run_nmf_with_enrichment(input_file='../sub_adata/strict_endocrine.h5ad', k=10):
    """
    Run NMF analysis with enrichment and multi-metadata visualization
    """
    
    print("="*60)
    print("ENHANCED NMF ANALYSIS WITH ENRICHMENT")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    adata = sc.read_h5ad(input_file)
    print(f"Original shape: {adata.shape}")
    
    # Check available metadata
    print("\n2. Checking metadata columns...")
    metadata_found = {}
    for col_type in ['cell_type', 'tissue', 'disease']:
        found_cols = []
        for col in adata.obs.columns:
            if col_type in col.lower():
                found_cols.append(col)
        if found_cols:
            metadata_found[col_type] = found_cols[0]
            print(f"  Found {col_type}: {found_cols[0]}")
    
    # Minimal filtering
    print("\n3. Minimal cell filtering...")
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"After filtering: {adata.shape}")
    
    # Store raw counts
    adata.raw = adata
    
    # Normalization
    print("\n4. Normalizing data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    print("\n5. Finding highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=False)
    hvg = adata.var_names[adata.var.highly_variable].tolist()
    
    # Subset to HVG
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    
    # Convert to dense array
    X = adata_hvg.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.maximum(X, 0)
    
    # Run NMF
    print(f"\n6. Running NMF with k={k}...")
    nmf_model = NMF(n_components=k, init='nndsvda', random_state=42, max_iter=500)
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_
    
    # Create output directory
    output_dir = "nmf_enrichment_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'enrichment'), exist_ok=True)
    
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
    
    # Get top genes per program
    print("\n7. Identifying top genes per program...")
    n_top_genes = 100  # Get more genes for enrichment
    top_genes_dict = {}
    
    with open(os.path.join(output_dir, 'top_genes_per_program.txt'), 'w') as f:
        for prog_idx in range(k):
            prog_name = f'Program_{prog_idx+1}'
            top_genes = gene_scores_df[prog_name].nlargest(n_top_genes)
            top_genes_dict[prog_name] = top_genes.index.tolist()
            
            f.write(f"\n{prog_name}:\n")
            f.write("-" * 40 + "\n")
            for i, (gene, score) in enumerate(top_genes.head(50).items(), 1):
                f.write(f"{i:3d}. {gene:15s} (score: {score:.4f})\n")
    
    # Create visualizations
    print("\n8. Creating visualizations...")
    create_activity_plots(adata, usage_df, metadata_found, output_dir)
    create_top_genes_heatmap(gene_scores_df, top_genes_dict, output_dir)
    create_usage_heatmap(usage_df, output_dir)
    
    # Perform enrichment analysis
    print("\n9. Performing KEGG/GO enrichment analysis...")
    perform_enrichment_analysis(top_genes_dict, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("\nKey outputs:")
    print("  Figures:")
    print("    - figures/usage_heatmap.png")
    print("    - figures/top_genes_per_program.png")
    print("    - figures/activity_by_cell_type.png")
    print("    - figures/activity_by_tissue.png")
    print("    - figures/activity_by_disease.png")
    print("  Enrichment:")
    print("    - enrichment/Program_X_GO_enrichment.csv")
    print("    - enrichment/Program_X_KEGG_enrichment.csv")
    print("    - enrichment/Program_X_enrichment_plot.png")
    
    return adata, usage_df, gene_scores_df

def create_activity_plots(adata, usage_df, metadata_found, output_dir):
    """
    Create activity plots for all available metadata
    """
    fig_dir = os.path.join(output_dir, 'figures')
    
    # Normalize usage by row
    usage_norm = usage_df.div(usage_df.sum(axis=1), axis=0)
    
    # Create plots for each metadata type
    for meta_type, meta_col in metadata_found.items():
        print(f"   Creating activity plot by {meta_type}...")
        
        # Calculate mean activity per group
        activity_data = []
        group_names = []
        
        for group in adata.obs[meta_col].unique():
            cells_in_group = adata.obs[adata.obs[meta_col] == group].index
            cells_in_group = [c for c in cells_in_group if c in usage_norm.index]
            
            if cells_in_group:
                mean_usage = usage_norm.loc[cells_in_group].mean()
                activity_data.append(mean_usage.values)
                group_names.append(str(group))
        
        if activity_data:
            activity_df = pd.DataFrame(activity_data, 
                                      columns=usage_df.columns,
                                      index=group_names)
            
            # Save CSV
            activity_df.to_csv(os.path.join(output_dir, f'activity_by_{meta_type}.csv'))
            
            # Create heatmap
            plt.figure(figsize=(10, max(6, len(group_names)*0.3)))
            sns.heatmap(activity_df.T, 
                       annot=True, 
                       fmt='.2f',
                       cmap='RdBu_r',
                       center=0,
                       cbar_kws={'label': 'Mean Activity'},
                       linewidths=0.5)
            plt.title(f'Program Activity by {meta_type.replace("_", " ").title()}')
            plt.xlabel(meta_type.replace("_", " ").title())
            plt.ylabel('Program')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'activity_by_{meta_type}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Also create a bar plot version
            fig, ax = plt.subplots(figsize=(12, 6))
            activity_df.T.plot(kind='bar', ax=ax, width=0.8)
            ax.set_xlabel('Program')
            ax.set_ylabel('Mean Activity')
            ax.set_title(f'Program Activity Distribution by {meta_type.replace("_", " ").title()}')
            ax.legend(title=meta_type.replace("_", " ").title(), 
                     bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'activity_barplot_by_{meta_type}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()

def create_top_genes_heatmap(gene_scores_df, top_genes_dict, output_dir):
    """
    Create top genes visualization
    """
    fig_dir = os.path.join(output_dir, 'figures')
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (prog, genes) in enumerate(top_genes_dict.items()):
        if idx >= 10:
            break
        
        ax = axes[idx]
        
        # Get top 20 genes
        top20_genes = genes[:20]
        scores = gene_scores_df.loc[top20_genes, prog].values
        
        # Create bar plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top20_genes)))
        y_pos = np.arange(len(top20_genes))
        ax.barh(y_pos, scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top20_genes, fontsize=8)
        ax.set_xlabel('Score')
        ax.set_title(prog, fontsize=10, fontweight='bold')
        ax.invert_yaxis()
    
    plt.suptitle('Top 20 Genes per Program', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'top_genes_per_program.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()

def create_usage_heatmap(usage_df, output_dir):
    """
    Create clustered usage heatmap
    """
    fig_dir = os.path.join(output_dir, 'figures')
    
    # Normalize usage
    usage_norm = usage_df.div(usage_df.sum(axis=1), axis=0)
    
    # Create clustered heatmap
    g = sns.clustermap(usage_norm.T, 
                       cmap='YlOrRd',
                       figsize=(15, 10),
                       cbar_kws={'label': 'Normalized Usage'},
                       yticklabels=True,
                       xticklabels=False,
                       linewidths=0,
                       rasterized=True)
    g.ax_heatmap.set_xlabel('Cells', fontsize=12)
    g.ax_heatmap.set_ylabel('Programs', fontsize=12)
    plt.suptitle('NMF Program Usage Matrix (Clustered)', y=1.02, fontsize=14)
    plt.savefig(os.path.join(fig_dir, 'usage_heatmap.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()

def perform_enrichment_analysis(top_genes_dict, output_dir):
    """
    Perform GO and KEGG enrichment analysis for each program
    """
    enrich_dir = os.path.join(output_dir, 'enrichment')
    
    # Gene sets to use
    gene_sets = {
        'GO_Biological_Process': 'GO_Biological_Process_2021',
        'GO_Molecular_Function': 'GO_Molecular_Function_2021', 
        'GO_Cellular_Component': 'GO_Cellular_Component_2021',
        'KEGG': 'KEGG_2021_Human'
    }
    
    all_enrichment_results = {}
    
    for prog_name, gene_list in top_genes_dict.items():
        print(f"   Analyzing {prog_name}...")
        
        # Use top 50 genes for enrichment
        genes_for_enrichment = gene_list[:50]
        
        prog_results = {}
        
        # Perform enrichment for each gene set
        for gs_name, gs_library in gene_sets.items():
            try:
                # Run enrichment using Enrichr
                enr = gp.enrichr(gene_list=genes_for_enrichment,
                               gene_sets=gs_library,
                               organism='human',
                               outdir=None,
                               cutoff=0.05)
                
                if enr.results is not None and not enr.results.empty:
                    # Store results
                    prog_results[gs_name] = enr.results
                    
                    # Save to CSV
                    csv_file = os.path.join(enrich_dir, f'{prog_name}_{gs_name}_enrichment.csv')
                    enr.results.head(20).to_csv(csv_file, index=False)
                    
            except Exception as e:
                print(f"     Warning: Could not perform {gs_name} enrichment for {prog_name}: {e}")
        
        all_enrichment_results[prog_name] = prog_results
        
        # Create combined enrichment plot
        if prog_results:
            create_enrichment_plot(prog_name, prog_results, enrich_dir)
    
    # Create summary enrichment heatmap
    create_enrichment_summary(all_enrichment_results, enrich_dir)
    
    return all_enrichment_results

def create_enrichment_plot(prog_name, prog_results, enrich_dir):
    """
    Create enrichment visualization for a program
    """
    # Combine top results from each category
    top_terms = []
    
    for gs_name, results in prog_results.items():
        if results is not None and not results.empty:
            # Get top 3 terms from each category
            top_3 = results.nsmallest(3, 'Adjusted P-value')[['Term', 'Adjusted P-value', 'Odds Ratio']]
            top_3['Category'] = gs_name.replace('_', ' ')
            top_terms.append(top_3)
    
    if top_terms:
        combined_df = pd.concat(top_terms, ignore_index=True)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(combined_df)*0.3)))
        
        # Plot 1: -log10(p-value)
        combined_df['neg_log_pval'] = -np.log10(combined_df['Adjusted P-value'] + 1e-10)
        combined_df = combined_df.sort_values('neg_log_pval')
        
        colors = {'GO Biological Process': 'steelblue',
                 'GO Molecular Function': 'orange',
                 'GO Cellular Component': 'green',
                 'KEGG': 'red'}
        
        bar_colors = [colors.get(cat, 'gray') for cat in combined_df['Category']]
        
        y_pos = np.arange(len(combined_df))
        ax1.barh(y_pos, combined_df['neg_log_pval'].values, color=bar_colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([term[:40] + '...' if len(term) > 40 else term 
                            for term in combined_df['Term']], fontsize=8)
        ax1.set_xlabel('-log10(Adjusted P-value)')
        ax1.set_title(f'{prog_name}: Enrichment Significance')
        ax1.axvline(x=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Odds Ratio
        ax2.barh(y_pos, combined_df['Odds Ratio'].values, color=bar_colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])
        ax2.set_xlabel('Odds Ratio')
        ax2.set_title(f'{prog_name}: Effect Size')
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
        labels = list(colors.keys())
        ax2.legend(handles, labels, loc='lower right', fontsize=8)
        
        plt.suptitle(f'{prog_name} Enrichment Analysis', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(enrich_dir, f'{prog_name}_enrichment_plot.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

def create_enrichment_summary(all_results, enrich_dir):
    """
    Create a summary heatmap of enrichments across all programs
    """
    # Collect top enriched terms for each program
    summary_data = {}
    
    for prog_name, prog_results in all_results.items():
        top_terms = []
        for gs_name, results in prog_results.items():
            if results is not None and not results.empty:
                # Get top term
                top_term = results.nsmallest(1, 'Adjusted P-value')
                if not top_term.empty:
                    term = top_term.iloc[0]['Term']
                    pval = top_term.iloc[0]['Adjusted P-value']
                    top_terms.append(f"{term[:30]} (p={pval:.2e})")
        
        if top_terms:
            summary_data[prog_name] = ', '.join(top_terms[:2])  # Show top 2
    
    if summary_data:
        # Save summary
        summary_df = pd.DataFrame(list(summary_data.items()), 
                                columns=['Program', 'Top Enriched Terms'])
        summary_df.to_csv(os.path.join(enrich_dir, 'enrichment_summary.csv'), index=False)
        
        print("\n   Enrichment Summary:")
        for _, row in summary_df.iterrows():
            print(f"     {row['Program']}: {row['Top Enriched Terms'][:80]}...")

if __name__ == "__main__":
    # Run the analysis
    adata, usage, gene_scores = run_nmf_with_enrichment()