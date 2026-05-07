#!/usr/bin/env python
"""
Complete NMF Analysis Pipeline with Configurable k
- Minimal cell filtering
- NMF decomposition with configurable k
- Activity plots by cell_type, disease, tissue
- KEGG/GO enrichment analysis
- Combined enrichment visualizations
"""

import os
import sys
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
    subprocess.check_call(['pip', 'install', '-q', 'gseapy'])
    import gseapy as gp

class NMFAnalysisPipeline:
    """
    Complete NMF analysis pipeline with enrichment
    """
    
    def __init__(self, input_file='../sub_adata/strict_endocrine.h5ad', k=20, output_base='nmf_results'):
        """
        Initialize the pipeline
        
        Parameters:
        -----------
        input_file : str
            Path to input h5ad file
        k : int
            Number of NMF components
        output_base : str
            Base directory for outputs
        """
        self.input_file = input_file
        self.k = k
        self.output_dir = f"{output_base}_k{k}"
        self.fig_dir = os.path.join(self.output_dir, 'figures')
        self.enrich_dir = os.path.join(self.output_dir, 'enrichment')
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.enrich_dir, exist_ok=True)
        
        # Initialize data containers
        self.adata = None
        self.adata_hvg = None
        self.usage_df = None
        self.gene_scores_df = None
        self.top_genes_dict = {}
        
        print(f"="*60)
        print(f"NMF ANALYSIS PIPELINE - k={k}")
        print(f"="*60)
        print(f"Output directory: {self.output_dir}")
    
    def load_and_filter_data(self):
        """
        Load data with minimal filtering
        """
        print("\n1. Loading and filtering data...")
        self.adata = sc.read_h5ad(self.input_file)
        print(f"   Original shape: {self.adata.shape}")
        
        # Minimal filtering
        sc.pp.filter_cells(self.adata, min_genes=10)
        sc.pp.filter_genes(self.adata, min_cells=3)
        print(f"   After minimal filtering: {self.adata.shape}")
        
        # Check metadata columns
        self.metadata_cols = {}
        for col_name in ['cell_type', 'disease', 'tissue']:
            if col_name in self.adata.obs.columns:
                self.metadata_cols[col_name] = col_name
                unique_vals = self.adata.obs[col_name].nunique()
                print(f"   Found {col_name}: {unique_vals} unique values")
        
        # Store raw counts
        self.adata.raw = self.adata
        
        # Normalization
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Find highly variable genes
        print("   Finding highly variable genes...")
        sc.pp.highly_variable_genes(self.adata, n_top_genes=3000, subset=False)
        
        # Subset to HVG
        self.adata_hvg = self.adata[:, self.adata.var.highly_variable].copy()
        print(f"   Selected {self.adata_hvg.shape[1]} highly variable genes")
    
    def run_nmf(self):
        """
        Run NMF decomposition
        """
        print(f"\n2. Running NMF with k={self.k}...")
        
        # Prepare matrix
        X = self.adata_hvg.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.maximum(X, 0)
        
        # Run NMF
        nmf_model = NMF(n_components=self.k, init='nndsvda', random_state=42, max_iter=500)
        W = nmf_model.fit_transform(X)  # Cell x Program
        H = nmf_model.components_        # Program x Gene
        
        print(f"   NMF completed!")
        print(f"   Usage matrix (W): {W.shape}")
        print(f"   Gene scores (H): {H.shape}")
        
        # Create DataFrames
        self.usage_df = pd.DataFrame(
            W, 
            index=self.adata_hvg.obs_names,
            columns=[f'Program_{i+1}' for i in range(self.k)]
        )
        
        self.gene_scores_df = pd.DataFrame(
            H.T,
            columns=[f'Program_{i+1}' for i in range(self.k)],
            index=self.adata_hvg.var_names
        )
        
        # Save matrices
        self.usage_df.to_csv(os.path.join(self.output_dir, 'usage_matrix.csv'))
        self.gene_scores_df.to_csv(os.path.join(self.output_dir, 'gene_scores.csv'))
        
        # Add usage to adata
        for col in self.usage_df.columns:
            self.adata.obs[col] = self.usage_df[col].values
    
    def identify_top_genes(self):
        """
        Identify top genes for each program
        """
        print(f"\n3. Identifying top genes for {self.k} programs...")
        
        n_top = 100  # Get top 100 for enrichment
        
        with open(os.path.join(self.output_dir, 'top_genes_per_program.txt'), 'w') as f:
            for prog_idx in range(self.k):
                prog_name = f'Program_{prog_idx+1}'
                top_genes = self.gene_scores_df[prog_name].nlargest(n_top)
                self.top_genes_dict[prog_name] = top_genes.index.tolist()
                
                # Write to file
                f.write(f"\n{prog_name}:\n")
                f.write("-" * 50 + "\n")
                for i, (gene, score) in enumerate(top_genes.head(30).items(), 1):
                    f.write(f"{i:3d}. {gene:20s} score: {score:.4f}\n")
                
                # Print summary
                top5 = ', '.join(top_genes.index[:5])
                print(f"   {prog_name}: {top5}")
    
    def create_usage_heatmap(self):
        """
        Create clustered usage heatmap
        """
        print("   Creating usage heatmap...")
        
        # Normalize usage
        usage_norm = self.usage_df.div(self.usage_df.sum(axis=1), axis=0)
        
        # Create clustered heatmap
        g = sns.clustermap(
            usage_norm.T,
            cmap='YlOrRd',
            figsize=(15, max(8, self.k*0.4)),
            cbar_kws={'label': 'Normalized Usage'},
            yticklabels=True,
            xticklabels=False,
            linewidths=0,
            rasterized=True
        )
        
        g.ax_heatmap.set_xlabel('Cells', fontsize=12)
        g.ax_heatmap.set_ylabel('Programs', fontsize=12)
        plt.suptitle(f'NMF Program Usage Matrix (k={self.k})', y=1.02, fontsize=14)
        
        plt.savefig(os.path.join(self.fig_dir, 'usage_heatmap.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_top_genes_plot(self):
        """
        Create top genes visualization
        """
        print("   Creating top genes plot...")
        
        # Calculate grid dimensions
        n_cols = 5
        n_rows = int(np.ceil(self.k / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(self.k):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            prog_name = f'Program_{idx+1}'
            top_genes = self.top_genes_dict[prog_name][:20]
            scores = self.gene_scores_df.loc[top_genes, prog_name].values
            
            # Create bar plot
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_genes)))
            y_pos = np.arange(len(top_genes))
            ax.barh(y_pos, scores, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_genes, fontsize=8)
            ax.set_xlabel('Score', fontsize=9)
            ax.set_title(prog_name, fontsize=10, fontweight='bold')
            ax.invert_yaxis()
        
        # Hide unused subplots
        for idx in range(self.k, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'Top 20 Genes per Program (k={self.k})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'top_genes_per_program.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_activity_plots(self):
        """
        Create activity plots for all metadata
        """
        print("   Creating activity plots...")
        
        # Normalize usage
        usage_norm = self.usage_df.div(self.usage_df.sum(axis=1), axis=0)
        
        for meta_name, meta_col in self.metadata_cols.items():
            print(f"     - Activity by {meta_name}")
            
            # Calculate mean activity per group
            activity_data = []
            group_names = []
            
            for group in self.adata.obs[meta_col].unique():
                cells_in_group = self.adata.obs[self.adata.obs[meta_col] == group].index
                cells_in_group = [c for c in cells_in_group if c in usage_norm.index]
                
                if cells_in_group:
                    mean_usage = usage_norm.loc[cells_in_group].mean()
                    activity_data.append(mean_usage.values)
                    group_names.append(str(group))
            
            if activity_data:
                activity_df = pd.DataFrame(
                    activity_data,
                    columns=self.usage_df.columns,
                    index=group_names
                )
                
                # Save CSV
                activity_df.to_csv(os.path.join(self.output_dir, f'activity_by_{meta_name}.csv'))
                
                # Create heatmap
                plt.figure(figsize=(max(10, self.k*0.5), max(6, len(group_names)*0.3)))
                sns.heatmap(
                    activity_df.T,
                    annot=True if len(group_names) <= 15 else False,
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    cbar_kws={'label': 'Mean Activity'},
                    linewidths=0.5,
                    linecolor='gray'
                )
                plt.title(f'Program Activity by {meta_name.title()} (k={self.k})', fontsize=14)
                plt.xlabel(meta_name.title(), fontsize=12)
                plt.ylabel('Program', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(self.fig_dir, f'activity_by_{meta_name}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
    
    def perform_enrichment(self):
        """
        Perform GO and KEGG enrichment analysis
        """
        print(f"\n4. Performing enrichment analysis for {self.k} programs...")
        
        # Gene sets to use
        gene_sets = {
            'GO_BP': 'GO_Biological_Process_2021',
            'GO_MF': 'GO_Molecular_Function_2021',
            'GO_CC': 'GO_Cellular_Component_2021',
            'KEGG': 'KEGG_2021_Human'
        }
        
        self.all_enrichment_results = {}
        
        for prog_idx in range(self.k):
            prog_name = f'Program_{prog_idx+1}'
            print(f"   Analyzing {prog_name}...")
            
            # Use top 50 genes for enrichment
            genes_for_enrichment = self.top_genes_dict[prog_name][:50]
            
            prog_results = {}
            
            for gs_name, gs_library in gene_sets.items():
                try:
                    # Run enrichment
                    enr = gp.enrichr(
                        gene_list=genes_for_enrichment,
                        gene_sets=gs_library,
                        organism='human',
                        outdir=None,
                        cutoff=0.05
                    )
                    
                    if enr.results is not None and not enr.results.empty:
                        prog_results[gs_name] = enr.results
                        
                        # Save CSV
                        csv_file = os.path.join(self.enrich_dir, f'{prog_name}_{gs_name}_enrichment.csv')
                        enr.results.head(20).to_csv(csv_file, index=False)
                        
                except Exception as e:
                    pass  # Silently skip errors
            
            self.all_enrichment_results[prog_name] = prog_results
    
    def create_combined_enrichment_plot(self):
        """
        Create combined enrichment visualization
        """
        print("   Creating combined enrichment plot...")
        
        # Determine grid layout
        n_cols = 5
        n_rows = int(np.ceil(self.k / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows*6))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Colors for categories
        colors = {
            'GO_BP': '#1f77b4',
            'GO_MF': '#ff7f0e',
            'GO_CC': '#2ca02c',
            'KEGG': '#d62728'
        }
        
        for idx in range(self.k):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            prog_name = f'Program_{idx+1}'
            
            if prog_name in self.all_enrichment_results:
                prog_data = []
                
                # Collect top terms from each category
                for gs_name, results in self.all_enrichment_results[prog_name].items():
                    if results is not None and not results.empty:
                        top_2 = results.nsmallest(2, 'Adjusted P-value')
                        for _, term_row in top_2.iterrows():
                            prog_data.append({
                                'Term': term_row['Term'][:35],
                                'neg_log_p': -np.log10(term_row['Adjusted P-value'] + 1e-100),
                                'Category': gs_name
                            })
                
                if prog_data:
                    plot_df = pd.DataFrame(prog_data)
                    plot_df = plot_df.sort_values('neg_log_p', ascending=True)
                    
                    y_pos = np.arange(len(plot_df))
                    bar_colors = [colors[cat] for cat in plot_df['Category']]
                    
                    ax.barh(y_pos, plot_df['neg_log_p'].values, color=bar_colors, alpha=0.7)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(plot_df['Term'].values, fontsize=7)
                    ax.axvline(x=-np.log10(0.05), color='black', linestyle='--', alpha=0.3)
                    ax.set_xlabel('-log10(adj. p)', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No enrichment', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title(prog_name, fontsize=10, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(self.k, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7, label=cat) 
                          for cat, color in colors.items()]
        fig.legend(handles=legend_elements, loc='upper center', ncol=4,
                  bbox_to_anchor=(0.5, -0.02), fontsize=10)
        
        plt.suptitle(f'Enrichment Analysis: All Programs (k={self.k})', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'all_programs_enrichment_combined.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_enrichment_summary(self):
        """
        Create enrichment summary
        """
        print("   Creating enrichment summary...")
        
        summary_data = []
        
        for prog_name, prog_results in self.all_enrichment_results.items():
            top_terms = []
            
            for gs_name, results in prog_results.items():
                if results is not None and not results.empty:
                    top_term = results.nsmallest(1, 'Adjusted P-value')
                    if not top_term.empty:
                        term = top_term.iloc[0]['Term'][:40]
                        pval = top_term.iloc[0]['Adjusted P-value']
                        top_terms.append(f"{gs_name}: {term} (p={pval:.2e})")
            
            if top_terms:
                summary_data.append({
                    'Program': prog_name,
                    'Top_Enrichments': '; '.join(top_terms[:2])
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(self.enrich_dir, 'enrichment_summary.csv'), index=False)
            
            print("\n   Top enrichments per program:")
            for _, row in summary_df.head(10).iterrows():
                print(f"     {row['Program']}: {row['Top_Enrichments'][:100]}...")
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        """
        # Step 1: Load and filter data
        self.load_and_filter_data()
        
        # Step 2: Run NMF
        self.run_nmf()
        
        # Step 3: Identify top genes
        self.identify_top_genes()
        
        # Step 4: Create visualizations
        print("\n5. Creating visualizations...")
        self.create_usage_heatmap()
        self.create_top_genes_plot()
        self.create_activity_plots()
        
        # Step 5: Perform enrichment
        self.perform_enrichment()
        
        # Step 6: Create combined plots
        print("\n6. Creating combined enrichment visualizations...")
        self.create_combined_enrichment_plot()
        self.create_enrichment_summary()
        
        # Final summary
        print("\n" + "="*60)
        print(f"ANALYSIS COMPLETE (k={self.k})")
        print("="*60)
        print(f"\nResults saved to: {self.output_dir}/")
        print("\nKey outputs:")
        print("  Main files:")
        print("    - usage_matrix.csv")
        print("    - gene_scores.csv")
        print("    - top_genes_per_program.txt")
        print("\n  Visualizations:")
        print("    - figures/usage_heatmap.png")
        print("    - figures/top_genes_per_program.png")
        print("    - figures/activity_by_*.png")
        print("    - figures/all_programs_enrichment_combined.png")
        print("\n  Enrichment:")
        print("    - enrichment/*_enrichment.csv")
        print("    - enrichment/enrichment_summary.csv")
        
        return self.adata, self.usage_df, self.gene_scores_df

def main():
    """
    Main function to run the analysis
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete NMF analysis with enrichment')
    parser.add_argument('--k', type=int, default=20, help='Number of NMF components (default: 20)')
    parser.add_argument('--input', type=str, default='../sub_adata/strict_endocrine.h5ad',
                       help='Input h5ad file path')
    parser.add_argument('--output', type=str, default='nmf_results',
                       help='Output directory base name')
    
    args = parser.parse_args()
    
    # Run analysis
    pipeline = NMFAnalysisPipeline(
        input_file=args.input,
        k=args.k,
        output_base=args.output
    )
    
    adata, usage, gene_scores = pipeline.run_complete_analysis()
    
    return adata, usage, gene_scores

if __name__ == "__main__":
    # Run with k=20 as default
    main()