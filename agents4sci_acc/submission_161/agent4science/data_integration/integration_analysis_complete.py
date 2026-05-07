#!/usr/bin/env python3
"""
Complete integration analysis: visualizations and SCIB benchmarking for neuroendocrine dataset
Combines improved_visualizations_no_scib.py and benchmark_scib_final.py
"""

import os
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure for SCIB metrics
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'

# Configure scanpy and plotting settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=150, facecolor='white', figsize=(5, 5))
sc.settings.figdir = '/scratch/rli/project/agent/data_integration/results/'

# Create output directory
os.makedirs(sc.settings.figdir, exist_ok=True)

print("="*80)
print("COMPLETE INTEGRATION ANALYSIS: VISUALIZATIONS + BENCHMARKING")
print("="*80)

# ==============================================================================
# PART 1: LOAD DATA
# ==============================================================================
print("\n1. Loading integrated dataset...")
adata = sc.read_h5ad('/scratch/rli/project/agent/data_integration/results/scvi_unique_integration/neuroendocrine_scvi_integrated_unique.h5ad')
print(f"Loaded dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# Print available keys for reference
print("\nAvailable data layers:")
print(f"  - .layers keys: {list(adata.layers.keys())}")
print(f"  - .obsm keys: {list(adata.obsm.keys())}")
print(f"  - .obs columns: {list(adata.obs.columns)[:10]}...")

# ==============================================================================
# PART 2: QC VISUALIZATIONS
# ==============================================================================
print("\n" + "="*80)
print("2. QC METRICS VISUALIZATION")
print("="*80)

# Create QC violin plots
print("\nCreating QC violin plots...")
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    save='_qc_metrics_violin.pdf'
)

# Create QC scatter plots
print("Creating QC scatter plots...")
sc.pl.scatter(
    adata,
    x="total_counts",
    y="n_genes_by_counts",
    color="pct_counts_mt",
    save='_qc_metrics_scatter.pdf'
)

# ==============================================================================
# PART 3: HVG AND BATCH DENSITY
# ==============================================================================
print("\n" + "="*80)
print("3. HIGHLY VARIABLE GENES AND BATCH EFFECTS")
print("="*80)

# Set scvi layer for HVG analysis
if 'scvi' in adata.layers:
    print("\nUsing scVI normalized data for HVG analysis...")
    adata.X = adata.layers['scvi'].copy()
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='batch')
    sc.pl.highly_variable_genes(adata, save='_hvg.pdf')
    
    # Density plot for batch effects
    if 'X_umap' in adata.obsm:
        print("\nCreating batch density plot...")
        sc.tl.embedding_density(adata, basis='umap', groupby='batch')
        sc.pl.embedding_density(adata, basis='umap', groupby='batch', save='_batch_density.pdf')

# ==============================================================================
# PART 4: INTEGRATION COMPARISON PLOT
# ==============================================================================
print("\n" + "="*80)
print("4. INTEGRATION COMPARISON VISUALIZATION")
print("="*80)

def create_improved_comparison_plot(adata):
    """Create integration comparison with single-column legends"""
    
    # Use gridspec for better control of spacing
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(24, 20))
    
    # Define what to plot
    methods = ['Unintegrated', 'ComBat', 'scVI']
    categories = ['Batch Effects', 'Cell Types', 'Tissues', 'Disease States']
    
    # Corresponding data and color keys
    embeddings = {
        'Unintegrated': 'X_umap_unintegrated',
        'ComBat': 'X_umap_combat', 
        'scVI': 'X_umap'
    }
    
    color_keys = {
        'Batch Effects': 'batch',
        'Cell Types': 'endocrine_type_simple',
        'Tissues': 'tissue',
        'Disease States': 'disease'
    }
    
    # Create grid with more horizontal spacing
    gs = GridSpec(4, 3, figure=fig, wspace=0.4, hspace=0.35)
    
    for i, category in enumerate(categories):
        for j, method in enumerate(methods):
            ax = fig.add_subplot(gs[i, j])
            
            # Get embedding
            if embeddings[method] in adata.obsm:
                X_emb = adata.obsm[embeddings[method]]
            else:
                print(f"Warning: {embeddings[method]} not found")
                continue
            
            # Get color values
            color_key = color_keys[category]
            if color_key not in adata.obs.columns:
                print(f"Warning: {color_key} not found")
                continue
                
            # Plot - only show legend in the last column (scVI)
            show_legend = (j == 2)  # Only show legend for scVI column
            
            if show_legend:
                # Show legend on right margin for all categories
                legend_loc = 'right margin'
            else:
                # No legend for first two columns
                legend_loc = None
            
            # Adjust font size based on category
            if category == 'Batch Effects' or category == 'Tissues':
                legend_fontsize = 'xx-small'  # Smaller for categories with many items
            else:
                legend_fontsize = 'small'
            
            sc.pl.embedding(
                adata,
                basis=embeddings[method].replace('X_', ''),
                color=color_key,
                ax=ax,
                show=False,
                legend_loc=legend_loc,
                legend_fontsize=legend_fontsize,
                s=1,
                title=f'{method}: {category}',
                frameon=False
            )
    
    plt.suptitle('Three-way Integration Comparison (Unique Datasets)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{sc.settings.figdir}/integration_comparison_improved.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(f'{sc.settings.figdir}/integration_comparison_improved.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Saved improved integration comparison plot")

# Create the improved comparison
create_improved_comparison_plot(adata)

# ==============================================================================
# PART 5: ENDOCRINE CELL MARKER DOTPLOT
# ==============================================================================
print("\n" + "="*80)
print("5. ENDOCRINE CELL MARKER DOTPLOT")
print("="*80)

# Define endocrine markers
endocrine_markers = {
    'P/D1 enteroendocrine cell': ['CHGA', 'TPH1', 'DDC', 'NTS'],
    'enteroendocrine cell': ['CHGA', 'GCG', 'SST', 'PCSK1N'],
    'enteroendocrine cell of colon': ['CHGA', 'PYY', 'GLP1R', 'INSL5'],
    'neuroendocrine cell': ['SYP', 'NCAM1', 'INSM1'],
    'type A enteroendocrine cell': ['GCG', 'GIP', 'CCK'],
    'type D enteroendocrine cell': ['SST', 'GHRL'],
    'type L enteroendocrine cell': ['GCG', 'PYY', 'NTS']
}

# Flatten marker list and remove duplicates
all_markers = list(set([m for markers in endocrine_markers.values() for m in markers]))

# Check which markers are present
present_markers = [m for m in all_markers if m in adata.var_names]
print(f"Found {len(present_markers)}/{len(all_markers)} markers in the dataset")

if present_markers:
    print("\nCreating marker dotplot...")
    sc.pl.dotplot(
        adata, 
        present_markers, 
        groupby='endocrine_type_simple',
        dendrogram=True,
        save='_endocrine_markers.pdf',
        figsize=(12, 6)
    )

# ==============================================================================
# PART 6: CELL TYPE DISTRIBUTION
# ==============================================================================
print("\n" + "="*80)
print("6. CELL TYPE AND BATCH DISTRIBUTION ANALYSIS")
print("="*80)

def create_distribution_plots(adata):
    """Create comprehensive distribution plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cell type distribution
    cell_counts = adata.obs['endocrine_type_simple'].value_counts()
    ax = axes[0, 0]
    cell_counts.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Number of cells')
    ax.set_title('Cell Type Distribution')
    ax.set_ylabel('')
    
    # 2. Batch distribution
    batch_counts = adata.obs['batch'].value_counts().head(20)  # Top 20 batches
    ax = axes[0, 1]
    batch_counts.plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Number of cells')
    ax.set_title('Top 20 Batch Distribution')
    ax.set_ylabel('')
    
    # 3. Tissue distribution
    if 'tissue' in adata.obs.columns:
        tissue_counts = adata.obs['tissue'].value_counts().head(15)
        ax = axes[1, 0]
        tissue_counts.plot(kind='barh', ax=ax, color='mediumseagreen')
        ax.set_xlabel('Number of cells')
        ax.set_title('Top 15 Tissue Distribution')
        ax.set_ylabel('')
    
    # 4. Disease state distribution
    if 'disease' in adata.obs.columns:
        disease_counts = adata.obs['disease'].value_counts()
        ax = axes[1, 1]
        disease_counts.plot(kind='barh', ax=ax, color='mediumpurple')
        ax.set_xlabel('Number of cells')
        ax.set_title('Disease State Distribution')
        ax.set_ylabel('')
    
    plt.suptitle('Dataset Composition Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{sc.settings.figdir}/dataset_distributions.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(f'{sc.settings.figdir}/dataset_distributions.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Saved distribution plots")

create_distribution_plots(adata)

# ==============================================================================
# PART 7: SCIB METRICS BENCHMARKING
# ==============================================================================
print("\n" + "="*80)
print("7. SCIB METRICS BENCHMARKING")
print("="*80)

# Import SCIB metrics
try:
    from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
    
    print("\nPreparing embeddings for benchmarking...")
    
    # Ensure batch and label are categorical
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obs["endocrine_type_simple"] = adata.obs["endocrine_type_simple"].astype("category")
    
    # Copy embeddings with clean method names
    embedding_keys = []
    
    # 1. Unintegrated (PCA)
    if "X_pca_unintegrated" in adata.obsm.keys():
        adata.obsm["Unintegrated"] = adata.obsm["X_pca_unintegrated"].copy()
        embedding_keys.append("Unintegrated")
        print(f"   ✓ Unintegrated: PCA {adata.obsm['Unintegrated'].shape}")
    elif "X_pca" in adata.obsm.keys():
        adata.obsm["Unintegrated"] = adata.obsm["X_pca"].copy()
        embedding_keys.append("Unintegrated")
        print(f"   ✓ Unintegrated: PCA {adata.obsm['Unintegrated'].shape}")
    
    # 2. ComBat (PCA after ComBat)
    if "X_pca_combat" in adata.obsm.keys():
        adata.obsm["ComBat"] = adata.obsm["X_pca_combat"].copy()
        embedding_keys.append("ComBat")
        print(f"   ✓ ComBat: PCA {adata.obsm['ComBat'].shape}")
    
    # 3. scVI (latent representation)
    if "X_scVI" in adata.obsm.keys():
        adata.obsm["scVI"] = adata.obsm["X_scVI"].copy()
        embedding_keys.append("scVI")
        print(f"   ✓ scVI: Latent {adata.obsm['scVI'].shape}")
    
    print(f"\nMethods to benchmark: {embedding_keys}")
    
    if len(embedding_keys) > 0:
        # Setup Benchmarker
        print("\n3. Setting up Benchmarker...")
        bm = Benchmarker(
            adata,
            batch_key="batch",
            label_key="endocrine_type_simple",
            embedding_obsm_keys=embedding_keys,
            n_jobs=32
        )
        
        # Reduce metrics for faster computation
        print("\n4. Running benchmark with selected metrics...")
        bm.benchmark(min_max_scale=False)
        
        # Get results
        print("\n5. Extracting results...")
        results_df = bm.get_results(min_max_scale=False)
        
        # Save raw results
        results_df.to_csv(f'{sc.settings.figdir}/scib_benchmark_results.csv')
        print(f"   ✓ Saved raw results to scib_benchmark_results.csv")
        
        # Display results
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        print("\nFull results table:")
        print(results_df.to_string())
        
        # Create visualization of results
        print("\n6. Creating benchmark visualization...")
        
        # Prepare data for plotting
        plot_df = results_df.T
        plot_df.index.name = 'Metric'
        plot_df = plot_df.reset_index()
        
        # Melt for seaborn
        plot_df_melted = plot_df.melt(id_vars='Metric', var_name='Method', value_name='Score')
        
        # Categorize metrics
        bio_metrics = ['NMI cluster/label', 'ARI cluster/label', 'ASW label', 
                       'ASW label/batch', 'Cell type ASW', 'Isolated label F1', 
                       'Isolated label ASW']
        batch_metrics = ['ASW batch', 'PCR batch', 'Batch ASW', 'Graph connectivity', 
                         'kBET', 'iLISI', 'cLISI']
        
        plot_df_melted['Category'] = plot_df_melted['Metric'].apply(
            lambda x: 'Bio Conservation' if any(m in x for m in bio_metrics) else 'Batch Correction'
        )
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for idx, (cat, ax) in enumerate(zip(['Bio Conservation', 'Batch Correction'], axes)):
            subset = plot_df_melted[plot_df_melted['Category'] == cat]
            if not subset.empty:
                pivot = subset.pivot(index='Metric', columns='Method', values='Score')
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                           center=0.5, vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
                ax.set_title(cat)
                ax.set_xlabel('')
                ax.set_ylabel('')
        
        plt.suptitle('SCIB Metrics Benchmark Results', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{sc.settings.figdir}/scib_benchmark_heatmap.pdf', bbox_inches='tight', dpi=150)
        plt.savefig(f'{sc.settings.figdir}/scib_benchmark_heatmap.png', bbox_inches='tight', dpi=150)
        plt.close()
        print("   ✓ Saved benchmark visualization")
        
        # Create summary scores plot
        print("\n7. Creating summary scores...")
        
        # Calculate aggregate scores
        bio_cols = [col for col in results_df.columns if any(m in col for m in bio_metrics)]
        batch_cols = [col for col in results_df.columns if any(m in col for m in batch_metrics)]
        
        if bio_cols and batch_cols:
            summary_scores = pd.DataFrame({
                'Bio Conservation': results_df[bio_cols].mean(axis=1),
                'Batch Correction': results_df[batch_cols].mean(axis=1),
                'Overall': results_df.mean(axis=1)
            })
            
            # Plot summary
            fig, ax = plt.subplots(figsize=(10, 6))
            summary_scores.plot(kind='bar', ax=ax)
            ax.set_ylabel('Average Score')
            ax.set_xlabel('Method')
            ax.set_title('Integration Method Performance Summary')
            ax.legend(title='Metric Category')
            ax.set_ylim([0, 1])
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{sc.settings.figdir}/scib_summary_scores.pdf', bbox_inches='tight', dpi=150)
            plt.savefig(f'{sc.settings.figdir}/scib_summary_scores.png', bbox_inches='tight', dpi=150)
            plt.close()
            print("   ✓ Saved summary scores")
            
            # Print summary
            print("\n" + "="*80)
            print("SUMMARY SCORES")
            print("="*80)
            print(summary_scores.to_string())
    
except ImportError:
    print("\n⚠ SCIB metrics not available. Skipping benchmarking.")
    print("  To enable benchmarking, install: pip install scib-metrics")

except Exception as e:
    print(f"\n⚠ Error during benchmarking: {e}")
    print("  Visualizations completed successfully, but benchmarking failed.")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {sc.settings.figdir}")
print("\nGenerated files:")
print("  - QC metrics: violin and scatter plots")
print("  - Highly variable genes plot")
print("  - Batch density plot")
print("  - Integration comparison (4x3 grid)")
print("  - Endocrine cell markers dotplot")
print("  - Dataset distribution plots")
if 'results_df' in locals():
    print("  - SCIB benchmark results and visualizations")
print("\n✓ Analysis complete!")