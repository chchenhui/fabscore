#!/usr/bin/env python3

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Starting SCENIC heatmap plotting at {timestamp}")

print("Loading data...")
adata = sc.read_h5ad('entero_hg38_scenic_full_results_with_tissue_combined.h5ad')
print(f"Loaded dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")

regulon_cols = [col for col in adata.obs.columns if col.endswith('(+)') or col.endswith('(-)')]
print(f"Found {len(regulon_cols)} regulons")

sc.settings.figdir = './figures/'
import os
os.makedirs('figures', exist_ok=True)

print("\n=== GENERATING HEATMAP PLOTS ===\n")

print("1. Top regulons by activity - average per tissue")
if 'tissue' in adata.obs.columns:
    tissue_regulon_mean = adata.obs.groupby('tissue')[regulon_cols].mean()
    
    top_var_regulons = tissue_regulon_mean.var().nlargest(50).index
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(tissue_regulon_mean[top_var_regulons].T, 
                cmap='viridis', 
                cbar_kws={'label': 'Mean AUCell Score'},
                yticklabels=True,
                xticklabels=True)
    plt.title('Top 50 Most Variable Regulons Across Tissues')
    plt.xlabel('Tissue')
    plt.ylabel('Regulon')
    plt.tight_layout()
    plt.savefig('figures/heatmap_01_tissue_top50_regulons.pdf', dpi=300)
    plt.savefig('figures/heatmap_01_tissue_top50_regulons.png', dpi=150)
    plt.close()
    print("   Saved: heatmap_01_tissue_top50_regulons")

print("2. All regulons clustered heatmap")
plt.figure(figsize=(20, 100))
tissue_regulon_mean_all = adata.obs.groupby('tissue')[regulon_cols].mean()
g = sns.clustermap(tissue_regulon_mean_all.T, 
                    cmap='RdBu_r', 
                    center=0,
                    cbar_kws={'label': 'Mean AUCell Score'},
                    figsize=(12, 30),
                    yticklabels=True,
                    xticklabels=True)
g.ax_heatmap.set_xlabel('Tissue')
g.ax_heatmap.set_ylabel('Regulon')
plt.suptitle('All Regulons Clustered by Activity Pattern', y=1.001)
plt.savefig('figures/heatmap_02_all_regulons_clustered.pdf', dpi=300)
plt.savefig('figures/heatmap_02_all_regulons_clustered.png', dpi=150)
plt.close()
print("   Saved: heatmap_02_all_regulons_clustered")

print("3. Cell type specific regulons")
if 'cell_type' in adata.obs.columns:
    celltype_regulon_mean = adata.obs.groupby('cell_type')[regulon_cols].mean()
    
    top_ct_regulons = celltype_regulon_mean.var().nlargest(50).index
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(celltype_regulon_mean[top_ct_regulons].T, 
                cmap='magma', 
                cbar_kws={'label': 'Mean AUCell Score'},
                yticklabels=True,
                xticklabels=True)
    plt.title('Top 50 Most Variable Regulons Across Cell Types')
    plt.xlabel('Cell Type')
    plt.ylabel('Regulon')
    plt.tight_layout()
    plt.savefig('figures/heatmap_03_celltype_top50_regulons.pdf', dpi=300)
    plt.savefig('figures/heatmap_03_celltype_top50_regulons.png', dpi=150)
    plt.close()
    print("   Saved: heatmap_03_celltype_top50_regulons")

print("4. Sample-specific regulon activity")
if 'sample' in adata.obs.columns:
    sample_regulon_mean = adata.obs.groupby('sample')[regulon_cols].mean()
    
    top_sample_regulons = sample_regulon_mean.var().nlargest(40).index
    
    plt.figure(figsize=(20, 12))
    sns.heatmap(sample_regulon_mean[top_sample_regulons].T, 
                cmap='coolwarm', 
                center=0,
                cbar_kws={'label': 'Mean AUCell Score'},
                yticklabels=True,
                xticklabels=True)
    plt.title('Top 40 Most Variable Regulons Across Samples')
    plt.xlabel('Sample')
    plt.ylabel('Regulon')
    plt.tight_layout()
    plt.savefig('figures/heatmap_04_sample_top40_regulons.pdf', dpi=300)
    plt.savefig('figures/heatmap_04_sample_top40_regulons.png', dpi=150)
    plt.close()
    print("   Saved: heatmap_04_sample_top40_regulons")

print("5. Highly active regulons (top by mean activity)")
regulon_means = adata.obs[regulon_cols].mean().sort_values(ascending=False)
top_active = regulon_means.head(30).index

subset_data = adata.obs.sample(n=min(5000, adata.shape[0]), random_state=42)

plt.figure(figsize=(14, 10))
sns.heatmap(subset_data[top_active].T, 
            cmap='YlOrRd', 
            cbar_kws={'label': 'AUCell Score'},
            yticklabels=True,
            xticklabels=False)
plt.title('Top 30 Most Active Regulons (5000 random cells)')
plt.xlabel('Cells')
plt.ylabel('Regulon')
plt.tight_layout()
plt.savefig('figures/heatmap_05_top30_active_regulons.pdf', dpi=300)
plt.savefig('figures/heatmap_05_top30_active_regulons.png', dpi=150)
plt.close()
print("   Saved: heatmap_05_top30_active_regulons")

print("6. Correlation between regulons")
top_var_regs = adata.obs[regulon_cols].var().nlargest(30).index
regulon_corr = adata.obs[top_var_regs].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(regulon_corr, 
            cmap='RdBu_r', 
            center=0,
            vmin=-1, vmax=1,
            square=True,
            cbar_kws={'label': 'Correlation'},
            xticklabels=True,
            yticklabels=True)
plt.title('Regulon-Regulon Correlation Matrix (Top 30 Variable)')
plt.tight_layout()
plt.savefig('figures/heatmap_06_regulon_correlation.pdf', dpi=300)
plt.savefig('figures/heatmap_06_regulon_correlation.png', dpi=150)
plt.close()
print("   Saved: heatmap_06_regulon_correlation")

print("7. Disease state comparison")
if 'disease' in adata.obs.columns:
    disease_regulon_mean = adata.obs.groupby('disease')[regulon_cols].mean()
    
    top_disease_regulons = disease_regulon_mean.var().nlargest(40).index
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(disease_regulon_mean[top_disease_regulons].T, 
                cmap='PiYG', 
                center=0,
                cbar_kws={'label': 'Mean AUCell Score'},
                yticklabels=True,
                xticklabels=True)
    plt.title('Top 40 Most Variable Regulons Across Disease States')
    plt.xlabel('Disease State')
    plt.ylabel('Regulon')
    plt.tight_layout()
    plt.savefig('figures/heatmap_07_disease_top40_regulons.pdf', dpi=300)
    plt.savefig('figures/heatmap_07_disease_top40_regulons.png', dpi=150)
    plt.close()
    print("   Saved: heatmap_07_disease_top40_regulons")

print("8. Z-scored regulon activity heatmap")
regulon_data = adata.obs[regulon_cols]
regulon_zscore = (regulon_data - regulon_data.mean()) / regulon_data.std()

top_var_z = regulon_zscore.var().nlargest(40).index
subset_cells = np.random.choice(adata.obs.index, size=min(3000, adata.shape[0]), replace=False)

plt.figure(figsize=(16, 12))
sns.heatmap(regulon_zscore.loc[subset_cells, top_var_z].T, 
            cmap='seismic', 
            center=0,
            vmin=-3, vmax=3,
            cbar_kws={'label': 'Z-score'},
            yticklabels=True,
            xticklabels=False)
plt.title('Z-scored Regulon Activity (Top 40 Variable, 3000 cells)')
plt.xlabel('Cells')
plt.ylabel('Regulon')
plt.tight_layout()
plt.savefig('figures/heatmap_08_zscore_regulons.pdf', dpi=300)
plt.savefig('figures/heatmap_08_zscore_regulons.png', dpi=150)
plt.close()
print("   Saved: heatmap_08_zscore_regulons")

print("9. Tissue-specific enriched regulons")
if 'tissue' in adata.obs.columns:
    tissue_mean = adata.obs.groupby('tissue')[regulon_cols].mean()
    global_mean = adata.obs[regulon_cols].mean()
    
    tissue_enrichment = tissue_mean.subtract(global_mean, axis=1)
    
    top_enriched = tissue_enrichment.abs().max().nlargest(50).index
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(tissue_enrichment[top_enriched].T, 
                cmap='PRGn', 
                center=0,
                cbar_kws={'label': 'Enrichment Score'},
                yticklabels=True,
                xticklabels=True)
    plt.title('Top 50 Tissue-Enriched Regulons')
    plt.xlabel('Tissue')
    plt.ylabel('Regulon')
    plt.tight_layout()
    plt.savefig('figures/heatmap_09_tissue_enriched_regulons.pdf', dpi=300)
    plt.savefig('figures/heatmap_09_tissue_enriched_regulons.png', dpi=150)
    plt.close()
    print("   Saved: heatmap_09_tissue_enriched_regulons")

print("10. Hierarchical clustering of tissues by regulon profile")
if 'tissue' in adata.obs.columns:
    tissue_profiles = adata.obs.groupby('tissue')[regulon_cols].mean()
    
    g = sns.clustermap(tissue_profiles.T.iloc[:100, :],  
                        cmap='viridis',
                        col_cluster=True,
                        row_cluster=True,
                        figsize=(10, 20),
                        cbar_kws={'label': 'Mean AUCell Score'},
                        yticklabels=True,
                        xticklabels=True)
    g.ax_heatmap.set_xlabel('Tissue')
    g.ax_heatmap.set_ylabel('Regulon (First 100)')
    plt.suptitle('Hierarchical Clustering of Tissues', y=1.001)
    plt.savefig('figures/heatmap_10_tissue_clustering.pdf', dpi=300)
    plt.savefig('figures/heatmap_10_tissue_clustering.png', dpi=150)
    plt.close()
    print("   Saved: heatmap_10_tissue_clustering")

print("\n=== GENERATING SUMMARY STATISTICS ===\n")

summary_stats = pd.DataFrame({
    'mean_activity': adata.obs[regulon_cols].mean(),
    'std_activity': adata.obs[regulon_cols].std(),
    'max_activity': adata.obs[regulon_cols].max(),
    'min_activity': adata.obs[regulon_cols].min(),
    'variance': adata.obs[regulon_cols].var(),
    'active_cells': (adata.obs[regulon_cols] > 0.01).sum(),
    'percent_active': (adata.obs[regulon_cols] > 0.01).sum() / adata.shape[0] * 100
})

summary_stats = summary_stats.sort_values('variance', ascending=False)
summary_stats.to_csv('figures/regulon_summary_statistics.csv')
print("Saved regulon summary statistics to regulon_summary_statistics.csv")

print("\n=== HEATMAP GENERATION COMPLETE ===")
print(f"\nGenerated 10 different heatmap visualizations in the 'figures' directory")
print("File formats: Both PDF (high quality) and PNG (preview) versions")

print("\n=== LIST OF GENERATED HEATMAPS ===")
print("1. heatmap_01_tissue_top50_regulons - Top 50 variable regulons across tissues")
print("2. heatmap_02_all_regulons_clustered - All regulons with hierarchical clustering")
print("3. heatmap_03_celltype_top50_regulons - Top 50 variable regulons across cell types")
print("4. heatmap_04_sample_top40_regulons - Top 40 variable regulons across samples")
print("5. heatmap_05_top30_active_regulons - Most active regulons across cells")
print("6. heatmap_06_regulon_correlation - Correlation matrix between regulons")
print("7. heatmap_07_disease_top40_regulons - Disease state comparison")
print("8. heatmap_08_zscore_regulons - Z-scored regulon activities")
print("9. heatmap_09_tissue_enriched_regulons - Tissue-specific enrichment")
print("10. heatmap_10_tissue_clustering - Hierarchical clustering of tissues")

print("\nAdditional file:")
print("- regulon_summary_statistics.csv - Statistical summary of all regulons")