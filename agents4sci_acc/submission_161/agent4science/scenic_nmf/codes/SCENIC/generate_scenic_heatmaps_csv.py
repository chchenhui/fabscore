#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print(f"[{datetime.now()}] Loading h5ad file...")
adata = sc.read_h5ad('entero_hg38_scenic_full_results_with_tissue_combined.h5ad')

print(f"[{datetime.now()}] Data shape: {adata.shape}")

# Extract regulon columns
regulon_cols = [col for col in adata.obs.columns if '(' in col and ')' in col]
print(f"[{datetime.now()}] Found {len(regulon_cols)} regulons")

# Create DataFrame with regulon activities and tissue info
print(f"[{datetime.now()}] Creating regulon activity matrix...")
regulon_df = adata.obs[regulon_cols].copy()
regulon_df['tissue_combined'] = adata.obs['tissue_combined']

# Save full regulon activity matrix to CSV
print(f"[{datetime.now()}] Saving full regulon activity matrix to CSV...")
regulon_df.to_csv('entero_scenic_regulon_activities_full.csv')
print(f"[{datetime.now()}] Saved full matrix: entero_scenic_regulon_activities_full.csv")

# Calculate mean regulon activity per tissue
print(f"[{datetime.now()}] Calculating mean regulon activity per tissue...")
mean_activity = regulon_df.groupby('tissue_combined')[regulon_cols].mean()

# Save mean activity matrix to CSV
mean_activity.to_csv('entero_scenic_mean_regulon_activity_by_tissue.csv')
print(f"[{datetime.now()}] Saved mean activity: entero_scenic_mean_regulon_activity_by_tissue.csv")

# Calculate standard deviation
std_activity = regulon_df.groupby('tissue_combined')[regulon_cols].std()
std_activity.to_csv('entero_scenic_std_regulon_activity_by_tissue.csv')
print(f"[{datetime.now()}] Saved std activity: entero_scenic_std_regulon_activity_by_tissue.csv")

# Find top variable regulons across tissues
print(f"[{datetime.now()}] Finding most variable regulons...")
regulon_variance = mean_activity.var(axis=0)
top_variable_regulons = regulon_variance.nlargest(50).index.tolist()

# Save top variable regulons list
pd.DataFrame({'regulon': top_variable_regulons, 'variance': regulon_variance[top_variable_regulons].values}).to_csv('entero_scenic_top50_variable_regulons.csv', index=False)
print(f"[{datetime.now()}] Saved top variable regulons: entero_scenic_top50_variable_regulons.csv")

# Plot 1: Heatmap of all regulons (scaled)
print(f"[{datetime.now()}] Creating heatmap of all regulons...")
plt.figure(figsize=(20, 10))
scaled_activity = (mean_activity - mean_activity.mean()) / mean_activity.std()
sns.heatmap(scaled_activity.T, cmap='RdBu_r', center=0, cbar_kws={'label': 'Z-score'}, 
            xticklabels=True, yticklabels=False)
plt.title('SCENIC Regulon Activity Heatmap (All Regulons)', fontsize=16)
plt.xlabel('Tissue Combined', fontsize=12)
plt.ylabel('Regulons', fontsize=12)
plt.tight_layout()
plt.savefig('entero_scenic_heatmap_all_regulons.png', dpi=300, bbox_inches='tight')
plt.savefig('entero_scenic_heatmap_all_regulons.pdf', bbox_inches='tight')
plt.close()
print(f"[{datetime.now()}] Saved: entero_scenic_heatmap_all_regulons.png/pdf")

# Plot 2: Heatmap of top 50 variable regulons
print(f"[{datetime.now()}] Creating heatmap of top 50 variable regulons...")
plt.figure(figsize=(16, 12))
scaled_top = scaled_activity[top_variable_regulons]
sns.heatmap(scaled_top.T, cmap='RdBu_r', center=0, cbar_kws={'label': 'Z-score'},
            xticklabels=True, yticklabels=True)
plt.title('SCENIC Top 50 Variable Regulons Activity Heatmap', fontsize=16)
plt.xlabel('Tissue Combined', fontsize=12)
plt.ylabel('Regulons', fontsize=12)
plt.tight_layout()
plt.savefig('entero_scenic_heatmap_top50_variable.png', dpi=300, bbox_inches='tight')
plt.savefig('entero_scenic_heatmap_top50_variable.pdf', bbox_inches='tight')
plt.close()
print(f"[{datetime.now()}] Saved: entero_scenic_heatmap_top50_variable.png/pdf")

# Plot 3: Clustered heatmap
print(f"[{datetime.now()}] Creating clustered heatmap...")
plt.figure(figsize=(16, 12))
g = sns.clustermap(scaled_top.T, cmap='RdBu_r', center=0, 
                   figsize=(16, 12), cbar_kws={'label': 'Z-score'},
                   xticklabels=True, yticklabels=True)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
g.fig.suptitle('SCENIC Top 50 Variable Regulons (Clustered)', fontsize=16, y=1.02)
plt.savefig('entero_scenic_heatmap_top50_clustered.png', dpi=300, bbox_inches='tight')
plt.savefig('entero_scenic_heatmap_top50_clustered.pdf', bbox_inches='tight')
plt.close()
print(f"[{datetime.now()}] Saved: entero_scenic_heatmap_top50_clustered.png/pdf")

# Find tissue-specific regulons
print(f"[{datetime.now()}] Finding tissue-specific regulons...")
tissue_specific_regulons = {}
for tissue in mean_activity.index:
    # Find regulons with high activity in this tissue compared to others
    tissue_mean = mean_activity.loc[tissue]
    other_mean = mean_activity.drop(tissue).mean()
    fold_change = tissue_mean / (other_mean + 0.001)  # Add small value to avoid division by zero
    top_specific = fold_change.nlargest(10).index.tolist()
    tissue_specific_regulons[tissue] = top_specific

# Save tissue-specific regulons
tissue_specific_df = pd.DataFrame.from_dict(tissue_specific_regulons, orient='index').T
tissue_specific_df.to_csv('entero_scenic_tissue_specific_top10_regulons.csv', index=False)
print(f"[{datetime.now()}] Saved tissue-specific regulons: entero_scenic_tissue_specific_top10_regulons.csv")

# Plot 4: Tissue-specific regulons heatmap
print(f"[{datetime.now()}] Creating tissue-specific regulons heatmap...")
all_specific_regulons = list(set([reg for regs in tissue_specific_regulons.values() for reg in regs]))
plt.figure(figsize=(16, len(all_specific_regulons)*0.3))
scaled_specific = scaled_activity[all_specific_regulons]
sns.heatmap(scaled_specific.T, cmap='RdBu_r', center=0, cbar_kws={'label': 'Z-score'},
            xticklabels=True, yticklabels=True)
plt.title('SCENIC Tissue-Specific Regulons Activity Heatmap', fontsize=16)
plt.xlabel('Tissue Combined', fontsize=12)
plt.ylabel('Regulons', fontsize=12)
plt.tight_layout()
plt.savefig('entero_scenic_heatmap_tissue_specific.png', dpi=300, bbox_inches='tight')
plt.savefig('entero_scenic_heatmap_tissue_specific.pdf', bbox_inches='tight')
plt.close()
print(f"[{datetime.now()}] Saved: entero_scenic_heatmap_tissue_specific.png/pdf")

# Calculate and save correlation matrix between tissues
print(f"[{datetime.now()}] Calculating tissue correlation matrix...")
tissue_corr = mean_activity.T.corr()
tissue_corr.to_csv('entero_scenic_tissue_correlation_matrix.csv')
print(f"[{datetime.now()}] Saved: entero_scenic_tissue_correlation_matrix.csv")

# Plot 5: Tissue correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(tissue_corr, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, annot=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
plt.title('Tissue Correlation Based on Regulon Activity', fontsize=16)
plt.tight_layout()
plt.savefig('entero_scenic_tissue_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('entero_scenic_tissue_correlation_heatmap.pdf', bbox_inches='tight')
plt.close()
print(f"[{datetime.now()}] Saved: entero_scenic_tissue_correlation_heatmap.png/pdf")

# Summary statistics
print(f"\n[{datetime.now()}] Summary Statistics:")
print(f"Total cells: {len(regulon_df)}")
print(f"Total regulons: {len(regulon_cols)}")
print(f"Tissues: {len(mean_activity)}")
print(f"\nTissue cell counts:")
print(adata.obs['tissue_combined'].value_counts())

# Create summary report
summary = {
    'total_cells': len(regulon_df),
    'total_regulons': len(regulon_cols),
    'total_tissues': len(mean_activity),
    'tissue_counts': adata.obs['tissue_combined'].value_counts().to_dict(),
    'top_variable_regulons': top_variable_regulons[:10],
    'files_generated': [
        'entero_scenic_regulon_activities_full.csv',
        'entero_scenic_mean_regulon_activity_by_tissue.csv',
        'entero_scenic_std_regulon_activity_by_tissue.csv',
        'entero_scenic_top50_variable_regulons.csv',
        'entero_scenic_tissue_specific_top10_regulons.csv',
        'entero_scenic_tissue_correlation_matrix.csv',
        'entero_scenic_heatmap_all_regulons.png/pdf',
        'entero_scenic_heatmap_top50_variable.png/pdf',
        'entero_scenic_heatmap_top50_clustered.png/pdf',
        'entero_scenic_heatmap_tissue_specific.png/pdf',
        'entero_scenic_tissue_correlation_heatmap.png/pdf'
    ]
}

# Save summary
import json
with open('entero_scenic_analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"[{datetime.now()}] Saved analysis summary: entero_scenic_analysis_summary.json")

print(f"\n[{datetime.now()}] All analyses completed successfully!")
print("\nGenerated files:")
for f in summary['files_generated']:
    print(f"  - {f}")