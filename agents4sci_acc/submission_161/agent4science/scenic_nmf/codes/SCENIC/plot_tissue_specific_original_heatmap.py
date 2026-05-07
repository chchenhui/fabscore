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

# Calculate mean regulon activity per tissue
print(f"[{datetime.now()}] Calculating mean regulon activity per tissue...")
mean_activity = regulon_df.groupby('tissue_combined')[regulon_cols].mean()

# Calculate tissue-specificity scores for each regulon
print(f"[{datetime.now()}] Calculating tissue-specificity scores...")
tissue_specificity_scores = pd.DataFrame(index=mean_activity.index, columns=regulon_cols, dtype=float)

for regulon in regulon_cols:
    for tissue in mean_activity.index:
        # Calculate specificity score: activity in this tissue vs. mean of others
        tissue_activity = mean_activity.loc[tissue, regulon]
        other_tissues_mean = mean_activity.drop(tissue)[regulon].mean()
        
        # Tissue-specificity score (difference between tissue and others)
        specificity_score = float(tissue_activity - other_tissues_mean)
        tissue_specificity_scores.loc[tissue, regulon] = specificity_score

# Find regulons with at least one tissue-specific score > 0.1
print(f"[{datetime.now()}] Finding regulons with tissue-specific score > 0.1...")
max_specificity_per_regulon = tissue_specificity_scores.max(axis=0)
selected_regulons = max_specificity_per_regulon[max_specificity_per_regulon > 0.1].index.tolist()

print(f"[{datetime.now()}] Found {len(selected_regulons)} regulons with tissue-specific score > 0.1")

# Save the tissue-specificity scores for selected regulons
tissue_specificity_selected = tissue_specificity_scores[selected_regulons]
tissue_specificity_selected.to_csv('entero_tissue_specificity_scores_above_0.1.csv')
print(f"[{datetime.now()}] Saved tissue-specificity scores: entero_tissue_specificity_scores_above_0.1.csv")

# Get original mean activities for selected regulons
original_values_selected = mean_activity[selected_regulons]

# Save the original values for selected regulons
original_values_selected.to_csv('entero_original_activity_tissue_specific_above_0.1.csv')
print(f"[{datetime.now()}] Saved original activity values: entero_original_activity_tissue_specific_above_0.1.csv")

# Create a summary of selected regulons
summary_df = pd.DataFrame({
    'regulon': selected_regulons,
    'max_specificity_score': max_specificity_per_regulon[selected_regulons].values,
    'most_specific_tissue': [tissue_specificity_scores[reg].idxmax() for reg in selected_regulons],
    'mean_activity_in_specific_tissue': [mean_activity.loc[tissue_specificity_scores[reg].idxmax(), reg] for reg in selected_regulons]
})
summary_df = summary_df.sort_values('max_specificity_score', ascending=False)
summary_df.to_csv('entero_tissue_specific_regulons_summary.csv', index=False)
print(f"[{datetime.now()}] Saved summary: entero_tissue_specific_regulons_summary.csv")

# Plot 1: Heatmap with original values (not z-scored)
print(f"[{datetime.now()}] Creating heatmap with original values...")
plt.figure(figsize=(16, min(20, len(selected_regulons)*0.25)))
sns.heatmap(original_values_selected.T, cmap='YlOrRd', cbar_kws={'label': 'Mean Regulon Activity'},
            xticklabels=True, yticklabels=True, vmin=0)
plt.title(f'SCENIC Regulon Activity - Original Values\n(Regulons with tissue-specificity score > 0.1, n={len(selected_regulons)})', fontsize=14)
plt.xlabel('Tissue Combined', fontsize=12)
plt.ylabel('Regulons', fontsize=12)
plt.tight_layout()
plt.savefig('entero_scenic_heatmap_tissue_specific_original_values.png', dpi=300, bbox_inches='tight')
plt.savefig('entero_scenic_heatmap_tissue_specific_original_values.pdf', bbox_inches='tight')
plt.close()
print(f"[{datetime.now()}] Saved: entero_scenic_heatmap_tissue_specific_original_values.png/pdf")

# Plot 2: Clustered heatmap with original values
print(f"[{datetime.now()}] Creating clustered heatmap with original values...")
g = sns.clustermap(original_values_selected.T, cmap='YlOrRd', 
                   figsize=(14, min(20, len(selected_regulons)*0.25)), 
                   cbar_kws={'label': 'Mean Regulon Activity'},
                   xticklabels=True, yticklabels=True, vmin=0)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
g.fig.suptitle(f'SCENIC Regulon Activity - Clustered Original Values\n(Regulons with tissue-specificity score > 0.1, n={len(selected_regulons)})', 
               fontsize=14, y=1.02)
plt.savefig('entero_scenic_heatmap_tissue_specific_original_clustered.png', dpi=300, bbox_inches='tight')
plt.savefig('entero_scenic_heatmap_tissue_specific_original_clustered.pdf', bbox_inches='tight')
plt.close()
print(f"[{datetime.now()}] Saved: entero_scenic_heatmap_tissue_specific_original_clustered.png/pdf")

# Plot 3: Side-by-side comparison - original values and specificity scores
print(f"[{datetime.now()}] Creating side-by-side comparison plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, min(20, len(selected_regulons)*0.25)))

# Original values
sns.heatmap(original_values_selected.T, cmap='YlOrRd', cbar_kws={'label': 'Mean Activity'}, 
            xticklabels=True, yticklabels=True, vmin=0, ax=ax1)
ax1.set_title('Original Mean Regulon Activity', fontsize=14)
ax1.set_xlabel('Tissue Combined', fontsize=12)
ax1.set_ylabel('Regulons', fontsize=12)

# Specificity scores
sns.heatmap(tissue_specificity_selected.T, cmap='RdBu_r', center=0, 
            cbar_kws={'label': 'Specificity Score'}, 
            xticklabels=True, yticklabels=True, ax=ax2)
ax2.set_title('Tissue-Specificity Scores', fontsize=14)
ax2.set_xlabel('Tissue Combined', fontsize=12)
ax2.set_ylabel('')

plt.suptitle(f'SCENIC Analysis: Tissue-Specific Regulons (specificity > 0.1, n={len(selected_regulons)})', 
             fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('entero_scenic_comparison_original_vs_specificity.png', dpi=300, bbox_inches='tight')
plt.savefig('entero_scenic_comparison_original_vs_specificity.pdf', bbox_inches='tight')
plt.close()
print(f"[{datetime.now()}] Saved: entero_scenic_comparison_original_vs_specificity.png/pdf")

# Plot 4: Top 30 most tissue-specific regulons only
print(f"[{datetime.now()}] Creating plot for top 30 most tissue-specific regulons...")
top30_regulons = summary_df.head(30)['regulon'].tolist()
top30_original = original_values_selected[top30_regulons]

plt.figure(figsize=(14, 10))
sns.heatmap(top30_original.T, cmap='YlOrRd', cbar_kws={'label': 'Mean Regulon Activity'},
            xticklabels=True, yticklabels=True, vmin=0, annot=True, fmt='.3f')
plt.title('Top 30 Most Tissue-Specific Regulons - Original Values', fontsize=14)
plt.xlabel('Tissue Combined', fontsize=12)
plt.ylabel('Regulons', fontsize=12)
plt.tight_layout()
plt.savefig('entero_scenic_top30_tissue_specific_original.png', dpi=300, bbox_inches='tight')
plt.savefig('entero_scenic_top30_tissue_specific_original.pdf', bbox_inches='tight')
plt.close()
print(f"[{datetime.now()}] Saved: entero_scenic_top30_tissue_specific_original.png/pdf")

# Print summary statistics
print(f"\n[{datetime.now()}] Summary Statistics:")
print(f"Total regulons analyzed: {len(regulon_cols)}")
print(f"Regulons with tissue-specificity score > 0.1: {len(selected_regulons)}")
print(f"Percentage of tissue-specific regulons: {len(selected_regulons)/len(regulon_cols)*100:.1f}%")
print(f"\nTop 10 most tissue-specific regulons:")
print(summary_df.head(10)[['regulon', 'max_specificity_score', 'most_specific_tissue']].to_string(index=False))

# Save summary statistics
with open('entero_tissue_specific_analysis_summary.txt', 'w') as f:
    f.write(f"SCENIC Tissue-Specific Analysis Summary\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write(f"=" * 50 + "\n\n")
    f.write(f"Total regulons analyzed: {len(regulon_cols)}\n")
    f.write(f"Regulons with tissue-specificity score > 0.1: {len(selected_regulons)}\n")
    f.write(f"Percentage of tissue-specific regulons: {len(selected_regulons)/len(regulon_cols)*100:.1f}%\n\n")
    f.write(f"Tissue distribution of most specific regulons:\n")
    tissue_dist = summary_df['most_specific_tissue'].value_counts()
    for tissue, count in tissue_dist.items():
        f.write(f"  {tissue}: {count} regulons\n")
    f.write(f"\nTop 20 most tissue-specific regulons:\n")
    f.write(summary_df.head(20).to_string(index=False))

print(f"[{datetime.now()}] Saved analysis summary: entero_tissue_specific_analysis_summary.txt")

print(f"\n[{datetime.now()}] Analysis completed successfully!")
print("\nGenerated files:")
print("  - entero_tissue_specificity_scores_above_0.1.csv")
print("  - entero_original_activity_tissue_specific_above_0.1.csv")
print("  - entero_tissue_specific_regulons_summary.csv")
print("  - entero_tissue_specific_analysis_summary.txt")
print("  - entero_scenic_heatmap_tissue_specific_original_values.png/pdf")
print("  - entero_scenic_heatmap_tissue_specific_original_clustered.png/pdf")
print("  - entero_scenic_comparison_original_vs_specificity.png/pdf")
print("  - entero_scenic_top30_tissue_specific_original.png/pdf")