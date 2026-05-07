#!/usr/bin/env python3

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=150, facecolor='white')

print(f"Starting comprehensive SCENIC UMAP plotting at {datetime.now()}")

# Load the h5ad file with new UMAP
adata = sc.read_h5ad('entero_hg38_scenic_full_results_with_new_umap.h5ad')
print(f"Loaded data with shape: {adata.shape}")

# Create output directory
output_dir = 'scenic_comprehensive_plots'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/regulons', exist_ok=True)
os.makedirs(f'{output_dir}/metadata', exist_ok=True)

# Use the SCENIC-based UMAP
adata.obsm['X_umap'] = adata.obsm['X_umap_scenic'].copy()

# ====================
# 1. Plot metadata features
# ====================
print("\n=== Plotting metadata features ===")

metadata_features = [
    'tissue_combined',
    'tissue',
    'cell_type',
    'inferred_cell_type',
    'disease',
    'disease_ontology_term_id',
    'donor_id',
    'sex_ontology_term_id',
    'donor_age'
]

for feature in metadata_features:
    if feature in adata.obs.columns:
        print(f"Plotting {feature}...")
        
        # Handle categorical vs continuous
        if feature == 'donor_age':
            # Convert age to numeric if it's not
            try:
                adata.obs['donor_age_numeric'] = pd.to_numeric(adata.obs['donor_age'], errors='coerce')
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                sc.pl.umap(adata, color='donor_age_numeric', ax=ax, show=False,
                          frameon=False, title=f'Donor Age', cmap='viridis')
                plt.savefig(f'{output_dir}/metadata/umap_{feature}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
            except:
                pass
        else:
            # Count unique values
            n_unique = adata.obs[feature].nunique()
            
            # Choose appropriate figure size and legend position
            if n_unique > 20:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                sc.pl.umap(adata, color=feature, ax=ax, show=False,
                          frameon=False, title=feature.replace('_', ' ').title(),
                          legend_loc='on data', legend_fontsize=8)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                sc.pl.umap(adata, color=feature, ax=ax, show=False,
                          frameon=False, title=feature.replace('_', ' ').title())
            
            plt.savefig(f'{output_dir}/metadata/umap_{feature}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

# ====================
# 2. Plot top regulons individually
# ====================
print("\n=== Plotting top regulons ===")

regulon_columns = [col for col in adata.obs.columns if '(+)' in col or '(-)' in col]
print(f"Found {len(regulon_columns)} regulons total")

# Calculate mean activity for each regulon to identify top ones
regulon_means = {}
for reg in regulon_columns:
    regulon_means[reg] = adata.obs[reg].mean()

# Sort by mean activity
top_regulons = sorted(regulon_means.items(), key=lambda x: x[1], reverse=True)[:50]

print(f"Plotting top 50 regulons by mean activity...")
for i, (regulon, mean_activity) in enumerate(top_regulons, 1):
    if i % 10 == 0:
        print(f"  Progress: {i}/50")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sc.pl.umap(adata, color=regulon, ax=ax, show=False,
               frameon=False, title=f'{regulon} (mean: {mean_activity:.3f})',
               cmap='viridis', vmin=0)
    
    safe_name = regulon.replace('(+)', '_pos').replace('(-)', '_neg')
    safe_name = safe_name.replace('/', '_').replace(' ', '_')
    
    plt.savefig(f'{output_dir}/regulons/umap_{safe_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# ====================
# 3. Create summary grid plots
# ====================
print("\n=== Creating summary grid plots ===")

# Grid plot for tissue and disease
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

features_grid = ['tissue_combined', 'disease', 'cell_type', 'inferred_cell_type']
for i, feature in enumerate(features_grid):
    if feature in adata.obs.columns and i < len(axes):
        sc.pl.umap(adata, color=feature, ax=axes[i], show=False,
                  frameon=False, title=feature.replace('_', ' ').title())

plt.tight_layout()
plt.savefig(f'{output_dir}/summary_metadata_grid.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved metadata grid plot")

# Grid plot for top 16 regulons
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()

for i, (regulon, mean_activity) in enumerate(top_regulons[:16]):
    sc.pl.umap(adata, color=regulon, ax=axes[i], show=False,
               frameon=False, title=f'{regulon}', cmap='viridis', vmin=0)

plt.tight_layout()
plt.savefig(f'{output_dir}/top16_regulons_grid.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved top 16 regulons grid plot")

# ====================
# 4. Create combined visualization
# ====================
print("\n=== Creating combined visualization ===")

# Select key regulons for endocrine cells
endocrine_regulons = []
for reg in regulon_columns:
    if any(tf in reg.upper() for tf in ['NEUROD', 'PAX', 'ISL', 'FOXA', 'PDX', 'NKX', 'INSM']):
        endocrine_regulons.append(reg)

if endocrine_regulons:
    print(f"Found {len(endocrine_regulons)} endocrine-related regulons")
    
    # Plot top endocrine regulons
    n_plots = min(9, len(endocrine_regulons))
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, regulon in enumerate(endocrine_regulons[:n_plots]):
        sc.pl.umap(adata, color=regulon, ax=axes[i], show=False,
                   frameon=False, title=regulon, cmap='viridis', vmin=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/endocrine_regulons_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved endocrine regulons grid plot")

# ====================
# 5. Generate summary statistics
# ====================
print("\n=== Generating summary report ===")

with open(f'{output_dir}/plotting_summary.txt', 'w') as f:
    f.write(f"SCENIC Comprehensive Plotting Summary\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write(f"{'='*50}\n\n")
    
    f.write(f"Dataset: {adata.shape[0]} cells × {adata.shape[1]} genes\n")
    f.write(f"Total regulons analyzed: {len(regulon_columns)}\n\n")
    
    f.write("Metadata features plotted:\n")
    for feature in metadata_features:
        if feature in adata.obs.columns:
            n_unique = adata.obs[feature].nunique()
            f.write(f"  - {feature}: {n_unique} unique values\n")
    
    f.write(f"\nTop 20 regulons by mean activity:\n")
    for i, (reg, mean_val) in enumerate(top_regulons[:20], 1):
        f.write(f"  {i}. {reg}: {mean_val:.4f}\n")
    
    f.write(f"\nTissue distribution:\n")
    if 'tissue_combined' in adata.obs.columns:
        tissue_counts = adata.obs['tissue_combined'].value_counts()
        for tissue, count in tissue_counts.items():
            percentage = (count / len(adata.obs)) * 100
            f.write(f"  - {tissue}: {count} cells ({percentage:.1f}%)\n")

print(f"\nAll plots saved in '{output_dir}' directory")
print(f"Completed at {datetime.now()}")