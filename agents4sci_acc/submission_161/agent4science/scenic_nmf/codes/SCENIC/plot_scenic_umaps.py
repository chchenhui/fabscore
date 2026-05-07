#!/usr/bin/env python3

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=150, facecolor='white')

print(f"Starting SCENIC UMAP plotting at {datetime.now()}")

adata = sc.read_h5ad('entero_hg38_scenic_full_results.h5ad')
print(f"Loaded data with shape: {adata.shape}")

regulon_columns = [col for col in adata.obs.columns if '(+)' in col or '(-)' in col]
print(f"Found {len(regulon_columns)} regulons to plot")

output_dir = 'scenic_umap_plots'
os.makedirs(output_dir, exist_ok=True)

if 'X_umap' not in adata.obsm:
    print("Computing UMAP...")
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    sc.tl.umap(adata)
    adata.write('entero_hg38_scenic_full_results.h5ad')
    print("UMAP computed and saved")

for i, regulon in enumerate(regulon_columns, 1):
    print(f"Plotting regulon {i}/{len(regulon_columns)}: {regulon}")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sc.pl.umap(adata, color=regulon, ax=ax, show=False, 
               frameon=False, title=f'{regulon} Activity',
               cmap='viridis')
    
    safe_name = regulon.replace('(+)', '_pos').replace('(-)', '_neg')
    safe_name = safe_name.replace('/', '_').replace(' ', '_')
    
    plt.savefig(f'{output_dir}/umap_{safe_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(regulon_columns)} regulons plotted")

fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()

for i, regulon in enumerate(regulon_columns[:16]):
    sc.pl.umap(adata, color=regulon, ax=axes[i], show=False,
               frameon=False, title=regulon, cmap='viridis')

plt.tight_layout()
plt.savefig(f'{output_dir}/umap_grid_top16.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved grid plot of top 16 regulons")

print(f"\nAll plots saved in '{output_dir}' directory")
print(f"Completed at {datetime.now()}")