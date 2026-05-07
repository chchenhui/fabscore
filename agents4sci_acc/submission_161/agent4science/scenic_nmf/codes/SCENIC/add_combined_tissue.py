#!/usr/bin/env python3

import scanpy as sc
import pandas as pd
import numpy as np
from tissue import tissue_dict

print("Loading h5ad file...")
adata = sc.read_h5ad('entero_hg38_scenic_full_results.h5ad')
print(f"Loaded data with shape: {adata.shape}")

# Create reverse mapping: tissue name -> combined category
tissue_to_combined = {}
for combined_category, tissue_list in tissue_dict.items():
    for tissue in tissue_list:
        tissue_to_combined[tissue.lower()] = combined_category

# Function to map tissue to combined category
def map_tissue_to_combined(tissue_value):
    if pd.isna(tissue_value):
        return "Unknown"
    
    tissue_lower = str(tissue_value).lower().strip()
    
    # Direct match
    if tissue_lower in tissue_to_combined:
        return tissue_to_combined[tissue_lower]
    
    # Partial match - check if any key is contained in the tissue value
    for tissue_key, combined_cat in tissue_to_combined.items():
        if tissue_key in tissue_lower:
            return combined_cat
    
    # Check if tissue value contains any of the combined categories
    for combined_cat, tissue_list in tissue_dict.items():
        for tissue in tissue_list:
            if tissue.lower() in tissue_lower:
                return combined_cat
    
    return "Other/Unclassified"

# Apply mapping to the 'tissue' column
print("\nMapping tissues to combined categories...")
adata.obs['tissue_combined'] = adata.obs['tissue'].apply(map_tissue_to_combined)

# Print summary statistics
print("\n=== Combined Tissue Summary ===")
tissue_summary = adata.obs['tissue_combined'].value_counts()
print(tissue_summary.to_string())

print(f"\nTotal cells: {len(adata.obs)}")
print(f"Number of combined tissue categories: {len(tissue_summary)}")

# Calculate percentage for each category
print("\n=== Percentage Distribution ===")
tissue_percentage = (tissue_summary / len(adata.obs) * 100).round(2)
for category, percentage in tissue_percentage.items():
    print(f"{category}: {percentage}% ({tissue_summary[category]} cells)")

# Create cross-tabulation to see mapping
print("\n=== Mapping Details (top 20 original tissues) ===")
crosstab = pd.crosstab(adata.obs['tissue'], adata.obs['tissue_combined'])
print(crosstab.head(20).to_string())

# Save the updated h5ad file
output_file = 'entero_hg38_scenic_full_results_with_tissue_combined.h5ad'
print(f"\nSaving updated h5ad file to: {output_file}")
adata.write(output_file)
print("File saved successfully!")

# Also save a mapping summary as CSV for reference
mapping_summary = pd.DataFrame({
    'original_tissue': adata.obs['tissue'],
    'tissue_combined': adata.obs['tissue_combined']
}).drop_duplicates().sort_values(['tissue_combined', 'original_tissue'])

mapping_summary.to_csv('tissue_mapping_summary.csv', index=False)
print(f"Tissue mapping summary saved to: tissue_mapping_summary.csv")