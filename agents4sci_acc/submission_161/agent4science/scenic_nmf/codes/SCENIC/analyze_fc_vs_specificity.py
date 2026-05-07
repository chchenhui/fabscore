#!/usr/bin/env python3

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load results
df = pd.read_csv('regulon_specificity_full_results.csv')
df_summary = pd.read_csv('regulon_specificity_summary.csv')

# Load original data to get sample sizes
adata = sc.read_h5ad('entero_hg38_scenic_full_results_with_new_umap.h5ad')

# Calculate sample sizes for each group
tissue_counts = adata.obs['tissue_combined'].value_counts().to_dict()
disease_counts = adata.obs['disease'].value_counts().to_dict()
celltype_counts = adata.obs['inferred_cell_type'].value_counts().to_dict()

# Add sample size information
df['tissue_n_cells'] = df['tissue_specific'].map(tissue_counts)
df['disease_n_cells'] = df['disease_specific'].map(disease_counts)
df['celltype_n_cells'] = df['celltype_specific'].map(celltype_counts)

# Bonferroni threshold
bonferroni_threshold = 0.05 / len(df)
print(f"Bonferroni-corrected p-value threshold: {bonferroni_threshold:.2e}")

# Create detailed analysis
print("\n" + "="*100)
print("ANALYSIS: HIGH FOLD CHANGE BUT NON-SPECIFIC REGULONS")
print("="*100)

# 1. Tissue Analysis
print("\n### TISSUE SPECIFICITY ANALYSIS ###")
print("\nRegulons with FC > 3 but classified as Non-specific:")
high_fc_non_specific_tissue = df[(df['tissue_fold_change'] > 3) & 
                                  (df_summary['tissue_specificity_category'] == 'Non-specific')]
high_fc_non_specific_tissue = high_fc_non_specific_tissue.sort_values('tissue_fold_change', ascending=False)

print(f"\nFound {len(high_fc_non_specific_tissue)} regulons with high FC but non-specific")
print("\nTop 15 examples:")
print(f"{'Regulon':<15} {'Tissue':<25} {'FC':<8} {'p-value':<12} {'Cohen d':<8} {'AUC':<6} {'n_cells':<8} {'Reason'}")
print("-"*110)

for idx, row in high_fc_non_specific_tissue.head(15).iterrows():
    reasons = []
    if row['tissue_pvalue'] >= bonferroni_threshold:
        reasons.append(f"p>{bonferroni_threshold:.1e}")
    if abs(row['tissue_cohens_d']) < 0.8:
        reasons.append(f"d<0.8")
    if row['tissue_auc'] < 0.7:
        reasons.append("AUC<0.7")
    if pd.notna(row['tissue_n_cells']) and row['tissue_n_cells'] < 10:
        reasons.append("n<10")
    
    print(f"{row['regulon']:<15} {row['tissue_specific']:<25} {row['tissue_fold_change']:<8.2f} "
          f"{row['tissue_pvalue']:<12.2e} {row['tissue_cohens_d']:<8.2f} {row['tissue_auc']:<6.3f} "
          f"{row['tissue_n_cells']:<8.0f} {', '.join(reasons)}")

# 2. Disease Analysis
print("\n### DISEASE SPECIFICITY ANALYSIS ###")
print("\nRegulons with FC > 2 but classified as Non-specific:")
high_fc_non_specific_disease = df[(df['disease_fold_change'] > 2) & 
                                   (df_summary['disease_specificity_category'] == 'Non-specific')]
high_fc_non_specific_disease = high_fc_non_specific_disease.sort_values('disease_fold_change', ascending=False)

print(f"\nFound {len(high_fc_non_specific_disease)} regulons with high FC but non-specific")
print("\nTop 15 examples:")
print(f"{'Regulon':<15} {'Disease':<30} {'FC':<8} {'p-value':<12} {'Cohen d':<8} {'AUC':<6} {'n_cells':<8} {'Reason'}")
print("-"*115)

for idx, row in high_fc_non_specific_disease.head(15).iterrows():
    reasons = []
    if row['disease_pvalue'] >= bonferroni_threshold:
        reasons.append(f"p>{bonferroni_threshold:.1e}")
    if abs(row['disease_cohens_d']) < 0.8:
        reasons.append(f"d<0.8")
    if row['disease_auc'] < 0.7:
        reasons.append("AUC<0.7")
    if pd.notna(row['disease_n_cells']) and row['disease_n_cells'] < 10:
        reasons.append("n<10")
    
    print(f"{row['regulon']:<15} {row['disease_specific']:<30} {row['disease_fold_change']:<8.2f} "
          f"{row['disease_pvalue']:<12.2e} {row['disease_cohens_d']:<8.2f} {row['disease_auc']:<6.3f} "
          f"{row['disease_n_cells'] if pd.notna(row['disease_n_cells']) else 0:<8.0f} {', '.join(reasons)}")

# 3. Create comparison table
print("\n### COMPARISON: HIGH FC SPECIFIC vs NON-SPECIFIC ###")

# For tissue
tissue_specific_high_fc = df[(df['tissue_fold_change'] > 3) & 
                             (df_summary['tissue_specificity_category'].isin(['Highly specific', 'Moderately specific']))]
tissue_nonspecific_high_fc = df[(df['tissue_fold_change'] > 3) & 
                                (df_summary['tissue_specificity_category'] == 'Non-specific')]

print(f"\nTissue-specific regulons with FC>3:")
print(f"  Specific: {len(tissue_specific_high_fc)} regulons")
print(f"    Mean p-value: {tissue_specific_high_fc['tissue_pvalue'].mean():.2e}")
print(f"    Mean Cohen's d: {tissue_specific_high_fc['tissue_cohens_d'].mean():.2f}")
print(f"    Mean AUC: {tissue_specific_high_fc['tissue_auc'].mean():.3f}")
print(f"    Mean n_cells: {tissue_specific_high_fc['tissue_n_cells'].mean():.0f}")

print(f"\n  Non-specific: {len(tissue_nonspecific_high_fc)} regulons")
print(f"    Mean p-value: {tissue_nonspecific_high_fc['tissue_pvalue'].mean():.2e}")
print(f"    Mean Cohen's d: {tissue_nonspecific_high_fc['tissue_cohens_d'].mean():.2f}")
print(f"    Mean AUC: {tissue_nonspecific_high_fc['tissue_auc'].mean():.3f}")
print(f"    Mean n_cells: {tissue_nonspecific_high_fc['tissue_n_cells'].mean():.0f}")

# 4. Statistical power analysis
print("\n### STATISTICAL POWER ANALYSIS ###")
print("\nGroups with very small sample sizes (<20 cells):")
small_groups = []
for tissue, count in tissue_counts.items():
    if count < 20:
        small_groups.append(f"  Tissue '{tissue}': {count} cells")
for disease, count in disease_counts.items():
    if count < 20:
        small_groups.append(f"  Disease '{disease}': {count} cells")

for group in small_groups:
    print(group)

print(f"\nNote: Small sample sizes lead to:")
print("  - High variance in fold change estimates")
print("  - Low statistical power (high p-values)")
print("  - Unreliable effect size estimates")

# 5. Create detailed table for export
detailed_results = df[['regulon', 
                       'tissue_specific', 'tissue_fold_change', 'tissue_pvalue', 'tissue_cohens_d', 'tissue_auc', 'tissue_n_cells',
                       'disease_specific', 'disease_fold_change', 'disease_pvalue', 'disease_cohens_d', 'disease_auc', 'disease_n_cells']].copy()

# Add significance flags
detailed_results['tissue_signif'] = detailed_results['tissue_pvalue'] < bonferroni_threshold
detailed_results['disease_signif'] = detailed_results['disease_pvalue'] < bonferroni_threshold

# Sort by tissue fold change
detailed_results = detailed_results.sort_values('tissue_fold_change', ascending=False)

# Save detailed results
detailed_results.to_csv('regulon_detailed_fc_pvalue_analysis.csv', index=False)
print(f"\nDetailed results saved to: regulon_detailed_fc_pvalue_analysis.csv")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Tissue plots
ax = axes[0, 0]
ax.scatter(df['tissue_fold_change'], -np.log10(df['tissue_pvalue']), alpha=0.5, s=20)
ax.axhline(y=-np.log10(bonferroni_threshold), color='r', linestyle='--', label='Bonferroni threshold')
ax.axvline(x=2, color='g', linestyle='--', label='FC=2')
ax.set_xlabel('Fold Change')
ax.set_ylabel('-log10(p-value)')
ax.set_title('Tissue: Volcano Plot')
ax.legend()

ax = axes[0, 1]
ax.scatter(df['tissue_fold_change'], df['tissue_cohens_d'], alpha=0.5, s=20)
ax.axhline(y=0.8, color='r', linestyle='--', label="Cohen's d=0.8")
ax.axvline(x=2, color='g', linestyle='--', label='FC=2')
ax.set_xlabel('Fold Change')
ax.set_ylabel("Cohen's d")
ax.set_title('Tissue: FC vs Effect Size')
ax.legend()

ax = axes[0, 2]
ax.scatter(df['tissue_fold_change'], df['tissue_auc'], alpha=0.5, s=20)
ax.axhline(y=0.7, color='r', linestyle='--', label='AUC=0.7')
ax.axvline(x=2, color='g', linestyle='--', label='FC=2')
ax.set_xlabel('Fold Change')
ax.set_ylabel('AUC-ROC')
ax.set_title('Tissue: FC vs AUC')
ax.legend()

# Disease plots
ax = axes[1, 0]
ax.scatter(df['disease_fold_change'], -np.log10(df['disease_pvalue']), alpha=0.5, s=20)
ax.axhline(y=-np.log10(bonferroni_threshold), color='r', linestyle='--', label='Bonferroni threshold')
ax.axvline(x=2, color='g', linestyle='--', label='FC=2')
ax.set_xlabel('Fold Change')
ax.set_ylabel('-log10(p-value)')
ax.set_title('Disease: Volcano Plot')
ax.legend()

ax = axes[1, 1]
ax.scatter(df['disease_fold_change'], df['disease_cohens_d'], alpha=0.5, s=20)
ax.axhline(y=0.8, color='r', linestyle='--', label="Cohen's d=0.8")
ax.axvline(x=2, color='g', linestyle='--', label='FC=2')
ax.set_xlabel('Fold Change')
ax.set_ylabel("Cohen's d")
ax.set_title('Disease: FC vs Effect Size')
ax.legend()

ax = axes[1, 2]
ax.scatter(df['disease_fold_change'], df['disease_auc'], alpha=0.5, s=20)
ax.axhline(y=0.7, color='r', linestyle='--', label='AUC=0.7')
ax.axvline(x=2, color='g', linestyle='--', label='FC=2')
ax.set_xlabel('Fold Change')
ax.set_ylabel('AUC-ROC')
ax.set_title('Disease: FC vs AUC')
ax.legend()

plt.tight_layout()
plt.savefig('fc_vs_specificity_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: fc_vs_specificity_analysis.png")

plt.close()

# Summary statistics
print("\n### SUMMARY ###")
print(f"Bonferroni correction threshold: p < {bonferroni_threshold:.2e}")
print(f"Total regulons: {len(df)}")
print(f"\nTissue analysis:")
print(f"  Regulons with FC>3: {len(df[df['tissue_fold_change'] > 3])}")
print(f"  Of these, significant (p<{bonferroni_threshold:.2e}): {len(df[(df['tissue_fold_change'] > 3) & (df['tissue_pvalue'] < bonferroni_threshold)])}")
print(f"\nDisease analysis:")
print(f"  Regulons with FC>2: {len(df[df['disease_fold_change'] > 2])}")
print(f"  Of these, significant (p<{bonferroni_threshold:.2e}): {len(df[(df['disease_fold_change'] > 2) & (df['disease_pvalue'] < bonferroni_threshold)])}")