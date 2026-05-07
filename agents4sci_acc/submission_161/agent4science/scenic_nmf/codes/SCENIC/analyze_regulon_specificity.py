#!/usr/bin/env python3

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

print(f"Starting regulon specificity analysis at {datetime.now()}")

# Load the h5ad file
print("\nLoading data...")
adata = sc.read_h5ad('entero_hg38_scenic_full_results_with_new_umap.h5ad')
print(f"Loaded data with shape: {adata.shape}")

# Get regulon columns
regulon_columns = [col for col in adata.obs.columns if '(+)' in col or '(-)' in col]
print(f"Found {len(regulon_columns)} regulons")

# Initialize results dictionary
results = []

def calculate_specificity_metrics(adata, regulon, group_column):
    """
    Calculate specificity metrics for a regulon across groups
    """
    groups = adata.obs[group_column].unique()
    group_means = {}
    group_stds = {}
    
    # Calculate mean and std for each group
    for group in groups:
        mask = adata.obs[group_column] == group
        group_vals = adata.obs.loc[mask, regulon].values
        group_means[group] = np.mean(group_vals)
        group_stds[group] = np.std(group_vals)
    
    # Find group with highest mean activity
    max_group = max(group_means, key=group_means.get)
    max_mean = group_means[max_group]
    
    # Calculate fold change vs other groups
    other_means = [v for k, v in group_means.items() if k != max_group]
    if other_means:
        avg_other_mean = np.mean(other_means)
        if avg_other_mean > 0:
            fold_change = max_mean / avg_other_mean
        else:
            fold_change = np.inf if max_mean > 0 else 1.0
    else:
        fold_change = 1.0
    
    # Calculate specificity score (Cohen's d effect size)
    if len(groups) > 1:
        mask_max = adata.obs[group_column] == max_group
        mask_other = ~mask_max
        
        vals_max = adata.obs.loc[mask_max, regulon].values
        vals_other = adata.obs.loc[mask_other, regulon].values
        
        # Cohen's d
        pooled_std = np.sqrt(((len(vals_max)-1)*np.var(vals_max) + 
                              (len(vals_other)-1)*np.var(vals_other)) / 
                             (len(vals_max) + len(vals_other) - 2))
        if pooled_std > 0:
            cohens_d = (np.mean(vals_max) - np.mean(vals_other)) / pooled_std
        else:
            cohens_d = 0.0
        
        # Statistical test (Mann-Whitney U)
        try:
            statistic, pvalue = stats.mannwhitneyu(vals_max, vals_other, alternative='greater')
        except:
            pvalue = 1.0
        
        # Calculate AUC-ROC as discrimination metric
        try:
            y_true = mask_max.astype(int)
            y_scores = adata.obs[regulon].values
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.5
    else:
        cohens_d = 0.0
        pvalue = 1.0
        auc = 0.5
    
    return {
        'specific_group': max_group,
        'mean_activity': max_mean,
        'fold_change': fold_change,
        'cohens_d': cohens_d,
        'pvalue': pvalue,
        'auc_roc': auc,
        'n_groups': len(groups)
    }

# Analyze each regulon
print("\nAnalyzing regulon specificity...")
for i, regulon in enumerate(regulon_columns):
    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/{len(regulon_columns)}")
    
    # Tissue specificity
    tissue_metrics = calculate_specificity_metrics(adata, regulon, 'tissue_combined')
    
    # Disease specificity  
    disease_metrics = calculate_specificity_metrics(adata, regulon, 'disease')
    
    # Cell type specificity
    celltype_metrics = calculate_specificity_metrics(adata, regulon, 'inferred_cell_type')
    
    results.append({
        'regulon': regulon,
        # Tissue specificity
        'tissue_specific': tissue_metrics['specific_group'],
        'tissue_mean_activity': tissue_metrics['mean_activity'],
        'tissue_fold_change': tissue_metrics['fold_change'],
        'tissue_cohens_d': tissue_metrics['cohens_d'],
        'tissue_pvalue': tissue_metrics['pvalue'],
        'tissue_auc': tissue_metrics['auc_roc'],
        # Disease specificity
        'disease_specific': disease_metrics['specific_group'],
        'disease_mean_activity': disease_metrics['mean_activity'],
        'disease_fold_change': disease_metrics['fold_change'],
        'disease_cohens_d': disease_metrics['cohens_d'],
        'disease_pvalue': disease_metrics['pvalue'],
        'disease_auc': disease_metrics['auc_roc'],
        # Cell type specificity
        'celltype_specific': celltype_metrics['specific_group'],
        'celltype_mean_activity': celltype_metrics['mean_activity'],
        'celltype_fold_change': celltype_metrics['fold_change'],
        'celltype_cohens_d': celltype_metrics['cohens_d'],
        'celltype_pvalue': celltype_metrics['pvalue'],
        'celltype_auc': celltype_metrics['auc_roc']
    })

# Create DataFrame
print("\nCreating results table...")
df_results = pd.DataFrame(results)

# Add significance flags (Bonferroni correction)
bonferroni_threshold = 0.05 / len(regulon_columns)
df_results['tissue_significant'] = df_results['tissue_pvalue'] < bonferroni_threshold
df_results['disease_significant'] = df_results['disease_pvalue'] < bonferroni_threshold
df_results['celltype_significant'] = df_results['celltype_pvalue'] < bonferroni_threshold

# Add specificity categories based on effect size
def categorize_specificity(cohens_d, pvalue, auc):
    if pvalue >= bonferroni_threshold:
        return 'Non-specific'
    elif abs(cohens_d) >= 0.8 and auc >= 0.7:
        return 'Highly specific'
    elif abs(cohens_d) >= 0.5 and auc >= 0.6:
        return 'Moderately specific'
    elif abs(cohens_d) >= 0.2:
        return 'Weakly specific'
    else:
        return 'Non-specific'

df_results['tissue_specificity_category'] = df_results.apply(
    lambda x: categorize_specificity(x['tissue_cohens_d'], x['tissue_pvalue'], x['tissue_auc']), axis=1)
df_results['disease_specificity_category'] = df_results.apply(
    lambda x: categorize_specificity(x['disease_cohens_d'], x['disease_pvalue'], x['disease_auc']), axis=1)
df_results['celltype_specificity_category'] = df_results.apply(
    lambda x: categorize_specificity(x['celltype_cohens_d'], x['celltype_pvalue'], x['celltype_auc']), axis=1)

# Sort by tissue specificity
df_results = df_results.sort_values('tissue_cohens_d', ascending=False)

# Save full results
print("\nSaving results...")
df_results.to_csv('regulon_specificity_full_results.csv', index=False)
print("  Saved full results to: regulon_specificity_full_results.csv")

# Create summary table with key metrics
summary_columns = [
    'regulon',
    'tissue_specific', 'tissue_fold_change', 'tissue_specificity_category',
    'disease_specific', 'disease_fold_change', 'disease_specificity_category',
    'celltype_specific', 'celltype_fold_change', 'celltype_specificity_category'
]
df_summary = df_results[summary_columns].copy()

# Round fold changes for readability
df_summary['tissue_fold_change'] = df_summary['tissue_fold_change'].round(2)
df_summary['disease_fold_change'] = df_summary['disease_fold_change'].round(2)
df_summary['celltype_fold_change'] = df_summary['celltype_fold_change'].round(2)

# Save summary
df_summary.to_csv('regulon_specificity_summary.csv', index=False)
print("  Saved summary to: regulon_specificity_summary.csv")

# Create top specific regulons for each category
print("\n=== Top Tissue-Specific Regulons ===")
top_tissue = df_results[df_results['tissue_specificity_category'].isin(['Highly specific', 'Moderately specific'])].head(20)
for _, row in top_tissue.iterrows():
    print(f"{row['regulon']:20s} -> {row['tissue_specific']:20s} (FC={row['tissue_fold_change']:.2f}, d={row['tissue_cohens_d']:.2f})")

print("\n=== Top Disease-Specific Regulons ===")
top_disease = df_results[df_results['disease_specificity_category'].isin(['Highly specific', 'Moderately specific'])].head(20)
for _, row in top_disease.iterrows():
    print(f"{row['regulon']:20s} -> {row['disease_specific']:20s} (FC={row['disease_fold_change']:.2f}, d={row['disease_cohens_d']:.2f})")

print("\n=== Top Cell Type-Specific Regulons ===")
top_celltype = df_results[df_results['celltype_specificity_category'].isin(['Highly specific', 'Moderately specific'])].head(20)
for _, row in top_celltype.iterrows():
    print(f"{row['regulon']:20s} -> {row['celltype_specific']:30s} (FC={row['celltype_fold_change']:.2f}, d={row['celltype_cohens_d']:.2f})")

# Create Excel file with multiple sheets
with pd.ExcelWriter('regulon_specificity_report.xlsx', engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='Summary', index=False)
    df_results.to_excel(writer, sheet_name='Full Results', index=False)
    top_tissue.to_excel(writer, sheet_name='Top Tissue Specific', index=False)
    top_disease.to_excel(writer, sheet_name='Top Disease Specific', index=False)
    top_celltype.to_excel(writer, sheet_name='Top Cell Type Specific', index=False)
print("\n  Saved Excel report to: regulon_specificity_report.xlsx")

# Generate statistics summary
print("\n=== Overall Statistics ===")
print(f"Total regulons analyzed: {len(df_results)}")
print("\nTissue Specificity:")
print(df_results['tissue_specificity_category'].value_counts())
print("\nDisease Specificity:")
print(df_results['disease_specificity_category'].value_counts())
print("\nCell Type Specificity:")
print(df_results['celltype_specificity_category'].value_counts())

print(f"\nAnalysis completed at {datetime.now()}")