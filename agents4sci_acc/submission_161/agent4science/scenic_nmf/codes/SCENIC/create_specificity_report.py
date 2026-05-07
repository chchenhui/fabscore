#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Load the results
df_full = pd.read_csv('regulon_specificity_full_results.csv')
df_summary = pd.read_csv('regulon_specificity_summary.csv')

# Create detailed report
print("=" * 80)
print("REGULON SPECIFICITY ANALYSIS REPORT")
print("=" * 80)

# Overall statistics
print("\n### OVERALL STATISTICS ###")
print(f"Total regulons analyzed: {len(df_summary)}")

# Tissue specificity
print("\n### TISSUE SPECIFICITY ###")
tissue_counts = df_summary['tissue_specificity_category'].value_counts()
print("Distribution by category:")
for cat, count in tissue_counts.items():
    percentage = (count / len(df_summary)) * 100
    print(f"  {cat:20s}: {count:4d} ({percentage:5.1f}%)")

print("\nTop 15 Tissue-Specific Regulons:")
top_tissue = df_full.nlargest(15, 'tissue_cohens_d')
for idx, row in top_tissue.iterrows():
    print(f"  {row['regulon']:15s} -> {row['tissue_specific']:25s} "
          f"(FC={row['tissue_fold_change']:6.2f}, Cohen's d={row['tissue_cohens_d']:5.2f}, AUC={row['tissue_auc']:5.3f})")

# Disease specificity
print("\n### DISEASE SPECIFICITY ###")
disease_counts = df_summary['disease_specificity_category'].value_counts()
print("Distribution by category:")
for cat, count in disease_counts.items():
    percentage = (count / len(df_summary)) * 100
    print(f"  {cat:20s}: {count:4d} ({percentage:5.1f}%)")

print("\nTop 15 Disease-Specific Regulons:")
top_disease = df_full.nlargest(15, 'disease_cohens_d')
for idx, row in top_disease.iterrows():
    print(f"  {row['regulon']:15s} -> {row['disease_specific']:30s} "
          f"(FC={row['disease_fold_change']:6.2f}, Cohen's d={row['disease_cohens_d']:5.2f}, AUC={row['disease_auc']:5.3f})")

# Cell type specificity
print("\n### CELL TYPE SPECIFICITY ###")
celltype_counts = df_summary['celltype_specificity_category'].value_counts()
print("Distribution by category:")
for cat, count in celltype_counts.items():
    percentage = (count / len(df_summary)) * 100
    print(f"  {cat:20s}: {count:4d} ({percentage:5.1f}%)")

print("\nTop 15 Cell Type-Specific Regulons:")
top_celltype = df_full.nlargest(15, 'celltype_cohens_d')
for idx, row in top_celltype.iterrows():
    print(f"  {row['regulon']:15s} -> {row['celltype_specific']:30s} "
          f"(FC={row['celltype_fold_change']:6.2f}, Cohen's d={row['celltype_cohens_d']:5.2f}, AUC={row['celltype_auc']:5.3f})")

# Multi-specific regulons
print("\n### MULTI-SPECIFIC REGULONS ###")
print("Regulons specific across multiple categories:")

# Find regulons that are highly/moderately specific in multiple categories
multi_specific = df_summary[
    (df_summary['tissue_specificity_category'].isin(['Highly specific', 'Moderately specific'])) |
    (df_summary['disease_specificity_category'].isin(['Highly specific', 'Moderately specific'])) |
    (df_summary['celltype_specificity_category'].isin(['Highly specific', 'Moderately specific']))
]

tissue_and_disease = multi_specific[
    (multi_specific['tissue_specificity_category'].isin(['Highly specific', 'Moderately specific'])) &
    (multi_specific['disease_specificity_category'].isin(['Highly specific', 'Moderately specific']))
]

if len(tissue_and_disease) > 0:
    print(f"\nTissue AND Disease specific ({len(tissue_and_disease)} regulons):")
    for idx, row in tissue_and_disease.head(10).iterrows():
        print(f"  {row['regulon']:15s}: {row['tissue_specific']:20s} ({row['tissue_specificity_category']}) "
              f"& {row['disease_specific']:25s} ({row['disease_specificity_category']})")

# Tissue distribution
print("\n### TISSUE DISTRIBUTION OF SPECIFIC REGULONS ###")
tissue_specific_regulons = df_summary[df_summary['tissue_specificity_category'].isin(['Highly specific', 'Moderately specific'])]
tissue_dist = tissue_specific_regulons['tissue_specific'].value_counts()
print("Number of specific regulons per tissue:")
for tissue, count in tissue_dist.items():
    print(f"  {tissue:30s}: {count:3d} regulons")

# Disease distribution
print("\n### DISEASE DISTRIBUTION OF SPECIFIC REGULONS ###")
disease_specific_regulons = df_summary[df_summary['disease_specificity_category'].isin(['Highly specific', 'Moderately specific'])]
disease_dist = disease_specific_regulons['disease_specific'].value_counts()
print("Number of specific regulons per disease:")
for disease, count in disease_dist.items():
    print(f"  {disease:40s}: {count:3d} regulons")

# Save report to file
with open('regulon_specificity_report.txt', 'w') as f:
    # Redirect print to file
    import sys
    old_stdout = sys.stdout
    sys.stdout = f
    
    # Repeat all the prints above
    print("=" * 80)
    print("REGULON SPECIFICITY ANALYSIS REPORT")
    print("=" * 80)
    
    print("\n### OVERALL STATISTICS ###")
    print(f"Total regulons analyzed: {len(df_summary)}")
    
    print("\n### TISSUE SPECIFICITY ###")
    print("Distribution by category:")
    for cat, count in tissue_counts.items():
        percentage = (count / len(df_summary)) * 100
        print(f"  {cat:20s}: {count:4d} ({percentage:5.1f}%)")
    
    print("\nTop 20 Tissue-Specific Regulons:")
    top_tissue = df_full.nlargest(20, 'tissue_cohens_d')
    for idx, row in top_tissue.iterrows():
        print(f"  {row['regulon']:15s} -> {row['tissue_specific']:25s} "
              f"(FC={row['tissue_fold_change']:6.2f}, Cohen's d={row['tissue_cohens_d']:5.2f}, AUC={row['tissue_auc']:5.3f})")
    
    print("\n### DISEASE SPECIFICITY ###")
    print("Distribution by category:")
    for cat, count in disease_counts.items():
        percentage = (count / len(df_summary)) * 100
        print(f"  {cat:20s}: {count:4d} ({percentage:5.1f}%)")
    
    print("\nTop 20 Disease-Specific Regulons:")
    top_disease = df_full.nlargest(20, 'disease_cohens_d')
    for idx, row in top_disease.iterrows():
        print(f"  {row['regulon']:15s} -> {row['disease_specific']:30s} "
              f"(FC={row['disease_fold_change']:6.2f}, Cohen's d={row['disease_cohens_d']:5.2f}, AUC={row['disease_auc']:5.3f})")
    
    sys.stdout = old_stdout

print("\nReport saved to: regulon_specificity_report.txt")