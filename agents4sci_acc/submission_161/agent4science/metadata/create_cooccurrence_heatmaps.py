#!/usr/bin/env python3
"""
Create co-occurrence heatmaps for tissue-disease and cell type-disease relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load metadata
print("Loading endocrine datasets metadata...")
df = pd.read_csv('/scratch/rli/project/agent/data_integration/unique_datasets_metadata_final.csv')

def parse_field(field_str):
    """Parse semicolon-separated fields"""
    if pd.isna(field_str):
        return []
    return [x.strip() for x in field_str.split(';')]

# Build co-occurrence matrices
print("\nBuilding co-occurrence matrices...")

# 1. Tissue-Disease co-occurrence
tissue_disease_matrix = defaultdict(lambda: defaultdict(int))
for idx, row in df.iterrows():
    tissues = parse_field(row['tissues'])
    diseases = parse_field(row['diseases'])
    
    for tissue in tissues:
        for disease in diseases:
            if disease != 'normal':  # Exclude normal samples
                tissue_disease_matrix[tissue][disease] += 1

# 2. Cell Type-Disease co-occurrence
celltype_disease_matrix = defaultdict(lambda: defaultdict(int))
for idx, row in df.iterrows():
    cell_types = parse_field(row['endocrine_cell_types'])
    diseases = parse_field(row['diseases'])
    
    for cell_type in cell_types:
        for disease in diseases:
            if disease != 'normal':  # Exclude normal samples
                celltype_disease_matrix[cell_type][disease] += 1

# Convert to DataFrames for visualization
# Get top tissues and diseases for tissue-disease heatmap
all_tissues_with_disease = set()
all_diseases = set()
for tissue, diseases in tissue_disease_matrix.items():
    if diseases:
        all_tissues_with_disease.add(tissue)
        all_diseases.update(diseases.keys())

# Sort by total occurrences
tissue_totals = {t: sum(tissue_disease_matrix[t].values()) for t in all_tissues_with_disease}
disease_totals = {d: sum(tissue_disease_matrix[t].get(d, 0) for t in all_tissues_with_disease) for d in all_diseases}

top_tissues = sorted(tissue_totals.keys(), key=lambda x: tissue_totals[x], reverse=True)[:20]
top_diseases = sorted(disease_totals.keys(), key=lambda x: disease_totals[x], reverse=True)[:15]

# Create tissue-disease DataFrame
tissue_disease_df = pd.DataFrame(0, index=top_tissues, columns=top_diseases)
for tissue in top_tissues:
    for disease in top_diseases:
        tissue_disease_df.loc[tissue, disease] = tissue_disease_matrix[tissue].get(disease, 0)

# Get all cell types with disease associations
all_celltypes_with_disease = set()
for cell_type, diseases in celltype_disease_matrix.items():
    if diseases:
        all_celltypes_with_disease.add(cell_type)

# Sort cell types by total disease occurrences
celltype_totals = {ct: sum(celltype_disease_matrix[ct].values()) for ct in all_celltypes_with_disease}
top_celltypes = sorted(celltype_totals.keys(), key=lambda x: celltype_totals[x], reverse=True)

# Create cell type-disease DataFrame
celltype_disease_df = pd.DataFrame(0, index=top_celltypes, columns=top_diseases)
for cell_type in top_celltypes:
    for disease in top_diseases:
        celltype_disease_df.loc[cell_type, disease] = celltype_disease_matrix[cell_type].get(disease, 0)

# Create comprehensive figure with both heatmaps
fig = plt.figure(figsize=(24, 20))

# ===== TISSUE-DISEASE HEATMAP =====
ax1 = plt.subplot(2, 2, 1)
sns.heatmap(tissue_disease_df, annot=True, fmt='d', cmap='YlOrRd', 
            cbar_kws={'label': 'Co-occurrence Count'}, ax=ax1,
            linewidths=0.5, linecolor='gray')
ax1.set_title('Tissue-Disease Co-occurrence Matrix', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Disease', fontsize=12)
ax1.set_ylabel('Tissue', fontsize=12)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=9)

# ===== CELL TYPE-DISEASE HEATMAP =====
ax2 = plt.subplot(2, 2, 2)
sns.heatmap(celltype_disease_df, annot=True, fmt='d', cmap='BuPu', 
            cbar_kws={'label': 'Co-occurrence Count'}, ax=ax2,
            linewidths=0.5, linecolor='gray')
ax2.set_title('Cell Type-Disease Co-occurrence Matrix', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Disease', fontsize=12)
ax2.set_ylabel('Cell Type', fontsize=12)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=9)

# ===== NORMALIZED TISSUE-DISEASE HEATMAP =====
ax3 = plt.subplot(2, 2, 3)
# Normalize by row (tissue) to show disease distribution per tissue
tissue_disease_norm = tissue_disease_df.div(tissue_disease_df.sum(axis=1), axis=0) * 100
tissue_disease_norm = tissue_disease_norm.fillna(0)
sns.heatmap(tissue_disease_norm, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Percentage (%)'}, ax=ax3,
            linewidths=0.5, linecolor='gray')
ax3.set_title('Tissue-Disease Distribution (% per Tissue)', fontsize=14, fontweight='bold', pad=20)
ax3.set_xlabel('Disease', fontsize=12)
ax3.set_ylabel('Tissue', fontsize=12)
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=9)

# ===== NORMALIZED CELL TYPE-DISEASE HEATMAP =====
ax4 = plt.subplot(2, 2, 4)
# Normalize by row (cell type) to show disease distribution per cell type
celltype_disease_norm = celltype_disease_df.div(celltype_disease_df.sum(axis=1), axis=0) * 100
celltype_disease_norm = celltype_disease_norm.fillna(0)
sns.heatmap(celltype_disease_norm, annot=True, fmt='.1f', cmap='PuBuGn', 
            cbar_kws={'label': 'Percentage (%)'}, ax=ax4,
            linewidths=0.5, linecolor='gray')
ax4.set_title('Cell Type-Disease Distribution (% per Cell Type)', fontsize=14, fontweight='bold', pad=20)
ax4.set_xlabel('Disease', fontsize=12)
ax4.set_ylabel('Cell Type', fontsize=12)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax4.get_yticklabels(), rotation=0, fontsize=9)

plt.suptitle('Disease Co-occurrence Analysis in Endocrine Cell Datasets', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('disease_cooccurrence_heatmaps.pdf', dpi=300, bbox_inches='tight')
plt.savefig('disease_cooccurrence_heatmaps.png', dpi=150, bbox_inches='tight')
print("\nSaved: disease_cooccurrence_heatmaps.pdf/png")

# Create focused heatmaps for specific disease categories
fig2, axes = plt.subplots(2, 2, figsize=(20, 16))

# Group diseases by category
cancer_diseases = ['lung adenocarcinoma', 'small cell lung carcinoma', 'colorectal cancer', 
                   'adenocarcinoma', 'neuroendocrine carcinoma', 'colorectal neoplasm']
inflammatory_diseases = ['Crohn disease', 'gastritis', 'Crohn ileitis']
metaplasia_diseases = ['Barrett esophagus', 'gastric intestinal metaplasia']
other_diseases = ['COVID-19', 'chronic obstructive pulmonary disease', 'benign prostatic hyperplasia']

# Cancer-focused tissue heatmap
ax = axes[0, 0]
cancer_tissue_df = tissue_disease_df[[d for d in cancer_diseases if d in tissue_disease_df.columns]]
if not cancer_tissue_df.empty:
    cancer_tissue_df = cancer_tissue_df[cancer_tissue_df.sum(axis=1) > 0]
    sns.heatmap(cancer_tissue_df, annot=True, fmt='d', cmap='Reds', ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('Tissue-Cancer Co-occurrence', fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

# Inflammatory-focused tissue heatmap
ax = axes[0, 1]
inflam_tissue_df = tissue_disease_df[[d for d in inflammatory_diseases if d in tissue_disease_df.columns]]
if not inflam_tissue_df.empty:
    inflam_tissue_df = inflam_tissue_df[inflam_tissue_df.sum(axis=1) > 0]
    sns.heatmap(inflam_tissue_df, annot=True, fmt='d', cmap='Oranges', ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('Tissue-Inflammatory Disease Co-occurrence', fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

# Cancer-focused cell type heatmap
ax = axes[1, 0]
cancer_celltype_df = celltype_disease_df[[d for d in cancer_diseases if d in celltype_disease_df.columns]]
if not cancer_celltype_df.empty:
    cancer_celltype_df = cancer_celltype_df[cancer_celltype_df.sum(axis=1) > 0]
    sns.heatmap(cancer_celltype_df, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('Cell Type-Cancer Co-occurrence', fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

# Inflammatory-focused cell type heatmap
ax = axes[1, 1]
inflam_celltype_df = celltype_disease_df[[d for d in inflammatory_diseases if d in celltype_disease_df.columns]]
if not inflam_celltype_df.empty:
    inflam_celltype_df = inflam_celltype_df[inflam_celltype_df.sum(axis=1) > 0]
    sns.heatmap(inflam_celltype_df, annot=True, fmt='d', cmap='Greens', ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('Cell Type-Inflammatory Disease Co-occurrence', fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.suptitle('Disease Category-Specific Co-occurrence Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('disease_category_heatmaps.pdf', dpi=300, bbox_inches='tight')
plt.savefig('disease_category_heatmaps.png', dpi=150, bbox_inches='tight')
print("Saved: disease_category_heatmaps.pdf/png")

# Generate summary statistics
print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)

print("\n1. TISSUE-DISEASE ASSOCIATIONS:")
print(f"   Total unique tissues with disease data: {len(all_tissues_with_disease)}")
print(f"   Total unique diseases (excluding normal): {len(all_diseases)}")
print(f"   Total tissue-disease pairs: {tissue_disease_df.values.sum():.0f}")

# Find strongest associations
tissue_disease_flat = []
for tissue in tissue_disease_df.index:
    for disease in tissue_disease_df.columns:
        if tissue_disease_df.loc[tissue, disease] > 0:
            tissue_disease_flat.append((tissue, disease, tissue_disease_df.loc[tissue, disease]))

tissue_disease_flat.sort(key=lambda x: x[2], reverse=True)
print("\n   Top 10 Strongest Tissue-Disease Associations:")
for tissue, disease, count in tissue_disease_flat[:10]:
    print(f"      {tissue} - {disease}: {count:.0f} occurrences")

print("\n2. CELL TYPE-DISEASE ASSOCIATIONS:")
print(f"   Total cell types with disease data: {len(all_celltypes_with_disease)}")
print(f"   Total cell type-disease pairs: {celltype_disease_df.values.sum():.0f}")

# Find strongest associations
celltype_disease_flat = []
for celltype in celltype_disease_df.index:
    for disease in celltype_disease_df.columns:
        if celltype_disease_df.loc[celltype, disease] > 0:
            celltype_disease_flat.append((celltype, disease, celltype_disease_df.loc[celltype, disease]))

celltype_disease_flat.sort(key=lambda x: x[2], reverse=True)
print("\n   Top 10 Strongest Cell Type-Disease Associations:")
for celltype, disease, count in celltype_disease_flat[:10]:
    print(f"      {celltype} - {disease}: {count:.0f} occurrences")

# Disease prevalence across tissues
print("\n3. DISEASE PREVALENCE:")
disease_tissue_counts = {d: len([t for t in all_tissues_with_disease 
                                 if tissue_disease_matrix[t].get(d, 0) > 0]) 
                        for d in all_diseases}
disease_tissue_counts = dict(sorted(disease_tissue_counts.items(), key=lambda x: x[1], reverse=True))

print("   Diseases by tissue distribution:")
for disease, tissue_count in list(disease_tissue_counts.items())[:10]:
    print(f"      {disease}: found in {tissue_count} different tissues")

# Cell type specialization for diseases
print("\n4. CELL TYPE SPECIALIZATION:")
for celltype in top_celltypes[:5]:
    diseases_for_celltype = [d for d in top_diseases if celltype_disease_matrix[celltype].get(d, 0) > 0]
    if diseases_for_celltype:
        print(f"   {celltype}:")
        print(f"      Associated with: {', '.join(diseases_for_celltype)}")

print("\n" + "="*60)
print("Analysis complete!")