#!/usr/bin/env python3
"""
Create co-occurrence heatmaps with tissue mapping for tissue-disease and cell type-disease relationships
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

# Tissue mapping dictionary
tissue_dict = {
    "Stomach": [
        "body of stomach", "cardia of stomach", "corpus", 
        "pyloric antrum", "stomach", "mucosa of stomach"
    ],
    "Small Intestine": [
        "ileum", "small intestine", "ileal epithelium", 
        "duodenum", "intestine", "epithelium of small intestine", 
        "jejunum", "hindgut", "lamina propria of small intestine",
        "duodeno-jejunal junction", "intestinal mucosa"
    ],
    "Large Intestine": [
        "colon", "rectum", "large intestine", "sigmoid colon", 
        "transverse colon", "ascending colon", "caecum", 
        "vermiform appendix", "descending colon",
        "colonic epithelium", "lamina propria of mucosa of colon",
        "right colon", "left colon", "caecum epithelium",
        "hepatic cecum", "hepatic flexure of colon"
    ],
    "Esophagus": [
        "lower esophagus", "esophagogastric junction", 
        "submucosal esophageal gland", "esophagus"
    ],
    "Liver and Biliary System": [
        "liver", "intrahepatic bile duct", "common bile duct", 
        "gallbladder", "biliary system"
    ],
    "Pancreas": [
        "pancreas", "islet of Langerhans"
    ],
    "Lung/Respiratory": [
        "lung", "alveolar sac", "bronchus", "pleural effusion",
        "respiratory airway", "lung parenchyma", "trachea",
        "epithelium of trachea", "lower lobe of left lung",
        "upper lobe of left lung", "segmental bronchus",
        "terminal bronchus", "nasal cavity"
    ],
    "Lymphatic/Immune": [
        "mesenteric lymph node", "lymph node", "axilla",
        "spleen", "thymus"
    ],
    "Endocrine": [
        "thyroid gland", "adrenal gland"
    ],
    "Reproductive": [
        "prostate gland", "transition zone of prostate",
        "peripheral zone of prostate", "uterus",
        "upper outer quadrant of breast"
    ],
    "Nervous System": [
        "brain", "cerebellum", "spinal cord"
    ],
    "Salivary": [
        "salivary gland epithelium"
    ],
    "Genitourinary": [
        "bladder organ", "kidney"
    ],
    "Other/Unclassified": [
        "bone spine", "nasopharynx", "gut wall", "skin of body",
        "yolk sac", "mucosa", "eye"
    ]
}

# Create reverse mapping
tissue_to_category = {}
for category, tissues in tissue_dict.items():
    for tissue in tissues:
        tissue_to_category[tissue.lower()] = category

def map_tissue_to_category(tissue):
    """Map a tissue to its category"""
    tissue_lower = tissue.lower().strip()
    return tissue_to_category.get(tissue_lower, "Other/Unclassified")

# Load metadata
print("Loading endocrine datasets metadata...")
df = pd.read_csv('/scratch/rli/project/agent/data_integration/unique_datasets_metadata_final.csv')

def parse_field(field_str):
    """Parse semicolon-separated fields"""
    if pd.isna(field_str):
        return []
    return [x.strip() for x in field_str.split(';')]

print("\nBuilding co-occurrence matrices with tissue mapping...")

# 1. Tissue Category-Disease co-occurrence
tissue_category_disease_matrix = defaultdict(lambda: defaultdict(int))
for idx, row in df.iterrows():
    tissues = parse_field(row['tissues'])
    diseases = parse_field(row['diseases'])
    
    # Get unique categories for this dataset
    categories_in_dataset = set()
    for tissue in tissues:
        categories_in_dataset.add(map_tissue_to_category(tissue))
    
    for category in categories_in_dataset:
        for disease in diseases:
            if disease != 'normal':  # Exclude normal samples
                tissue_category_disease_matrix[category][disease] += 1

# 2. Cell Type-Disease co-occurrence (same as before)
celltype_disease_matrix = defaultdict(lambda: defaultdict(int))
for idx, row in df.iterrows():
    cell_types = parse_field(row['endocrine_cell_types'])
    diseases = parse_field(row['diseases'])
    
    for cell_type in cell_types:
        for disease in diseases:
            if disease != 'normal':
                celltype_disease_matrix[cell_type][disease] += 1

# 3. Cell Type-Tissue Category co-occurrence
celltype_tissue_category_matrix = defaultdict(lambda: defaultdict(int))
for idx, row in df.iterrows():
    cell_types = parse_field(row['endocrine_cell_types'])
    tissues = parse_field(row['tissues'])
    
    # Get unique categories for this dataset
    categories_in_dataset = set()
    for tissue in tissues:
        categories_in_dataset.add(map_tissue_to_category(tissue))
    
    for cell_type in cell_types:
        for category in categories_in_dataset:
            celltype_tissue_category_matrix[cell_type][category] += 1

# Get all categories and diseases
all_categories_with_disease = set(tissue_category_disease_matrix.keys())
all_diseases = set()
for category, diseases in tissue_category_disease_matrix.items():
    all_diseases.update(diseases.keys())

# Sort by totals
category_totals = {c: sum(tissue_category_disease_matrix[c].values()) for c in all_categories_with_disease}
disease_totals = {d: sum(tissue_category_disease_matrix[c].get(d, 0) for c in all_categories_with_disease) for d in all_diseases}

top_categories = sorted(category_totals.keys(), key=lambda x: category_totals[x], reverse=True)
top_diseases = sorted(disease_totals.keys(), key=lambda x: disease_totals[x], reverse=True)[:15]

# Create tissue category-disease DataFrame
tissue_category_disease_df = pd.DataFrame(0, index=top_categories, columns=top_diseases)
for category in top_categories:
    for disease in top_diseases:
        tissue_category_disease_df.loc[category, disease] = tissue_category_disease_matrix[category].get(disease, 0)

# Get cell types with disease associations
all_celltypes_with_disease = set(celltype_disease_matrix.keys())
celltype_totals = {ct: sum(celltype_disease_matrix[ct].values()) for ct in all_celltypes_with_disease}
top_celltypes = sorted(celltype_totals.keys(), key=lambda x: celltype_totals[x], reverse=True)

# Create cell type-disease DataFrame
celltype_disease_df = pd.DataFrame(0, index=top_celltypes, columns=top_diseases)
for cell_type in top_celltypes:
    for disease in top_diseases:
        celltype_disease_df.loc[cell_type, disease] = celltype_disease_matrix[cell_type].get(disease, 0)

# Get top cell types and tissue categories for cell type-tissue matrix
top_10_celltypes = list(celltype_totals.keys())[:10]
all_tissue_categories = set()
for ct in top_10_celltypes:
    all_tissue_categories.update(celltype_tissue_category_matrix[ct].keys())
all_tissue_categories = sorted(all_tissue_categories)

# Create cell type-tissue category DataFrame
celltype_tissue_df = pd.DataFrame(0, index=top_10_celltypes, columns=all_tissue_categories)
for cell_type in top_10_celltypes:
    for category in all_tissue_categories:
        celltype_tissue_df.loc[cell_type, category] = celltype_tissue_category_matrix[cell_type].get(category, 0)

# Create comprehensive figure with mapped tissue categories
fig = plt.figure(figsize=(26, 22))

# ===== TISSUE CATEGORY-DISEASE HEATMAP =====
ax1 = plt.subplot(3, 2, 1)
sns.heatmap(tissue_category_disease_df, annot=True, fmt='d', cmap='YlOrRd', 
            cbar_kws={'label': 'Co-occurrence Count'}, ax=ax1,
            linewidths=0.5, linecolor='gray')
ax1.set_title('Tissue Category-Disease Co-occurrence Matrix', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Disease', fontsize=12)
ax1.set_ylabel('Tissue Category', fontsize=12)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=10)

# ===== CELL TYPE-DISEASE HEATMAP =====
ax2 = plt.subplot(3, 2, 2)
sns.heatmap(celltype_disease_df, annot=True, fmt='d', cmap='BuPu', 
            cbar_kws={'label': 'Co-occurrence Count'}, ax=ax2,
            linewidths=0.5, linecolor='gray')
ax2.set_title('Cell Type-Disease Co-occurrence Matrix', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Disease', fontsize=12)
ax2.set_ylabel('Cell Type', fontsize=12)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=9)

# ===== NORMALIZED TISSUE CATEGORY-DISEASE HEATMAP =====
ax3 = plt.subplot(3, 2, 3)
tissue_category_disease_norm = tissue_category_disease_df.div(tissue_category_disease_df.sum(axis=1), axis=0) * 100
tissue_category_disease_norm = tissue_category_disease_norm.fillna(0)
sns.heatmap(tissue_category_disease_norm, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Percentage (%)'}, ax=ax3,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=50)
ax3.set_title('Tissue Category-Disease Distribution (% per Category)', fontsize=14, fontweight='bold', pad=20)
ax3.set_xlabel('Disease', fontsize=12)
ax3.set_ylabel('Tissue Category', fontsize=12)
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=10)

# ===== NORMALIZED CELL TYPE-DISEASE HEATMAP =====
ax4 = plt.subplot(3, 2, 4)
celltype_disease_norm = celltype_disease_df.div(celltype_disease_df.sum(axis=1), axis=0) * 100
celltype_disease_norm = celltype_disease_norm.fillna(0)
sns.heatmap(celltype_disease_norm, annot=True, fmt='.1f', cmap='PuBuGn', 
            cbar_kws={'label': 'Percentage (%)'}, ax=ax4,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=50)
ax4.set_title('Cell Type-Disease Distribution (% per Cell Type)', fontsize=14, fontweight='bold', pad=20)
ax4.set_xlabel('Disease', fontsize=12)
ax4.set_ylabel('Cell Type', fontsize=12)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax4.get_yticklabels(), rotation=0, fontsize=9)

# ===== CELL TYPE-TISSUE CATEGORY HEATMAP =====
ax5 = plt.subplot(3, 2, 5)
sns.heatmap(celltype_tissue_df, annot=True, fmt='d', cmap='viridis', 
            cbar_kws={'label': 'Co-occurrence Count'}, ax=ax5,
            linewidths=0.5, linecolor='gray')
ax5.set_title('Cell Type-Tissue Category Co-occurrence Matrix', fontsize=14, fontweight='bold', pad=20)
ax5.set_xlabel('Tissue Category', fontsize=12)
ax5.set_ylabel('Cell Type', fontsize=12)
plt.setp(ax5.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax5.get_yticklabels(), rotation=0, fontsize=9)

# ===== NORMALIZED CELL TYPE-TISSUE CATEGORY HEATMAP =====
ax6 = plt.subplot(3, 2, 6)
celltype_tissue_norm = celltype_tissue_df.div(celltype_tissue_df.sum(axis=1), axis=0) * 100
celltype_tissue_norm = celltype_tissue_norm.fillna(0)
sns.heatmap(celltype_tissue_norm, annot=True, fmt='.1f', cmap='coolwarm', center=0,
            cbar_kws={'label': 'Percentage (%)'}, ax=ax6,
            linewidths=0.5, linecolor='gray')
ax6.set_title('Cell Type-Tissue Category Distribution (% per Cell Type)', fontsize=14, fontweight='bold', pad=20)
ax6.set_xlabel('Tissue Category', fontsize=12)
ax6.set_ylabel('Cell Type', fontsize=12)
plt.setp(ax6.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax6.get_yticklabels(), rotation=0, fontsize=9)

plt.suptitle('Mapped Tissue Category Co-occurrence Analysis in Endocrine Cell Datasets', 
             fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('mapped_cooccurrence_heatmaps.pdf', dpi=300, bbox_inches='tight')
plt.savefig('mapped_cooccurrence_heatmaps.png', dpi=150, bbox_inches='tight')
print("\nSaved: mapped_cooccurrence_heatmaps.pdf/png")

# Create focused analysis for GI vs Respiratory systems
fig2, axes = plt.subplots(2, 2, figsize=(18, 14))

# GI System Analysis (combining Stomach, Small Intestine, Large Intestine, Esophagus)
gi_categories = ['Stomach', 'Small Intestine', 'Large Intestine', 'Esophagus']
respiratory_categories = ['Lung/Respiratory']

# GI-focused disease heatmap
ax = axes[0, 0]
gi_disease_df = tissue_category_disease_df.loc[tissue_category_disease_df.index.isin(gi_categories)]
if not gi_disease_df.empty:
    gi_disease_df = gi_disease_df.loc[:, gi_disease_df.sum() > 0]
    sns.heatmap(gi_disease_df, annot=True, fmt='d', cmap='Oranges', ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('GI System-Disease Co-occurrence', fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

# Respiratory-focused disease heatmap
ax = axes[0, 1]
resp_disease_df = tissue_category_disease_df.loc[tissue_category_disease_df.index.isin(respiratory_categories)]
if not resp_disease_df.empty:
    resp_disease_df = resp_disease_df.loc[:, resp_disease_df.sum() > 0]
    sns.heatmap(resp_disease_df, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('Respiratory System-Disease Co-occurrence', fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

# Cell type distribution in GI vs Respiratory
ax = axes[1, 0]
gi_celltype_data = []
for cell_type in top_10_celltypes:
    gi_total = sum([celltype_tissue_category_matrix[cell_type].get(cat, 0) for cat in gi_categories])
    gi_celltype_data.append(gi_total)

bars = ax.barh(top_10_celltypes, gi_celltype_data, color='coral')
ax.set_xlabel('Occurrence Count')
ax.set_title('Cell Types in GI System', fontsize=12, fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(gi_celltype_data):
    ax.text(v + 0.1, i, str(v), va='center')

ax = axes[1, 1]
resp_celltype_data = []
for cell_type in top_10_celltypes:
    resp_total = celltype_tissue_category_matrix[cell_type].get('Lung/Respiratory', 0)
    resp_celltype_data.append(resp_total)

bars = ax.barh(top_10_celltypes, resp_celltype_data, color='skyblue')
ax.set_xlabel('Occurrence Count')
ax.set_title('Cell Types in Respiratory System', fontsize=12, fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(resp_celltype_data):
    ax.text(v + 0.1, i, str(v), va='center')

plt.suptitle('GI vs Respiratory System Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gi_respiratory_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('gi_respiratory_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: gi_respiratory_comparison.pdf/png")

# Generate summary statistics
print("\n" + "="*60)
print("STATISTICAL SUMMARY WITH TISSUE MAPPING")
print("="*60)

print("\n1. TISSUE CATEGORY-DISEASE ASSOCIATIONS:")
print(f"   Total tissue categories with disease data: {len(all_categories_with_disease)}")
print(f"   Total unique diseases (excluding normal): {len(all_diseases)}")
print(f"   Total tissue category-disease pairs: {tissue_category_disease_df.values.sum():.0f}")

# Find strongest associations
category_disease_flat = []
for category in tissue_category_disease_df.index:
    for disease in tissue_category_disease_df.columns:
        if tissue_category_disease_df.loc[category, disease] > 0:
            category_disease_flat.append((category, disease, tissue_category_disease_df.loc[category, disease]))

category_disease_flat.sort(key=lambda x: x[2], reverse=True)
print("\n   Top 10 Strongest Tissue Category-Disease Associations:")
for category, disease, count in category_disease_flat[:10]:
    print(f"      {category} - {disease}: {count:.0f} occurrences")

print("\n2. TISSUE CATEGORY DISTRIBUTION:")
for category in top_categories:
    total = sum(tissue_category_disease_matrix[category].values())
    normal = tissue_category_disease_matrix[category].get('normal', 0)
    diseased = total  # Already excluded normal in matrix building
    print(f"   {category}: {diseased} disease occurrences")

print("\n3. CELL TYPE TISSUE PREFERENCES:")
print("   Cell types with strongest tissue category preferences:")
for cell_type in top_10_celltypes[:5]:
    categories = celltype_tissue_category_matrix[cell_type]
    if categories:
        top_cat = max(categories.items(), key=lambda x: x[1])
        total = sum(categories.values())
        percentage = top_cat[1] / total * 100
        print(f"      {cell_type}: {top_cat[0]} ({percentage:.1f}% of occurrences)")

print("\n4. GI vs RESPIRATORY SYSTEM COMPARISON:")
gi_total = sum([tissue_category_disease_df.loc[cat].sum() for cat in gi_categories if cat in tissue_category_disease_df.index])
resp_total = tissue_category_disease_df.loc['Lung/Respiratory'].sum() if 'Lung/Respiratory' in tissue_category_disease_df.index else 0
print(f"   GI System disease occurrences: {gi_total:.0f}")
print(f"   Respiratory System disease occurrences: {resp_total:.0f}")
print(f"   GI/Respiratory ratio: {gi_total/max(resp_total, 1):.2f}x")

print("\n" + "="*60)
print("Analysis with tissue mapping complete!")