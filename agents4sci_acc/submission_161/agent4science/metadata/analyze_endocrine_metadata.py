#!/usr/bin/env python3
"""
Analyze endocrine cell metadata to understand relationships between cell types, tissues, and diseases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load metadata
print("Loading endocrine datasets metadata...")
df = pd.read_csv('/scratch/rli/project/agent/data_integration/unique_datasets_metadata_final.csv')

print(f"\nDataset Overview:")
print(f"Total datasets: {len(df)}")
print(f"Total endocrine cells: {df['endocrine_cell_count'].sum():,}")
print(f"Total cells across all datasets: {df['dataset_total_cell_count'].sum():,}")
print(f"Average endocrine percentage: {df['endocrine_percentage'].mean():.2f}%")

# Parse cell types, tissues, and diseases
def parse_field(field_str):
    """Parse semicolon-separated fields"""
    if pd.isna(field_str):
        return []
    return [x.strip() for x in field_str.split(';')]

# Extract all unique values
all_cell_types = []
all_tissues = []
all_diseases = []

for idx, row in df.iterrows():
    all_cell_types.extend(parse_field(row['endocrine_cell_types']))
    all_tissues.extend(parse_field(row['tissues']))
    all_diseases.extend(parse_field(row['diseases']))

# Count occurrences
cell_type_counts = Counter(all_cell_types)
tissue_counts = Counter(all_tissues)
disease_counts = Counter(all_diseases)

print(f"\n=== CELL TYPE DISTRIBUTION ===")
print(f"Unique endocrine cell types: {len(cell_type_counts)}")
print("\nTop 15 most common endocrine cell types:")
for cell_type, count in cell_type_counts.most_common(15):
    print(f"  {cell_type}: {count} datasets")

print(f"\n=== TISSUE DISTRIBUTION ===")
print(f"Unique tissues: {len(tissue_counts)}")
print("\nTop 20 most common tissues:")
for tissue, count in tissue_counts.most_common(20):
    print(f"  {tissue}: {count} datasets")

print(f"\n=== DISEASE ASSOCIATIONS ===")
print(f"Unique disease states: {len(disease_counts)}")
print("\nDisease distribution:")
for disease, count in disease_counts.most_common():
    print(f"  {disease}: {count} datasets ({count/len(df)*100:.1f}%)")

# Analyze cell type-tissue relationships
print(f"\n=== CELL TYPE-TISSUE RELATIONSHIPS ===")

# Create a matrix of cell type vs tissue
cell_tissue_matrix = {}
for idx, row in df.iterrows():
    cell_types = parse_field(row['endocrine_cell_types'])
    tissues = parse_field(row['tissues'])
    
    for cell_type in cell_types:
        if cell_type not in cell_tissue_matrix:
            cell_tissue_matrix[cell_type] = {}
        for tissue in tissues:
            if tissue not in cell_tissue_matrix[cell_type]:
                cell_tissue_matrix[cell_type][tissue] = 0
            cell_tissue_matrix[cell_type][tissue] += 1

# Find tissue-specific vs ubiquitous cell types
print("\nTissue specificity of endocrine cell types:")
for cell_type, tissues in cell_tissue_matrix.items():
    if len(tissues) == 1:
        print(f"  {cell_type}: TISSUE-SPECIFIC to {list(tissues.keys())[0]}")
    elif len(tissues) >= 10:
        print(f"  {cell_type}: UBIQUITOUS (found in {len(tissues)} tissues)")

# Analyze cell type-disease relationships
print(f"\n=== CELL TYPE-DISEASE RELATIONSHIPS ===")

cell_disease_matrix = {}
for idx, row in df.iterrows():
    cell_types = parse_field(row['endocrine_cell_types'])
    diseases = parse_field(row['diseases'])
    
    for cell_type in cell_types:
        if cell_type not in cell_disease_matrix:
            cell_disease_matrix[cell_type] = {}
        for disease in diseases:
            if disease not in cell_disease_matrix[cell_type]:
                cell_disease_matrix[cell_type][disease] = 0
            cell_disease_matrix[cell_type][disease] += 1

# Find disease-associated cell types
print("\nCell types found in disease contexts:")
for cell_type, diseases in cell_disease_matrix.items():
    disease_list = [d for d in diseases.keys() if d != 'normal']
    if disease_list:
        print(f"  {cell_type}: {', '.join(disease_list)}")

# Tissue-disease associations
print(f"\n=== TISSUE-DISEASE ASSOCIATIONS ===")

tissue_disease_matrix = {}
for idx, row in df.iterrows():
    tissues = parse_field(row['tissues'])
    diseases = parse_field(row['diseases'])
    
    for tissue in tissues:
        if tissue not in tissue_disease_matrix:
            tissue_disease_matrix[tissue] = {}
        for disease in diseases:
            if disease not in tissue_disease_matrix[tissue]:
                tissue_disease_matrix[tissue][disease] = 0
            tissue_disease_matrix[tissue][disease] += 1

# Find tissues with most disease studies
tissues_with_diseases = {}
for tissue, diseases in tissue_disease_matrix.items():
    disease_list = [d for d in diseases.keys() if d != 'normal']
    if disease_list:
        tissues_with_diseases[tissue] = disease_list

print(f"\nTissues studied in disease contexts ({len(tissues_with_diseases)} total):")
for tissue, diseases in sorted(tissues_with_diseases.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
    print(f"  {tissue}: {', '.join(diseases)}")

# Analyze dataset size distribution
print(f"\n=== DATASET SIZE ANALYSIS ===")
print(f"Smallest dataset: {df['dataset_total_cell_count'].min():,} cells")
print(f"Largest dataset: {df['dataset_total_cell_count'].max():,} cells")
print(f"Median dataset size: {df['dataset_total_cell_count'].median():,.0f} cells")
print(f"Mean dataset size: {df['dataset_total_cell_count'].mean():,.0f} cells")

# Endocrine enrichment analysis
print(f"\n=== ENDOCRINE ENRICHMENT ANALYSIS ===")
high_enrichment = df[df['endocrine_percentage'] > 5]
print(f"Datasets with >5% endocrine cells: {len(high_enrichment)} ({len(high_enrichment)/len(df)*100:.1f}%)")
print("\nTop 10 datasets by endocrine percentage:")
for idx, row in df.nlargest(10, 'endocrine_percentage').iterrows():
    print(f"  {row['dataset_title'][:60]}...")
    print(f"    {row['endocrine_percentage']:.2f}% endocrine ({row['endocrine_cell_count']:,}/{row['dataset_total_cell_count']:,} cells)")
    print(f"    Cell types: {row['endocrine_cell_types']}")
    print(f"    Tissues: {row['tissues']}")

# Assay technology distribution
print(f"\n=== ASSAY TECHNOLOGY DISTRIBUTION ===")
all_assays = []
for assays_str in df['assays']:
    all_assays.extend(parse_field(assays_str))

assay_counts = Counter(all_assays)
print(f"Unique assay technologies: {len(assay_counts)}")
print("\nTop 10 most common assays:")
for assay, count in assay_counts.most_common(10):
    print(f"  {assay}: {count} datasets")

# Organ system categorization
print(f"\n=== ORGAN SYSTEM CATEGORIZATION ===")

organ_systems = {
    'Gastrointestinal': ['stomach', 'intestine', 'colon', 'duodenum', 'ileum', 'jejunum', 
                         'rectum', 'caecum', 'appendix', 'esophagus', 'gut'],
    'Respiratory': ['lung', 'trachea', 'bronchus', 'alveolar', 'respiratory', 'nasal'],
    'Endocrine': ['pancreas', 'islet', 'adrenal', 'thyroid'],
    'Genitourinary': ['prostate', 'bladder', 'kidney', 'urethra', 'uterus'],
    'Hepatobiliary': ['liver', 'biliary'],
    'Nervous': ['brain', 'cerebellum', 'spinal cord'],
    'Cardiovascular': ['heart', 'aorta', 'artery', 'vein'],
    'Immune': ['lymph node', 'spleen', 'thymus'],
    'Other': ['breast', 'skin', 'eye', 'yolk sac', 'pleural']
}

# Categorize tissues
tissue_to_system = {}
for system, keywords in organ_systems.items():
    for tissue in tissue_counts.keys():
        tissue_lower = tissue.lower()
        for keyword in keywords:
            if keyword in tissue_lower:
                tissue_to_system[tissue] = system
                break

# Count by organ system
system_counts = Counter()
for tissue, count in tissue_counts.items():
    system = tissue_to_system.get(tissue, 'Other')
    system_counts[system] += count

print("\nEndocrine cells by organ system:")
for system, count in system_counts.most_common():
    print(f"  {system}: {count} dataset occurrences")

# Cross-tissue endocrine cell types
print(f"\n=== CROSS-TISSUE ENDOCRINE CELL TYPES ===")
for cell_type, tissues in cell_tissue_matrix.items():
    if len(tissues) >= 5:
        top_tissues = sorted(tissues.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n{cell_type} (found in {len(tissues)} tissues):")
        print(f"  Top tissues: {', '.join([f'{t[0]} ({t[1]}x)' for t in top_tissues])}")

# Disease-specific insights
print(f"\n=== DISEASE-SPECIFIC INSIGHTS ===")

# Group diseases into categories
disease_categories = {
    'Cancer': ['carcinoma', 'adenocarcinoma', 'cancer', 'adenoma', 'polyp', 'neoplasm'],
    'Inflammatory': ['Crohn', 'colitis', 'gastritis', 'inflammatory'],
    'Metabolic': ['diabetes', 'obesity'],
    'Metaplasia': ['metaplasia', 'Barrett'],
    'Fibrotic': ['fibrosis', 'fibrotic'],
    'Infectious': ['COVID-19'],
    'Other': ['hyperplasia', 'disorder', 'disease']
}

disease_category_counts = Counter()
for disease in disease_counts.keys():
    if disease == 'normal':
        continue
    categorized = False
    for category, keywords in disease_categories.items():
        for keyword in keywords:
            if keyword.lower() in disease.lower():
                disease_category_counts[category] += disease_counts[disease]
                categorized = True
                break
        if categorized:
            break
    if not categorized:
        disease_category_counts['Other'] += disease_counts[disease]

print("\nDisease categories:")
for category, count in disease_category_counts.most_common():
    print(f"  {category}: {count} dataset occurrences")

print("\n" + "="*60)
print("Analysis complete!")