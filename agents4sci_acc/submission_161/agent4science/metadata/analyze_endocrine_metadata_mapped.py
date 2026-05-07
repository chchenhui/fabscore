#!/usr/bin/env python3
"""
Analyze endocrine cell metadata with tissue mapping to understand relationships between cell types, tissues, and diseases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
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

# Create reverse mapping for tissue to category
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

# Extract and map tissues to categories
all_cell_types = []
all_tissues_raw = []
all_tissue_categories = []
all_diseases = []

tissue_category_counts = Counter()
tissue_raw_counts = Counter()

for idx, row in df.iterrows():
    all_cell_types.extend(parse_field(row['endocrine_cell_types']))
    tissues = parse_field(row['tissues'])
    all_tissues_raw.extend(tissues)
    
    # Map tissues to categories
    for tissue in tissues:
        category = map_tissue_to_category(tissue)
        all_tissue_categories.append(category)
        tissue_category_counts[category] += 1
        tissue_raw_counts[tissue] += 1
    
    all_diseases.extend(parse_field(row['diseases']))

# Count occurrences
cell_type_counts = Counter(all_cell_types)
disease_counts = Counter(all_diseases)

print(f"\n=== CELL TYPE DISTRIBUTION ===")
print(f"Unique endocrine cell types: {len(cell_type_counts)}")
print("\nTop 15 most common endocrine cell types:")
for cell_type, count in cell_type_counts.most_common(15):
    print(f"  {cell_type}: {count} datasets")

print(f"\n=== TISSUE CATEGORY DISTRIBUTION (MAPPED) ===")
print(f"Unique tissue categories: {len(tissue_category_counts)}")
print("\nTissue category distribution:")
for category, count in tissue_category_counts.most_common():
    percentage = count / sum(tissue_category_counts.values()) * 100
    print(f"  {category}: {count} occurrences ({percentage:.1f}%)")

print(f"\n=== RAW TISSUE DISTRIBUTION ===")
print(f"Unique raw tissues: {len(tissue_raw_counts)}")
print("\nTop 20 most common raw tissues:")
for tissue, count in tissue_raw_counts.most_common(20):
    category = map_tissue_to_category(tissue)
    print(f"  {tissue} [{category}]: {count} datasets")

print(f"\n=== DISEASE ASSOCIATIONS ===")
print(f"Unique disease states: {len(disease_counts)}")
print("\nDisease distribution:")
for disease, count in disease_counts.most_common():
    print(f"  {disease}: {count} datasets ({count/len(df)*100:.1f}%)")

# Analyze cell type-tissue category relationships
print(f"\n=== CELL TYPE-TISSUE CATEGORY RELATIONSHIPS ===")

cell_tissue_category_matrix = defaultdict(lambda: defaultdict(int))
for idx, row in df.iterrows():
    cell_types = parse_field(row['endocrine_cell_types'])
    tissues = parse_field(row['tissues'])
    
    for cell_type in cell_types:
        for tissue in tissues:
            category = map_tissue_to_category(tissue)
            cell_tissue_category_matrix[cell_type][category] += 1

# Find tissue category preferences for each cell type
print("\nCell type distribution across tissue categories:")
for cell_type in cell_type_counts.most_common(10):
    cell_type = cell_type[0]
    categories = cell_tissue_category_matrix[cell_type]
    if categories:
        print(f"\n{cell_type}:")
        total = sum(categories.values())
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total * 100
            print(f"  {category}: {count} occurrences ({percentage:.1f}%)")

# Analyze tissue category-disease relationships
print(f"\n=== TISSUE CATEGORY-DISEASE ASSOCIATIONS ===")

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
            tissue_category_disease_matrix[category][disease] += 1

# Find diseases by tissue category
print("\nDisease associations by tissue category:")
for category in tissue_category_counts.most_common():
    category = category[0]
    diseases = tissue_category_disease_matrix[category]
    disease_list = [d for d in diseases.keys() if d != 'normal']
    if disease_list:
        print(f"\n{category}:")
        for disease in disease_list:
            count = diseases[disease]
            print(f"  {disease}: {count} datasets")

# Cell type-disease analysis with tissue context
print(f"\n=== CELL TYPE-DISEASE RELATIONSHIPS WITH TISSUE CONTEXT ===")

cell_disease_tissue_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for idx, row in df.iterrows():
    cell_types = parse_field(row['endocrine_cell_types'])
    tissues = parse_field(row['tissues'])
    diseases = parse_field(row['diseases'])
    
    categories_in_dataset = set()
    for tissue in tissues:
        categories_in_dataset.add(map_tissue_to_category(tissue))
    
    for cell_type in cell_types:
        for disease in diseases:
            if disease != 'normal':
                for category in categories_in_dataset:
                    cell_disease_tissue_matrix[cell_type][disease][category] += 1

# Print cell type-disease associations with tissue context
print("\nTop cell type-disease associations with tissue context:")
for cell_type in ['enteroendocrine cell', 'neuroendocrine cell', 'lung neuroendocrine cell']:
    if cell_type in cell_disease_tissue_matrix:
        print(f"\n{cell_type}:")
        for disease, tissues in cell_disease_tissue_matrix[cell_type].items():
            tissue_list = [f"{t} ({c})" for t, c in tissues.items()]
            print(f"  {disease}: {', '.join(tissue_list)}")

# Dataset enrichment analysis by tissue category
print(f"\n=== ENDOCRINE ENRICHMENT BY TISSUE CATEGORY ===")

# Calculate enrichment by tissue category
tissue_category_enrichment = defaultdict(list)
for idx, row in df.iterrows():
    tissues = parse_field(row['tissues'])
    enrichment = row['endocrine_percentage']
    
    categories_in_dataset = set()
    for tissue in tissues:
        categories_in_dataset.add(map_tissue_to_category(tissue))
    
    for category in categories_in_dataset:
        tissue_category_enrichment[category].append(enrichment)

print("\nAverage endocrine enrichment by tissue category:")
for category, enrichments in sorted(tissue_category_enrichment.items()):
    avg_enrichment = np.mean(enrichments)
    median_enrichment = np.median(enrichments)
    max_enrichment = np.max(enrichments)
    print(f"  {category}:")
    print(f"    Mean: {avg_enrichment:.2f}%, Median: {median_enrichment:.2f}%, Max: {max_enrichment:.2f}%")

# Summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"Tissue categories identified: {len(tissue_category_counts)}")
print(f"Raw tissues mapped: {len(tissue_raw_counts)}")
print(f"Most common tissue category: {tissue_category_counts.most_common(1)[0][0]} ({tissue_category_counts.most_common(1)[0][1]} occurrences)")
print(f"Most diverse cell type: {max(cell_tissue_category_matrix.items(), key=lambda x: len(x[1]))[0]} (found in {len(cell_tissue_category_matrix[max(cell_tissue_category_matrix.items(), key=lambda x: len(x[1]))[0]])} categories)")

# Cross-tissue category cell types
print(f"\n=== CROSS-TISSUE CATEGORY CELL TYPES ===")
for cell_type, categories in cell_tissue_category_matrix.items():
    if len(categories) >= 5:
        print(f"\n{cell_type} (found in {len(categories)} tissue categories):")
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        print(f"  Distribution: {', '.join([f'{cat} ({count})' for cat, count in sorted_categories[:5]])}")

print("\n" + "="*60)
print("Analysis with tissue mapping complete!")