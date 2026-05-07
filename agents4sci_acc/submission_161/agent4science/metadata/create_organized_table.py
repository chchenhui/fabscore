#!/usr/bin/env python3
"""
Create an organized table from unique metadata with key columns:
- Tissue
- Study 
- Number of cells
- Author (from collection name)
- Percent of endocrine
- In vitro/In vivo
"""

import pandas as pd
import numpy as np
import re

# Define tissue mapping dictionary
tissue_mapping = {
    # Stomach
    'stomach': 'Stomach',
    'gastric': 'Stomach',
    'pyloric antrum': 'Stomach',
    'pyloric canal': 'Stomach',
    'fundus of stomach': 'Stomach',
    'body of stomach': 'Stomach',
    'cardia of stomach': 'Stomach',
    
    # Small Intestine
    'small intestine': 'Small Intestine',
    'duodenum': 'Small Intestine',
    'jejunum': 'Small Intestine',
    'ileum': 'Small Intestine',
    'intestine': 'Small Intestine',
    'small bowel': 'Small Intestine',
    'terminal ileum': 'Small Intestine',
    'proximal ileum': 'Small Intestine',
    'distal ileum': 'Small Intestine',
    
    # Large Intestine
    'colon': 'Large Intestine',
    'large intestine': 'Large Intestine',
    'rectum': 'Large Intestine',
    'caecum': 'Large Intestine',
    'cecum': 'Large Intestine',
    'sigmoid colon': 'Large Intestine',
    'ascending colon': 'Large Intestine',
    'descending colon': 'Large Intestine',
    'transverse colon': 'Large Intestine',
    'rectosigmoid': 'Large Intestine',
    'rectosigmoid colon': 'Large Intestine',
    'colonic epithelium': 'Large Intestine',
    'hepatic flexure of colon': 'Large Intestine',
    'splenic flexure of colon': 'Large Intestine',
    'hepatic cecum': 'Large Intestine',
    'vermiform appendix': 'Large Intestine',
    
    # Esophagus
    'esophagus': 'Esophagus',
    'oesophagus': 'Esophagus',
    'esophagogastric junction': 'Esophagus',
    'lower esophagus': 'Esophagus',
    'submucosal esophageal gland': 'Esophagus',
    
    # Liver and Biliary
    'liver': 'Liver and Biliary System',
    'hepatic': 'Liver and Biliary System',
    'biliary': 'Liver and Biliary System',
    'biliary system': 'Liver and Biliary System',
    'gallbladder': 'Liver and Biliary System',
    'bile duct': 'Liver and Biliary System',
    
    # Pancreas
    'pancreas': 'Pancreas',
    'pancreatic': 'Pancreas',
    'islet of langerhans': 'Pancreas',
    
    # Lung/Respiratory
    'lung': 'Lung/Respiratory',
    'pulmonary': 'Lung/Respiratory',
    'respiratory': 'Lung/Respiratory',
    'trachea': 'Lung/Respiratory',
    'bronchus': 'Lung/Respiratory',
    'bronchi': 'Lung/Respiratory',
    'alveolar': 'Lung/Respiratory',
    'alveolar sac': 'Lung/Respiratory',
    'lung parenchyma': 'Lung/Respiratory',
    'respiratory airway': 'Lung/Respiratory',
    'segmental bronchus': 'Lung/Respiratory',
    'nasal': 'Lung/Respiratory',
    'nasopharynx': 'Lung/Respiratory',
    
    # Lymphatic/Immune
    'lymph node': 'Lymphatic/Immune',
    'lymphatic': 'Lymphatic/Immune',
    'spleen': 'Lymphatic/Immune',
    'thymus': 'Lymphatic/Immune',
    'mesenteric lymph node': 'Lymphatic/Immune',
    
    # Endocrine
    'thyroid': 'Endocrine',
    'adrenal': 'Endocrine',
    'adrenal gland': 'Endocrine',
    'pituitary': 'Endocrine',
    
    # Reproductive
    'prostate': 'Reproductive',
    'uterus': 'Reproductive',
    'ovary': 'Reproductive',
    'testis': 'Reproductive',
    'fallopian tube': 'Reproductive',
    'endometrium': 'Reproductive',
    
    # Nervous System
    'brain': 'Nervous System',
    'cerebellum': 'Nervous System',
    'spinal cord': 'Nervous System',
    'nerve': 'Nervous System',
    
    # Genitourinary
    'kidney': 'Genitourinary',
    'bladder': 'Genitourinary',
    'bladder organ': 'Genitourinary',
    'ureter': 'Genitourinary',
    'urethra': 'Genitourinary',
    
    # Other
    'breast': 'Other/Unclassified',
    'skin': 'Other/Unclassified',
    'eye': 'Other/Unclassified',
    'yolk sac': 'Other/Unclassified',
    'pleural': 'Other/Unclassified',
    'pleural effusion': 'Other/Unclassified',
    'heart': 'Other/Unclassified'
}

def map_tissue(tissue_str):
    """Map raw tissue to category"""
    if pd.isna(tissue_str):
        return 'Not specified'
    
    tissue_lower = tissue_str.lower().strip()
    
    # Direct mapping
    if tissue_lower in tissue_mapping:
        return tissue_mapping[tissue_lower]
    
    # Partial matching for complex tissue names
    for key, value in tissue_mapping.items():
        if key in tissue_lower:
            return value
    
    return 'Other/Unclassified'

# Load metadata
print("Loading unique datasets metadata...")
df = pd.read_csv('/scratch/rli/project/agent/data_integration/unique_datasets_metadata_final.csv')

# Create organized table
organized_data = []

for idx, row in df.iterrows():
    # Extract study info
    study_title = row['dataset_title']
    collection_name = row['collection_name']
    
    # Extract author from collection name (usually first author or consortium)
    # Pattern: look for text before first parenthesis or comma
    author_match = re.search(r'^([^(,]+)', collection_name)
    if author_match:
        author = author_match.group(1).strip()
        # Clean up common patterns
        author = author.replace('Integrated ', '').replace('human ', '').replace('Human ', '')
        # If it's too long, take first few words
        author_words = author.split()
        if len(author_words) > 3:
            author = ' '.join(author_words[:3]) + ' et al.'
    else:
        author = collection_name[:30] + '...' if len(collection_name) > 30 else collection_name
    
    # Process tissues - map to categories and keep raw tissues
    tissues = row['tissues']
    raw_tissues = []
    mapped_categories = set()
    
    if pd.notna(tissues):
        tissue_list = [t.strip() for t in tissues.split(';')]
        
        # Map each tissue to category
        for tissue in tissue_list:
            category = map_tissue(tissue)
            mapped_categories.add(category)
            raw_tissues.append(tissue)
        
        # Create display strings
        # Take up to 3 raw tissues for display
        raw_tissue_display = '; '.join(raw_tissues[:3])
        if len(raw_tissues) > 3:
            raw_tissue_display += f' (+{len(raw_tissues)-3} more)'
        
        # Show mapped categories
        mapped_display = '; '.join(sorted(mapped_categories))
    else:
        raw_tissue_display = 'Not specified'
        mapped_display = 'Not specified'
    
    # Determine in vitro/in vivo based on title and tissue
    in_vitro_keywords = ['organoid', 'cell line', 'culture', 'spheroid', 'in vitro', 'iPSC', 'ESC']
    in_vivo_keywords = ['biopsy', 'resection', 'tissue', 'patient', 'donor', 'healthy', 'primary']
    
    # Check title and tissues for keywords
    combined_text = (study_title + ' ' + tissues if pd.notna(tissues) else study_title).lower()
    
    if any(keyword.lower() in combined_text for keyword in in_vitro_keywords):
        sample_type = 'In vitro'
    elif any(keyword.lower() in combined_text for keyword in in_vivo_keywords):
        sample_type = 'In vivo'
    else:
        # Default based on common patterns
        if 'organoid' in collection_name.lower():
            sample_type = 'In vitro'
        else:
            sample_type = 'In vivo'
    
    # Create row for table
    organized_data.append({
        'Dataset_ID': row['dataset_id'],
        'Tissue_Category': mapped_display,
        'Raw_Tissue': raw_tissue_display,
        'Study': study_title[:60] + '...' if len(study_title) > 60 else study_title,
        'Num_Cells': f"{row['dataset_total_cell_count']:,}",
        'Num_Endocrine': f"{row['endocrine_cell_count']:,}",
        'Author/Collection': author,
        'Percent_Endocrine': f"{row['endocrine_percentage']:.2f}%",
        'Sample_Type': sample_type,
        'Disease': row['diseases'] if pd.notna(row['diseases']) else 'normal',
        'Technology': row['assays'].split(';')[0].strip() if pd.notna(row['assays']) else 'Not specified'
    })

# Create DataFrame
organized_df = pd.DataFrame(organized_data)

# Sort by percent endocrine (descending)
organized_df['sort_key'] = organized_df['Percent_Endocrine'].str.rstrip('%').astype(float)
organized_df = organized_df.sort_values('sort_key', ascending=False)
organized_df = organized_df.drop('sort_key', axis=1)

# Save as CSV
csv_path = '/scratch/rli/project/agent/metadata/endocrine_datasets_organized_table.csv'
organized_df.to_csv(csv_path, index=False)
print(f"\nSaved CSV table to: {csv_path}")

# Create markdown version
markdown_path = '/scratch/rli/project/agent/metadata/endocrine_datasets_organized_table.md'
with open(markdown_path, 'w') as f:
    f.write("# Endocrine Cell Datasets - Organized Summary\n\n")
    f.write(f"**Total Datasets**: {len(organized_df)}\n")
    f.write(f"**Total Cells**: {df['dataset_total_cell_count'].sum():,}\n")
    f.write(f"**Total Endocrine Cells**: {df['endocrine_cell_count'].sum():,}\n")
    f.write(f"**Average Endocrine Percentage**: {df['endocrine_percentage'].mean():.2f}%\n\n")
    
    # Summary statistics
    f.write("## Summary Statistics\n\n")
    f.write(f"- **In vitro samples**: {(organized_df['Sample_Type'] == 'In vitro').sum()}\n")
    f.write(f"- **In vivo samples**: {(organized_df['Sample_Type'] == 'In vivo').sum()}\n")
    f.write(f"- **Highest enrichment**: {organized_df.iloc[0]['Percent_Endocrine']} ({organized_df.iloc[0]['Study'][:40]}...)\n")
    f.write(f"- **Largest dataset**: {organized_df.loc[organized_df['Num_Cells'].str.replace(',','').astype(int).idxmax(), 'Num_Cells']} cells\n\n")
    
    # Main table
    f.write("## Full Dataset Table\n\n")
    f.write("| Dataset ID | Tissue Category | Raw Tissue | Study | Num Cells | Num Endocrine | Author/Collection | % Endocrine | Sample Type | Disease | Technology |\n")
    f.write("|------------|-----------------|------------|-------|-----------|---------------|-------------------|-------------|-------------|---------|------------|\n")
    
    for idx, row in organized_df.iterrows():
        # Escape pipe characters in text fields
        dataset_id = row['Dataset_ID'][:8] + '...' if len(row['Dataset_ID']) > 8 else row['Dataset_ID']
        tissue_cat = row['Tissue_Category'].replace('|', '\\|')
        raw_tissue = row['Raw_Tissue'].replace('|', '\\|')
        study = row['Study'].replace('|', '\\|')
        author = row['Author/Collection'].replace('|', '\\|')
        disease = row['Disease'].replace('|', '\\|')
        
        f.write(f"| {dataset_id} | {tissue_cat} | {raw_tissue} | {study} | {row['Num_Cells']} | {row['Num_Endocrine']} | "
                f"{author} | {row['Percent_Endocrine']} | {row['Sample_Type']} | {disease} | {row['Technology']} |\n")
    
    f.write("\n---\n")
    f.write(f"*Generated: 2025-09-08*\n")

print(f"Saved Markdown table to: {markdown_path}")

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total datasets: {len(organized_df)}")
print(f"In vitro samples: {(organized_df['Sample_Type'] == 'In vitro').sum()}")
print(f"In vivo samples: {(organized_df['Sample_Type'] == 'In vivo').sum()}")
print(f"\nTop 5 datasets by endocrine enrichment:")
for i in range(min(5, len(organized_df))):
    row = organized_df.iloc[i]
    print(f"  {i+1}. {row['Percent_Endocrine']} - {row['Study'][:50]}...")
    print(f"     Tissue Category: {row['Tissue_Category']}")
    print(f"     Raw Tissue: {row['Raw_Tissue'][:50]}")
    print(f"     Type: {row['Sample_Type']}, Cells: {row['Num_Cells']}")

print("\n" + "="*60)
print("Table generation complete!")