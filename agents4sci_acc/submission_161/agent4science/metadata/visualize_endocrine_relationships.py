#!/usr/bin/env python3
"""
Create comprehensive visualizations of endocrine cell metadata relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load metadata
df = pd.read_csv('/scratch/rli/project/agent/data_integration/unique_datasets_metadata_final.csv')

def parse_field(field_str):
    """Parse semicolon-separated fields"""
    if pd.isna(field_str):
        return []
    return [x.strip() for x in field_str.split(';')]

print("Creating visualizations...")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# 1. Cell Type Distribution
ax1 = plt.subplot(3, 3, 1)
cell_type_counts = Counter()
for cell_types in df['endocrine_cell_types']:
    cell_type_counts.update(parse_field(cell_types))

top_cell_types = dict(cell_type_counts.most_common(10))
bars = ax1.barh(list(top_cell_types.keys()), list(top_cell_types.values()), color='steelblue')
ax1.set_xlabel('Number of Datasets')
ax1.set_title('Top 10 Endocrine Cell Types', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
for i, (k, v) in enumerate(top_cell_types.items()):
    ax1.text(v + 0.5, i, str(v), va='center')

# 2. Tissue Distribution
ax2 = plt.subplot(3, 3, 2)
tissue_counts = Counter()
for tissues in df['tissues']:
    tissue_counts.update(parse_field(tissues))

top_tissues = dict(tissue_counts.most_common(15))
bars = ax2.barh(list(top_tissues.keys()), list(top_tissues.values()), color='coral')
ax2.set_xlabel('Number of Datasets')
ax2.set_title('Top 15 Tissues with Endocrine Cells', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
for i, (k, v) in enumerate(top_tissues.items()):
    ax2.text(v + 0.5, i, str(v), va='center')

# 3. Disease Distribution
ax3 = plt.subplot(3, 3, 3)
disease_counts = Counter()
for diseases in df['diseases']:
    disease_counts.update(parse_field(diseases))

# Remove 'normal' for disease visualization
disease_counts_filtered = {k: v for k, v in disease_counts.items() if k != 'normal'}
if disease_counts_filtered:
    disease_df = pd.DataFrame(list(disease_counts_filtered.items()), columns=['Disease', 'Count'])
    disease_df = disease_df.sort_values('Count', ascending=False).head(10)
    bars = ax3.barh(disease_df['Disease'], disease_df['Count'], color='salmon')
    ax3.set_xlabel('Number of Datasets')
    ax3.set_title('Top 10 Disease Associations', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    for i, row in enumerate(disease_df.itertuples()):
        ax3.text(row.Count + 0.1, i, str(row.Count), va='center')

# 4. Endocrine Percentage Distribution
ax4 = plt.subplot(3, 3, 4)
ax4.hist(df['endocrine_percentage'], bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Endocrine Percentage (%)')
ax4.set_ylabel('Number of Datasets')
ax4.set_title('Distribution of Endocrine Cell Percentages', fontsize=12, fontweight='bold')
ax4.axvline(df['endocrine_percentage'].median(), color='red', linestyle='--', 
            label=f'Median: {df["endocrine_percentage"].median():.2f}%')
ax4.legend()

# 5. Dataset Size Distribution
ax5 = plt.subplot(3, 3, 5)
sizes = df['dataset_total_cell_count'] / 1000  # Convert to thousands
ax5.hist(np.log10(sizes + 1), bins=30, color='darkorange', edgecolor='black', alpha=0.7)
ax5.set_xlabel('Log10(Cell Count in Thousands)')
ax5.set_ylabel('Number of Datasets')
ax5.set_title('Distribution of Dataset Sizes', fontsize=12, fontweight='bold')

# 6. Organ System Distribution
ax6 = plt.subplot(3, 3, 6)
organ_systems = {
    'GI': ['stomach', 'intestine', 'colon', 'duodenum', 'ileum', 'jejunum', 
           'rectum', 'caecum', 'appendix', 'esophagus', 'gut'],
    'Respiratory': ['lung', 'trachea', 'bronchus', 'alveolar', 'respiratory', 'nasal'],
    'Endocrine': ['pancreas', 'islet', 'adrenal', 'thyroid'],
    'Genitourinary': ['prostate', 'bladder', 'kidney', 'urethra', 'uterus'],
    'Hepatobiliary': ['liver', 'biliary'],
    'Other': []
}

system_counts = Counter()
for tissues in df['tissues']:
    for tissue in parse_field(tissues):
        tissue_lower = tissue.lower()
        assigned = False
        for system, keywords in organ_systems.items():
            if system == 'Other':
                continue
            for keyword in keywords:
                if keyword in tissue_lower:
                    system_counts[system] += 1
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            system_counts['Other'] += 1

system_df = pd.DataFrame(list(system_counts.items()), columns=['System', 'Count'])
system_df = system_df.sort_values('Count', ascending=False)
colors = sns.color_palette('Set2', len(system_df))
wedges, texts, autotexts = ax6.pie(system_df['Count'], labels=system_df['System'], 
                                     autopct='%1.1f%%', colors=colors, startangle=90)
ax6.set_title('Organ System Distribution', fontsize=12, fontweight='bold')

# 7. Cell Type-Tissue Heatmap (top combinations)
ax7 = plt.subplot(3, 3, 7)

# Create matrix for top cell types and tissues
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

# Get top 8 cell types and top 10 tissues
top_8_cells = [ct for ct, _ in cell_type_counts.most_common(8)]
top_10_tissues = [t for t, _ in tissue_counts.most_common(10)]

# Create heatmap data
heatmap_data = []
for cell_type in top_8_cells:
    row = []
    for tissue in top_10_tissues:
        count = cell_tissue_matrix.get(cell_type, {}).get(tissue, 0)
        row.append(count)
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, index=top_8_cells, columns=top_10_tissues)
sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax7, cbar_kws={'label': 'Co-occurrence'})
ax7.set_title('Cell Type-Tissue Co-occurrence', fontsize=12, fontweight='bold')
ax7.set_xlabel('')
ax7.set_ylabel('')
plt.setp(ax7.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax7.get_yticklabels(), rotation=0, fontsize=8)

# 8. Technology Distribution
ax8 = plt.subplot(3, 3, 8)
assay_counts = Counter()
for assays in df['assays']:
    assay_counts.update(parse_field(assays))

top_assays = dict(assay_counts.most_common(8))
assay_df = pd.DataFrame(list(top_assays.items()), columns=['Assay', 'Count'])
bars = ax8.bar(range(len(assay_df)), assay_df['Count'], color='teal')
ax8.set_xticks(range(len(assay_df)))
ax8.set_xticklabels(assay_df['Assay'], rotation=45, ha='right')
ax8.set_ylabel('Number of Datasets')
ax8.set_title('Top Sequencing Technologies', fontsize=12, fontweight='bold')
for i, v in enumerate(assay_df['Count']):
    ax8.text(i, v + 0.5, str(v), ha='center')

# 9. Endocrine Enrichment vs Dataset Size
ax9 = plt.subplot(3, 3, 9)
x = np.log10(df['dataset_total_cell_count'] + 1)
y = np.log10(df['endocrine_percentage'] + 0.01)
scatter = ax9.scatter(x, y, c=df['endocrine_cell_count'], s=50, alpha=0.6, 
                      cmap='viridis', edgecolors='black', linewidth=0.5)
ax9.set_xlabel('Log10(Total Cell Count)')
ax9.set_ylabel('Log10(Endocrine %)')
ax9.set_title('Endocrine Enrichment vs Dataset Size', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax9)
cbar.set_label('Endocrine Cell Count', rotation=270, labelpad=20)

plt.suptitle('Endocrine Cell Dataset Analysis: Cell Types, Tissues, and Diseases', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('endocrine_metadata_overview.pdf', dpi=300, bbox_inches='tight')
plt.savefig('endocrine_metadata_overview.png', dpi=150, bbox_inches='tight')
print("Saved: endocrine_metadata_overview.pdf/png")

# Create a network visualization of cell type-tissue-disease relationships
fig2, axes = plt.subplots(1, 2, figsize=(20, 10))

# Network 1: Cell Type - Tissue Network
ax = axes[0]
G1 = nx.Graph()

# Add nodes and edges for top cell types and tissues
for idx, row in df.iterrows():
    cell_types = parse_field(row['endocrine_cell_types'])
    tissues = parse_field(row['tissues'])
    
    for cell_type in cell_types[:3]:  # Limit to avoid overcrowding
        if cell_type in top_8_cells:
            for tissue in tissues[:3]:
                if tissue in top_10_tissues:
                    G1.add_edge(cell_type, tissue)

# Set node colors
node_colors = []
for node in G1.nodes():
    if node in top_8_cells:
        node_colors.append('lightblue')
    else:
        node_colors.append('lightcoral')

pos = nx.spring_layout(G1, k=2, iterations=50)
nx.draw_networkx_nodes(G1, pos, node_color=node_colors, node_size=1000, alpha=0.8, ax=ax)
nx.draw_networkx_edges(G1, pos, alpha=0.3, ax=ax)
nx.draw_networkx_labels(G1, pos, font_size=8, ax=ax)
ax.set_title('Cell Type - Tissue Network', fontsize=14, fontweight='bold')
ax.axis('off')

# Network 2: Cell Type - Disease Network
ax = axes[1]
G2 = nx.Graph()

# Add nodes and edges for cell types and diseases
for idx, row in df.iterrows():
    cell_types = parse_field(row['endocrine_cell_types'])
    diseases = parse_field(row['diseases'])
    
    for cell_type in cell_types[:2]:
        if cell_type in top_8_cells:
            for disease in diseases:
                if disease != 'normal':
                    G2.add_edge(cell_type, disease)

# Set node colors
node_colors = []
for node in G2.nodes():
    if node in top_8_cells:
        node_colors.append('lightgreen')
    else:
        node_colors.append('lightsalmon')

pos = nx.spring_layout(G2, k=2, iterations=50)
nx.draw_networkx_nodes(G2, pos, node_color=node_colors, node_size=800, alpha=0.8, ax=ax)
nx.draw_networkx_edges(G2, pos, alpha=0.3, ax=ax)
nx.draw_networkx_labels(G2, pos, font_size=7, ax=ax)
ax.set_title('Cell Type - Disease Network', fontsize=14, fontweight='bold')
ax.axis('off')

plt.suptitle('Network Relationships in Endocrine Cell Datasets', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('endocrine_networks.pdf', dpi=300, bbox_inches='tight')
plt.savefig('endocrine_networks.png', dpi=150, bbox_inches='tight')
print("Saved: endocrine_networks.pdf/png")

print("\nVisualization complete!")