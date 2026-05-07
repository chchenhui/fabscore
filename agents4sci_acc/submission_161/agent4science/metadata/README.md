# Endocrine Cell Metadata Analysis

This directory contains comprehensive analysis of endocrine cell metadata from 64 CellxGene datasets, examining relationships between cell types, tissues, and diseases.

## Overview
- **Datasets analyzed**: 64
- **Total endocrine cells**: 100,628
- **Total cells**: 16.3 million
- **Average endocrine percentage**: 1.97%

## Key Findings
- **175 tissue-disease pairs** identified
- **59 cell type-disease pairs** documented
- **43 tissues** with disease associations
- **14 endocrine cell types** linked to diseases
- **25 unique diseases** studied

## Files

### Analysis Scripts
- `analyze_endocrine_metadata.py` - Main analysis script for metadata exploration
- `visualize_endocrine_relationships.py` - Creates overview visualizations and networks
- `create_cooccurrence_heatmaps.py` - Generates tissue-disease and cell type-disease heatmaps

### Results
- `ENDOCRINE_METADATA_INSIGHTS.md` - Comprehensive analysis summary and insights
- `endocrine_metadata_overview.pdf/png` - 9-panel visualization overview
- `endocrine_networks.pdf/png` - Network relationships between entities
- `disease_cooccurrence_heatmaps.pdf/png` - Main co-occurrence heatmaps (4 panels)
- `disease_category_heatmaps.pdf/png` - Disease category-specific analysis

## Top Discoveries

### Most Common Endocrine Cell Types
1. Enteroendocrine cells (24 datasets, 44 tissues)
2. Neuroendocrine cells (17 datasets, 23 tissues)
3. Lung neuroendocrine cells (11 datasets, 6 tissues)

### Disease Associations
- **Cancer**: 28% of disease mentions
- **Inflammatory**: 12% (Crohn's disease most widespread - 22 tissues)
- **Metaplasia**: 9% (Barrett's esophagus, gastric intestinal metaplasia)

### Tissue Distribution
- **Gastrointestinal**: 63% of tissue occurrences
- **Respiratory**: 17%
- **Genitourinary**: 6%

## Usage

Run the analysis:
```bash
conda activate data_integration
python analyze_endocrine_metadata.py
python visualize_endocrine_relationships.py
python create_cooccurrence_heatmaps.py
```

## Data Source
Analysis based on: `/scratch/rli/data/neuroendocrine_dataset/endocrine_datasets_summary.csv`

---
*Generated: 2025-09-01*