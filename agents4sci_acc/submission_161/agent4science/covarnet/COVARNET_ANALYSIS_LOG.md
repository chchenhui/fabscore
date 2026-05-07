# CoVarNet Analysis Log - Endocrine Niche Discovery

**Date**: 2025-08-26  
**Dataset**: Neuroendocrine integrated single-cell data (scVI)  
**Purpose**: Cross-tissue cellular module discovery for endocrine niches

## Overview
Prepared metadata for CoVarNet analysis to identify cellular modules and niches surrounding endocrine cells across multiple tissues. The analysis will reveal cross-tissue patterns of cell-cell interactions in the endocrine microenvironment.

## Dataset Statistics

### Input Data
- **Source**: `/scratch/rli/project/agent/results/data_integration_2025-08-25/scvi_unique_integration/neuroendocrine_scvi_integrated_unique.h5ad`
- **Total cells**: 247,404
- **Total genes**: 28,359
- **Total tissues**: 48
- **Total batches**: 15 (all with ≥100 cells)

### Cell Composition
| Major Cluster | Cell Count | Percentage | Description |
|--------------|------------|------------|-------------|
| **Epithelial** | 144,270 | 58.3% | Enterocytes, foveolar cells, alveolar cells, etc. |
| **Immune** | 31,908 | 12.9% | T cells, B cells, plasma cells, macrophages |
| **Unknown** | 21,636 | 8.8% | Unclassified cells |
| **Stromal** | 20,141 | 8.1% | Fibroblasts, mesenchymal cells, smooth muscle |
| **Endocrine** | 10,321 | 4.2% | Neuroendocrine, enteroendocrine cells |
| **Stem/Progenitor** | 7,938 | 3.2% | Stem cells and progenitor populations |
| **Neural/Glial** | 5,026 | 2.0% | Neurons, glial cells, enteric neurons |
| **Other** | 4,378 | 1.8% | Miscellaneous cell types |
| **Endothelial** | 1,786 | 0.7% | Blood vessel and lymphatic endothelium |

### Top Tissues (by cell count)
1. Body of stomach: 64,293 cells (26.0%)
2. Ileum: 47,993 cells (19.4%)
3. Lung: 18,803 cells (7.6%)
4. Alveolar sac: 17,425 cells (7.0%)
5. Small intestine: 17,130 cells (6.9%)
6. Colon: 14,274 cells (5.8%)
7. Ileal epithelium: 9,352 cells (3.8%)
8. Duodenum: 8,870 cells (3.6%)

## Endocrine Cell Analysis

### Endocrine Subtypes
- **Neuroendocrine cells**: 3,751 (36.3% of endocrine)
- **Enteroendocrine cells**: 3,196 (31.0%)
- **Type G enteroendocrine**: 1,515 (14.7%)
- **P/D1 enteroendocrine**: 711 (6.9%)
- **Type EC enteroendocrine**: 382 (3.7%)
- **Type L enteroendocrine**: 330 (3.2%)
- **Type A enteroendocrine**: 286 (2.8%)

### Endocrine Niche Composition
The non-endocrine cells (95.8% of dataset) that form the endocrine niche include:

#### Epithelial Niche (60.9% of non-endocrine)
- Enterocytes: 40,312 cells
- Foveolar cells: 30,353 cells
- Stem cells: 23,337 cells
- Alveolar cells: 9,435 cells
- Mucous neck cells: 7,483 cells

#### Immune Niche (13.5% of non-endocrine)
- Goblet cells: 8,467 cells
- Intestinal goblet cells: 4,700 cells
- Club cells: 4,500 cells
- IgA plasma cells: 2,768 cells
- Activated T cells: 1,440 cells

#### Stromal Niche (8.5% of non-endocrine)
- Mesodermal cells: 7,389 cells
- Stromal cells: 6,759 cells
- Smooth muscle cells: 1,334 cells
- Pericytes: 1,101 cells
- Interstitial cells of Cajal: 969 cells

#### Vascular Niche (0.8% of non-endocrine)
- Vein endothelial: 598 cells
- Capillary endothelial: 516 cells
- Arterial endothelial: 370 cells
- Lymphatic endothelial: 222 cells

#### Neural Niche (2.1% of non-endocrine)
- Glial cells: 3,042 cells
- Motor neurons: 433 cells
- Neurons: 420 cells
- Oligodendrocyte precursors: 293 cells

## Metadata Structure

### File Format
The CoVarNet metadata CSV contains the following columns:
- **cellID**: Unique identifier (format: `C{7-digit}_tissue`)
- **sampleID**: Batch identifier (15 unique batches)
- **tissue**: Tissue of origin (48 unique tissues)
- **majorCluster**: Broad cell category (9 categories)
- **subCluster**: Fine-grained cell type (145 types)
- **cellSort**: All marked as "cell" (single-cell data)

### Output Files
1. **covarnet_metadata_filtered_v2.csv** - Primary input for CoVarNet (247,404 cells)
2. **covarnet_metadata_full_v2.csv** - Complete metadata backup
3. **covarnet_summary_stats_v2.csv** - Summary statistics
4. **rebuild_metadata_v2.py** - Script to regenerate metadata
5. **tutorial_discovery.R** - CoVarNet discovery tutorial code

## Cell Type Classification Logic

The classification system prioritizes endocrine cells first, then classifies remaining cells into major categories based on keyword matching:

1. **Endocrine**: Keywords include endocrine, neuroendocrine, enteroendocrine, chromaffin, and specific hormone-producing cells
2. **Immune**: T cells, B cells, plasma cells, macrophages, NK cells, dendritic cells
3. **Endothelial**: Blood vessel and lymphatic endothelium
4. **Stromal**: Fibroblasts, mesenchymal cells, smooth muscle, pericytes
5. **Neural/Glial**: Neurons, glial cells, Schwann cells, oligodendrocytes
6. **Epithelial**: Non-endocrine epithelial cells including enterocytes, goblet cells, alveolar cells
7. **Stem/Progenitor**: Stem cells and progenitor populations
8. **Other**: Type B pancreatic cells and other unclassified types
9. **Unknown**: Cells marked as "unknown" in original annotation

## Next Steps for CoVarNet Analysis

1. **Install CoVarNet R package** in R 4.1.2 environment
2. **Load metadata**: Use `covarnet_metadata_filtered_v2.csv`
3. **Calculate cell type frequencies** per sample
4. **Normalize frequencies** (min-max normalization)
5. **Run NMF** to identify cellular modules (rank 2-20)
6. **Select optimal module number** based on cophenetic correlation
7. **Analyze module composition** and tissue distribution
8. **Identify endocrine-associated modules**

## Key Insights

- **Endocrine cells represent only 4.2%** of the dataset, emphasizing the importance of studying their cellular niche
- **Epithelial cells dominate** the endocrine microenvironment (58.3% of all cells)
- **Strong immune component** with 12.9% of cells being immune cells
- **Cross-tissue representation** with cells from 48 different tissues
- **Balanced batch distribution** with all 15 batches having sufficient cells (≥100)

## Technical Notes

- Used `cell_type` column instead of `endocrine_type_simple` for complete cell type information
- Filtered to samples with ≥100 cells (all 15 batches qualified)
- Cell IDs formatted for conciseness while maintaining uniqueness
- Metadata compatible with CoVarNet's expected input format

## Directory Cleanup (2025-09-02)

### Removed Directories
- **archive_old/** - 17 obsolete Python/R scripts from initial attempts
- **results/covarnet_20250827-2/** - Duplicate analysis outputs
- **archive_network_intermediate/** - Intermediate network visualization scripts

### Preserved Scripts
All three network visualization scripts retained (serve different purposes):
1. **create_network_visualizations.R** - General network structure and communities
2. **create_endocrine_analysis.R** - Endocrine-specific analysis
3. **create_endocrine_network_final.R** - Final tissue-annotated visualizations

### Current Directory Structure
- Main analysis script: `neuroendocrine_covarnet_discovery_improved.R`
- Cross-tissue analysis: `analyze_endocrine_cross_tissue.R`
- Network scripts: 3 complementary visualization scripts
- Data files: Metadata (v2), RDS results, PDF visualizations
- Documentation: Analysis logs, insights, and summaries

---
*Analysis prepared for cross-tissue cellular module discovery using CoVarNet framework*
*Last updated: 2025-09-02*