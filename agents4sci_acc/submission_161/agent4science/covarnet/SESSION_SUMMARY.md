# CoVarNet Analysis Session Summary
**Last Updated**: 2025-09-02  
**Location**: `/scratch/rli/project/agent/covarnet/`

## Project Overview
Analyzed neuroendocrine dataset using CoVarNet framework for cellular module discovery and network visualization across 48 tissues to understand endocrine cellular niches.

## Key Parameters Used
- K = 12 (number of NMF modules)
- top_n = 10 (top edges per module)
- corr = 0.1 (correlation threshold)
- fdr = 0.05 (FDR threshold)

## Data Processing
- Original tissues: 48 types mapped to 13 categories
- Tissue mapping implemented for cleaner visualization
- 247,404 cells from 15 samples analyzed
- 145 cell types identified

## Network Analysis Results
- 124 nodes, 1,150 significant edges (FDR < 0.05)
- 16 communities detected (modularity: 0.316)
- 145 edges involving endocrine cells
- 8 major cell type groups identified

## Generated Visualizations

### Original Network Plots (preserved)
- network_global_enhanced_v1.pdf - Force-directed layout
- network_global_enhanced_v2.pdf - Circular layout

### Improved Network Visualizations
- network_global_simplified.pdf - Major cell type groups only
- network_top_connected.pdf - Top 30 most connected nodes
- network_subgroups.pdf - Sub-networks for each major category
- network_communities.pdf - Community detection visualization

### Endocrine-Focused Visualizations
- network_global_endocrine_focus.pdf - Global with endocrine relationship strength
- network_communities_annotated.pdf - Annotated communities with endocrine highlights
- network_endocrine_subnetwork.pdf - Endocrine cells and direct connections

## Module Analysis
- module_weights_all_K12.pdf - Complete weight heatmap
- module_weights_top15_enhanced_K12.pdf - Top 15 samples per module
- module_distribution_tissue_category_K12.pdf - Distribution by tissue category
- module_distribution_majorCluster_K12.pdf - Distribution by cell type

## Endocrine Analysis
Top endocrine-enriched modules:
- CM06: 33.6% enrichment
- CM07: 27.8% enrichment
- CM12: 25.2% enrichment
- CM04: 20.0% enrichment

## File Organization
- Main script: neuroendocrine_covarnet_discovery_improved.R
- Helper scripts: create_improved_network_plots.R, create_endocrine_focused_networks.R
- Results archived: /results/covarnet_20250827-2/
- Data files: cor_pair_K12.rds, network_K12.rds, endocrine_enrichment_K12.rds

## Technical Notes
- Fixed network$global structure (LIST with node/edge dataframes, not igraph)
- Used gr.igraph_global() with network$global directly
- Implemented tissue mapping dictionary for categorization
- Added community detection with Louvain algorithm
- Enhanced visualizations with endocrine relationship strength

## Key Findings
- Successfully identified cellular modules with K=12
- Endocrine cells show strong modular organization (4 modules >20% enrichment)
- Clear clustering of cell types in network communities
- Tissue categories provide cleaner visualization than individual tissues
- Small intestine serves as central hub (63% of non-endocrine connections)
- Strong immune-endocrine axis across all tissues

## Directory Cleanup (2025-09-02)
### Removed
- `archive_old/` - 17 obsolete scripts
- `results/covarnet_20250827-2/` - Duplicate outputs
- `archive_network_intermediate/` - Intermediate scripts

### Preserved
All three network visualization scripts (serve complementary purposes):
1. **create_network_visualizations.R** - General network structure
2. **create_endocrine_analysis.R** - Endocrine-specific analysis
3. **create_endocrine_network_final.R** - Tissue-annotated visualizations

## Current Status
- ✅ Analysis complete with 12 cellular modules identified
- ✅ Network visualizations generated (11 PDFs)
- ✅ Cross-tissue insights documented
- ✅ Directory cleaned and organized
- ✅ Memory logs updated

