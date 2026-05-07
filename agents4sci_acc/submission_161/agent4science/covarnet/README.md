# CoVarNet Analysis Directory

This directory contains the prepared data and scripts for CoVarNet cellular module discovery analysis on the neuroendocrine single-cell dataset.

## Current Files

### Main Analysis Files
- **covarnet_metadata_filtered_v2.csv** - Primary CoVarNet input with 247,404 cells
- **covarnet_metadata_full_v2.csv** - Complete metadata backup
- **covarnet_summary_stats_v2.csv** - Summary statistics
- **COVARNET_ANALYSIS_LOG.md** - Detailed analysis documentation

### Scripts
- **rebuild_metadata_v2.py** - Script to regenerate metadata from integrated h5ad
- **tutorial_discovery.R** - CoVarNet discovery tutorial code
- **tutorial_discovery.html** - Tutorial documentation

### Archive
- **archive_old/** - Contains previous versions and exploratory scripts

## Quick Start

### Load metadata in R:
```r
library(CoVarNet)
meta <- read.csv("covarnet_metadata_filtered_v2.csv", row.names = 1)
```

### Run CoVarNet analysis:
```r
# Calculate frequencies
mat_fq_raw <- freq_calculate(meta)

# Normalize
mat_fq_norm <- freq_normalize(mat_fq_raw, normalize="minmax")

# Run NMF
res <- nmf(mat_fq_norm, rank = 2:20, method = "nsNMF", seed = rep(123456, 6))
```

## Dataset Summary
- **Total cells**: 247,404
- **Tissues**: 48
- **Batches**: 15
- **Major clusters**: 9 (Endocrine, Epithelial, Immune, Stromal, etc.)
- **Fine cell types**: 145

## Cell Composition
- Epithelial: 58.3%
- Immune: 12.9%
- Stromal: 8.1%
- **Endocrine: 4.2%** (focus of analysis)
- Other categories: 16.5%

## Contact
For questions about the metadata preparation, see `COVARNET_ANALYSIS_LOG.md`