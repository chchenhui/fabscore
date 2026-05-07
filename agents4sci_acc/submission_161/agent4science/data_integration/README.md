# Neuroendocrine Data Integration Pipeline

## Overview
Complete pipeline for integrating neuroendocrine datasets using multiple batch correction methods (Unintegrated, ComBat, scVI) with comprehensive benchmarking and visualization.

## Directory Structure
```
/scratch/rli/project/agent/data_integration/
├── results/                              # All outputs consolidated here
│   ├── scvi_unique_integration/         # Integrated datasets (~14GB)
│   │   ├── neuroendocrine_scvi_integrated_unique.h5ad
│   │   ├── scvi_model/
│   │   └── [subdirectories]
│   ├── *.pdf, *.png, *.svg             # All visualizations
│   ├── *.csv                            # Statistics and metrics
│   └── INTEGRATED_ANALYSIS_SUMMARY.md   # Complete analysis summary
├── DATA_INTEGRATION_LOG.md              # Detailed project log
└── unique_datasets_metadata_final.csv   # Dataset metadata (55 unique)
```

## Main Pipeline Scripts

### 1. Data Processing
**`neuroendocrine_process_unique.py`**
- Loads 55 unique datasets (247,404 cells × 28,359 genes)
- Applies QC filtering (min_genes=200, max_mt=20%)
- Creates merged dataset with batch information
- Generates QC visualizations
- Output: `results/qc_*.png`, `results/*_unique.csv`

### 2. Integration Pipeline
**`neuroendocrine_integration_scvi_unique.py`**
- Implements three integration methods:
  - Unintegrated (baseline PCA)
  - ComBat (linear batch correction)
  - scVI (deep learning VAE)
- Parameters: 3000 HVGs, 30 latent dimensions, 50 epochs
- Output: `results/scvi_unique_integration/`

### 3. Analysis & Benchmarking
**`integration_analysis_complete.py`**
- Complete visualization pipeline
- QC metrics, batch effects, integration comparisons
- Endocrine cell marker analysis
- Output: `results/integration_comparison_improved.pdf`

**`benchmark_scib_final.py`**
- SCIB-metrics benchmarking
- Evaluates biological conservation and batch correction
- Output: `results/scib_results_table_*.svg`

### 4. Execution Scripts
**`run_analysis.sh`**
- SLURM submission for complete analysis
- GPU-enabled (1 H100), 256GB RAM, 48hr runtime
- Runs `integration_analysis_complete.py`

**`run_scvi_integration.sh`**
- SLURM submission for integration only
- Similar resource requirements

## Key Results

### Integration Performance (SCIB Benchmarking)
| Method | Bio Conservation | Batch Correction | Overall Score |
|--------|-----------------|------------------|---------------|
| Unintegrated | 0.000 | 0.400 | 0.200 |
| ComBat | 1.000 | 0.301 | 0.651 |
| **scVI** | 0.756 | 0.713 | **0.734** |

### Dataset Statistics
- **Total Cells**: 247,404
- **Total Genes**: 28,359
- **Unique Datasets**: 55
- **Batch Count**: 15
- **Tissue Types**: 48
- **Cell Types**: Multiple endocrine/neuroendocrine

## Critical Technical Notes

### Embedding Selection
**IMPORTANT**: Use correct embeddings for analysis:
- Unintegrated: 50D PCA (NOT UMAP)
- ComBat: 30D PCA (NOT UMAP)
- scVI: 30D latent space (NOT UMAP)

UMAP is for visualization only and should not be used for quantitative metrics.

### Memory Management
- Requires 256GB RAM for full dataset
- Use sparse matrices when possible
- Clear intermediate variables with `gc.collect()`

## Quick Start

### Run Complete Analysis
```bash
cd /scratch/rli/project/agent/data_integration
sbatch run_analysis.sh
```

### Run Integration Only
```bash
python neuroendocrine_integration_scvi_unique.py
```

### Run Benchmarking Only
```bash
python benchmark_scib_final.py
```

## Output Files

### Integrated Datasets
- `results/scvi_unique_integration/neuroendocrine_scvi_integrated_unique.h5ad`
  - Contains layers: raw, combat, scvi
  - Contains embeddings: X_umap, X_umap_unintegrated, X_umap_combat
  - Size: ~2.4GB compressed

### Visualizations
- QC metrics: `qc_violin_*.png`, `qc_scatter_*.png`
- Integration comparison: `integration_comparison_improved.pdf`
- Batch effects: `batch_overlap_visualization.pdf`, `umap_density_*.pdf`
- SCIB results: `scib_results_table_scaled_final.svg`

### Statistics
- `batch_information_unique.csv`: Batch-level statistics
- `integration_summary_unique.csv`: Integration metrics
- `scib_summary_final.csv`: SCIB benchmark scores

## Dependencies
- scanpy >= 1.9.0
- scvi-tools >= 1.0.0
- scib >= 1.1.0
- PyTorch with CUDA support
- Additional: numpy, pandas, matplotlib, seaborn

## Environment Setup
```bash
conda activate data_integration
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,expandable_segments:True
export OMP_NUM_THREADS=5
```

## Troubleshooting

### GPU Memory Issues
- Reduce batch size in scVI training
- Use `PYTORCH_CUDA_ALLOC_CONF` settings
- Clear GPU cache: `torch.cuda.empty_cache()`

### File Not Found
- Check paths in scripts point to `results/` directory
- Ensure previous steps completed successfully

### Out of Memory
- Subsample dataset for testing
- Use HPC nodes with more RAM
- Process in chunks

## Last Successful Run
- **Date**: 2025-08-26
- **Runtime**: ~48 minutes for integration
- **Status**: Complete success with all methods

## Citation
If using this pipeline, please cite:
- scvi-tools: Lopez et al., 2018
- SCIB: Luecken et al., 2021
- ComBat: Johnson et al., 2007

## Contact
Pipeline developed for neuroendocrine cell atlas project
Location: /scratch/rli/project/agent/data_integration/