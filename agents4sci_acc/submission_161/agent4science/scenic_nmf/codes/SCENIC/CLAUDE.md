# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## SCENIC Pipeline Setup for Single-Cell Regulatory Network Analysis

### Overview
This directory contains SCENIC (Single-Cell rEgulatory Network Inference and Clustering) analysis pipelines for human (hg38) single-cell datasets. The pipeline converts ENSEMBL gene IDs to gene symbols and runs the full SCENIC workflow.

### Environment Setup
```bash
# Activate conda environment
source /wanglab/jguo/miniconda3/etc/profile.d/conda.sh
conda activate pyscenic

# Required packages (already installed):
# - pyscenic (0.12.1+8.gd2309fe)
# - mygene (for ENSEMBL → gene symbol conversion)
# - scanpy, pandas, numpy, loompy
```

### Key Database Files (Required)
Located in `/scratch/jguo/senic/`:
- `allTFs_hg38.txt` - Human transcription factors list
- `hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather` - Ranking database
- `hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather` - Ranking database
- `motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl` - Motif annotations

### Directory Structure
```
/scratch/jguo/unique_data/senic/
├── trial/                     # Test runs with 1% subsampled data
├── normal_entero/             # Normal enteroendocrine full dataset analysis
├── entero/                    # Strict endocrine full dataset analysis
└── SCENIC_hg38_Documentation.md  # Original documentation
```

### Running SCENIC Analysis

#### Test Run (1% of cells)
For testing the pipeline with a small subset:
```python
# In the Python script, subsample 1% of cells
n_cells_sample = int(n_cells_total * 0.01)  # 1%
sample_indices = np.random.choice(n_cells_total, size=n_cells_sample, replace=False)
adata_test = adata_full[sample_indices, :].copy()
```

#### Full Dataset Analysis with SBATCH
Create SBATCH script with proper resources:
```bash
#!/bin/bash
#SBATCH --job-name=scenic_analysis
#SBATCH --output=scenic_%j.out
#SBATCH --error=scenic_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=100G
#SBATCH --partition=general
```

Submit job:
```bash
sbatch run_scenic.sbatch
```

### Key Pipeline Steps

1. **Data Loading and Gene Conversion**
   - Load h5ad file
   - Check if genes are ENSEMBL IDs or symbols
   - Convert ENSEMBL to symbols using mygene API (batch processing)
   - Handle duplicates with make_unique()

2. **Loom File Creation**
   - Convert sparse matrix to dense if needed
   - Create loom with correct dimensions (genes x cells)
   - Include nGene and nUMI calculations (axis=1 for cells)

3. **GRN Inference (GRNBoost2)**
   ```bash
   pyscenic grn input.loom allTFs_hg38.txt -o adjacencies.csv --num_workers 28 --method grnboost2
   ```

4. **Regulon Prediction (cisTarget)**
   ```bash
   pyscenic ctx adjacencies.csv [databases] --annotations_fname motifs.tbl --expression_mtx_fname input.loom --output regulons.csv --mask_dropouts --num_workers 28
   ```

5. **AUCell Scoring**
   ```bash
   pyscenic aucell input.loom regulons.csv --output aucell.csv --num_workers 28
   ```

6. **Results Integration**
   - Load AUCell matrix
   - Add regulon activities to adata.obs
   - Save as h5ad file with integrated results

### Common Issues and Solutions

#### Dimension Mismatch in Loom Creation
**Problem**: `Column attribute 'nGene' is not the same length as number of columns`
**Solution**: Ensure correct axis for calculations:
```python
"nGene": np.array(np.sum(expression_matrix > 0, axis=1)).flatten(),  # axis=1 for cells
"nUMI": np.array(np.sum(expression_matrix, axis=1)).flatten(),  # axis=1 for cells
```

#### Empty GRN Output File
**Problem**: Previous interrupted run creates empty output file
**Solution**: Remove empty file before rerunning:
```bash
rm -f adjacencies.csv
```

#### Resource Settings for SLURM
- Use 28-30 workers for parallel processing
- Request 100GB memory for full datasets
- Allow 7 days for complete analysis
- Use general partition (not long partition)

### Monitoring Progress

Check job status:
```bash
squeue -j [JOB_ID]
```

Monitor logs:
```bash
# SLURM output
tail -f scenic_[JOB_ID].out

# Python logs (timestamped)
tail -f scenic_*_[TIMESTAMP].log
```

Check generated files:
```bash
ls -lah *.loom *.csv *.h5ad
```

### Datasets Analyzed

1. **neuroendocrine_scvi_integrated_unique.h5ad**
   - Location: `/scratch/jguo/unique_data/`
   - Size: 247,404 cells × 28,359 genes
   - Test run: 1% (2,474 cells) - Completed with 460 regulons

2. **normal.h5ad**
   - Location: `/scratch/jguo/unique_data/sub_adata/disease_sub_adata/`
   - Size: 146,813 cells × 22,365 genes
   - Test run: 1% (1,468 cells) - Completed with 465 regulons

3. **normal_enteroendocrine.h5ad**
   - Location: `/scratch/jguo/unique_data/sub_adata/disease_sub_adata/`
   - Size: 10,177 cells × 22,365 genes
   - Full dataset analysis - Running on SLURM

4. **strict_endocrine.h5ad**
   - Location: `/scratch/jguo/unique_data/sub_adata/`
   - Full dataset analysis - Submitted to SLURM

### Expected Timeline
- Data loading: 5-20 minutes
- Gene conversion: 10-30 minutes (depends on dataset size)
- GRN inference: 1-8 hours
- cisTarget: 1-3 hours
- AUCell: 30-60 minutes
- Total: 3-12 hours for full datasets

### Output Files
- `*_scenic_full.loom` - Expression matrix for SCENIC
- `*_adj_full.csv` - Gene regulatory network adjacencies
- `*_reg_full.csv` - High-confidence regulons
- `*_aucell_full.csv` - Regulon activity scores
- `*_scenic_full_results.h5ad` - Final AnnData with regulon activities