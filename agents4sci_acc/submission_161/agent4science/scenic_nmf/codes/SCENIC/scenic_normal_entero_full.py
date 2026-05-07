#!/usr/bin/env python3

import os, glob, re, pickle
from functools import partial
from collections import OrderedDict
import operator as op
from cytoolz import compose
import sys
import traceback
import logging
from datetime import datetime
import random

import pandas as pd
import seaborn as sns
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import loompy as lp

from pyscenic.export import export2loom, add_scenic_metadata
from pyscenic.utils import load_motifs
from pyscenic.transform import df2regulons
from pyscenic.aucell import aucell
from pyscenic.binarization import binarize
from pyscenic.rss import regulon_specificity_scores
from pyscenic.plotting import plot_binarization, plot_rss

def setup_logging():
    """Setup logging to both file and console"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"scenic_normal_entero_full_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

def convert_ensembl_to_symbols(adata, logger):
    """Convert ENSEMBL IDs to gene symbols using mygene"""
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        
        logger.info("Converting ENSEMBL IDs to gene symbols...")
        
        # Get ENSEMBL IDs (remove version numbers if present)
        ensembl_ids = [gene_id.split('.')[0] for gene_id in adata.var_names]
        
        # Query mygene in batches to avoid timeout
        batch_size = 1000
        all_results = []
        
        for i in range(0, len(ensembl_ids), batch_size):
            batch = ensembl_ids[i:i+batch_size]
            logger.info(f"Converting batch {i//batch_size + 1}/{(len(ensembl_ids)-1)//batch_size + 1}")
            
            try:
                results = mg.querymany(batch, scopes='ensembl.gene', fields='symbol', species='human')
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Batch conversion failed: {e}")
                # Add placeholder results for failed batch
                for gene_id in batch:
                    all_results.append({'query': gene_id, 'symbol': gene_id})
        
        # Create mapping dictionary
        gene_mapping = {}
        for result in all_results:
            query_id = result['query']
            if 'symbol' in result and result['symbol']:
                gene_mapping[query_id] = result['symbol']
            else:
                # Keep original ENSEMBL ID if no symbol found
                gene_mapping[query_id] = query_id
        
        # Apply mapping to AnnData
        original_names = [gene_id.split('.')[0] for gene_id in adata.var_names]
        new_names = [gene_mapping.get(gene_id, gene_id) for gene_id in original_names]
        
        # Handle duplicates by making unique
        adata.var_names = new_names
        adata.var_names_make_unique()
        
        logger.info(f"Converted {sum(1 for old, new in zip(original_names, new_names) if old != new)} genes to symbols")
        logger.info(f"Sample converted genes: {list(adata.var_names[:10])}")
        
        return True
        
    except ImportError:
        logger.warning("mygene not available, trying alternative approach...")
        return False
    except Exception as e:
        logger.error(f"Gene conversion failed: {e}")
        return False

def main():
    logger, log_filename = setup_logging()
    
    try:
        # Set scanpy settings
        sc.settings.njobs = 8
        logger.info("Starting SCENIC analysis (FULL dataset) for normal enteroendocrine dataset...")
        logger.info(f"Log file: {log_filename}")
        logger.info("=" * 60)

        # Load normal enteroendocrine dataset
        logger.info("Loading normal enteroendocrine dataset...")
        data_path = '/scratch/jguo/unique_data/sub_adata/disease_sub_adata/normal_enteroendocrine.h5ad'
        
        if not os.path.exists(data_path):
            logger.error(f"Dataset not found at {data_path}")
            return
            
        # Load dataset
        logger.info("Loading full dataset...")
        adata_full = sc.read_h5ad(data_path)
        logger.info(f"Full dataset shape: {adata_full.shape}")
        
        # Use the full dataset (no subsampling)
        adata_test = adata_full.copy()
        logger.info(f"Using full dataset: {adata_test.shape}")
        logger.info(f"Original gene names: {list(adata_test.var_names[:5])}")
        
        # Check if gene names are already symbols or ENSEMBL IDs
        first_gene = str(adata_test.var_names[0])
        if first_gene.startswith('ENSG'):
            # Convert gene names from ENSEMBL to symbols
            success = convert_ensembl_to_symbols(adata_test, logger)
            if not success:
                # Fallback: try to extract symbols from existing annotations
                if 'gene_symbols' in adata_test.var.columns:
                    logger.info("Using existing gene_symbols column")
                    adata_test.var_names = adata_test.var['gene_symbols']
                    adata_test.var_names_make_unique()
                elif 'gene_names' in adata_test.var.columns:
                    logger.info("Using existing gene_names column")
                    adata_test.var_names = adata_test.var['gene_names']
                    adata_test.var_names_make_unique()
                else:
                    logger.warning("Cannot convert gene names - continuing with ENSEMBL IDs")
        else:
            logger.info("Gene names appear to be symbols already, no conversion needed")
        
        logger.info(f"Final gene names: {list(adata_test.var_names[:5])}")
        
        # Check if data needs to be converted to dense
        if hasattr(adata_test.X, 'todense'):
            logger.info("Data is sparse, will convert to dense for loom creation")
                
        # Create loom file for SCENIC analysis
        f_loom_path_scenic = 'normal_entero_hg38_scenic_full.loom'
        
        if not os.path.exists(f_loom_path_scenic):
            logger.info("Creating loom file for SCENIC analysis...")
            
            # Prepare data for loom creation
            if hasattr(adata_test.X, 'todense'):
                expression_matrix = np.array(adata_test.X.todense())
            else:
                expression_matrix = np.array(adata_test.X)
                
            # Create basic row and column attributes for the loom file:
            row_attrs = {
                "Gene": np.array(adata_test.var_names, dtype=str),
            }
            col_attrs = {
                "CellID": np.array(adata_test.obs_names, dtype=str),
                "nGene": np.array(np.sum(expression_matrix > 0, axis=1)).flatten(),  # axis=1 for cells
                "nUMI": np.array(np.sum(expression_matrix, axis=1)).flatten(),  # axis=1 for cells
            }
            
            logger.info(f"Expression matrix shape: {expression_matrix.shape}")
            logger.info(f"Row attrs shape: {len(row_attrs['Gene'])}")
            logger.info(f"Col attrs shape: {len(col_attrs['CellID'])}")
            
            # Create loom file - Note: loompy expects genes x cells
            lp.create(f_loom_path_scenic, expression_matrix.T, row_attrs, col_attrs)
            logger.info(f"Created loom file: {f_loom_path_scenic}")
        else:
            logger.info(f"Loom file already exists: {f_loom_path_scenic}")

        # Define paths for hg38 databases
        f_tfs = '/scratch/jguo/senic/allTFs_hg38.txt'
        f_db_names = '/scratch/jguo/senic/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather /scratch/jguo/senic/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather'
        f_motif_path = '/scratch/jguo/senic/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl'

        # Check if required files exist
        required_files = [f_tfs, f_motif_path]
        required_files.extend(f_db_names.split())
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"Required file not found: {file_path}")
                logger.info("Please ensure all SCENIC database files are in /scratch/jguo/senic/")
                return
            else:
                logger.info(f"Found: {os.path.basename(file_path)}")

        logger.info("All required files found. Starting SCENIC workflow...")
        logger.info("=" * 50)

        # Step 1: GRN inference using GRNBoost2
        logger.info("Step 1: Gene Regulatory Network inference...")
        grn_output = 'normal_entero_hg38_adj_full.csv'
        if not os.path.exists(grn_output):
            grn_cmd = f"pyscenic grn {f_loom_path_scenic} {f_tfs} -o {grn_output} --num_workers 28 --method grnboost2"
            logger.info(f"Running: {grn_cmd}")
            exit_code = os.system(grn_cmd)
            if exit_code == 0:
                logger.info(f"GRN inference completed successfully. Output: {grn_output}")
            else:
                logger.error(f"GRN inference failed with exit code: {exit_code}")
                return
        else:
            logger.info(f"GRN output already exists: {grn_output}")

        # Check GRN output
        if os.path.exists(grn_output):
            adj_df = pd.read_csv(grn_output)
            logger.info(f"GRN adjacencies shape: {adj_df.shape}")
            logger.info(f"Sample adjacencies:\n{adj_df.head()}")

        # Step 2: Regulon prediction (cisTarget)
        logger.info("\nStep 2: Regulon prediction using cisTarget...")
        regulons_output = 'normal_entero_hg38_reg_full.csv'
        if not os.path.exists(regulons_output):
            ctx_cmd = f"pyscenic ctx {grn_output} {f_db_names} --annotations_fname {f_motif_path} --expression_mtx_fname {f_loom_path_scenic} --output {regulons_output} --mask_dropouts --num_workers 28"
            logger.info(f"Running: {ctx_cmd}")
            exit_code = os.system(ctx_cmd)
            if exit_code == 0:
                logger.info(f"Regulon prediction completed successfully. Output: {regulons_output}")
            else:
                logger.error(f"Regulon prediction failed with exit code: {exit_code}")
                return
        else:
            logger.info(f"Regulons output already exists: {regulons_output}")

        # Step 3: AUCell scoring
        logger.info("\nStep 3: AUCell scoring...")
        aucell_output = 'normal_entero_hg38_aucell_full.csv'
        if not os.path.exists(aucell_output):
            aucell_cmd = f"pyscenic aucell {f_loom_path_scenic} {regulons_output} --output {aucell_output} --num_workers 28"
            logger.info(f"Running: {aucell_cmd}")
            exit_code = os.system(aucell_cmd)
            if exit_code == 0:
                logger.info(f"AUCell scoring completed successfully. Output: {aucell_output}")
            else:
                logger.error(f"AUCell scoring failed with exit code: {exit_code}")
                return
        else:
            logger.info(f"AUCell output already exists: {aucell_output}")

        # Step 4: Load results and add to AnnData
        logger.info("\nStep 4: Loading results and integrating with AnnData...")
        if os.path.exists(aucell_output):
            # Load AUCell results
            auc_mtx = pd.read_csv(aucell_output, index_col=0)
            logger.info(f"AUCell matrix shape: {auc_mtx.shape}")
            
            # Ensure index compatibility
            auc_mtx.index = auc_mtx.index.astype(str)
            
            # Add regulon activities to adata.obs
            adata_test.obs = pd.concat([adata_test.obs, auc_mtx], axis=1)
            logger.info(f"Updated adata.obs shape: {adata_test.obs.shape}")
            
            # Save updated AnnData object
            output_h5ad = 'normal_entero_hg38_scenic_full_results.h5ad'
            adata_test.write(output_h5ad)
            logger.info(f"Saved results to: {output_h5ad}")
            
            # Get regulon names for summary
            regulon_cols = [col for col in auc_mtx.columns if '(+)' in col]
            logger.info(f"Found {len(regulon_cols)} regulons")
            if regulon_cols:
                logger.info(f"Sample regulons: {regulon_cols[:10]}")
            
            logger.info("=" * 60)
            logger.info("SCENIC full dataset analysis completed successfully!")
            logger.info(f"Results saved to: {output_h5ad}")
            logger.info(f"Regulon activities added to adata.obs with {len(regulon_cols)} regulons")
            
        else:
            logger.error("AUCell output not found. Please check the SCENIC pipeline execution.")

        logger.info("=" * 60)
        logger.info("SCENIC full workflow completed!")
        
    except Exception as e:
        logger.error(f"ERROR in main execution: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()