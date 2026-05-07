"""
Validation script to verify all visualization outputs were created successfully.
"""

import os
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_outputs():
    """Validate all generated visualization outputs."""
    logger.info("Starting output validation...")
    
    # Auto-detect paths
    if Path("/research_storage/outputs/figures").exists():
        # Modal environment
        base_path = "/research_storage/outputs"
    else:
        # Local environment
        base_path = "idea_14_workspace/outputs"
    
    # Expected output files
    expected_files = {
        # Figures
        f"{base_path}/figures/figure_1_auroc_comparison.png": {"type": "image", "min_size": 10000, "max_size": 5000000},
        f"{base_path}/figures/figure_2_se_vs_length.png": {"type": "image", "min_size": 10000, "max_size": 5000000},
        f"{base_path}/figures/figure_3_hyperparameter_brittleness.png": {"type": "image", "min_size": 10000, "max_size": 5000000},
        f"{base_path}/figures/figure_4_fn_breakdown.png": {"type": "image", "min_size": 10000, "max_size": 5000000},
        
        # Tables
        f"{base_path}/tables/table_2_fnr_comparison.csv": {"type": "table", "min_rows": 10, "min_cols": 5},
        f"{base_path}/tables/table_2_fnr_comparison.md": {"type": "markdown", "min_size": 1000, "max_size": 50000},
        
        # Intermediate data files
        f"{base_path}/visualisation/temp/f1_data.csv": {"type": "table", "min_rows": 5, "min_cols": 5},
        f"{base_path}/visualisation/temp/t2_data.csv": {"type": "table", "min_rows": 10, "min_cols": 5},
        f"{base_path}/visualisation/temp/f3_data.csv": {"type": "table", "min_rows": 8, "min_cols": 3},
        f"{base_path}/visualisation/temp/f4_data.csv": {"type": "table", "min_rows": 4, "min_cols": 3},
    }
    
    validation_results = []
    all_valid = True
    
    for file_path, specs in expected_files.items():
        result = validate_file(file_path, specs)
        validation_results.append(result)
        if not result["valid"]:
            all_valid = False
    
    # Print validation results
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS")
    logger.info("="*60)
    
    for result in validation_results:
        status = "✓ PASS" if result["valid"] else "✗ FAIL"
        logger.info(f"{status} | {result['file']}")
        if not result["valid"]:
            logger.error(f"      Error: {result['error']}")
    
    logger.info("="*60)
    
    if all_valid:
        logger.info("🎉 All visualization artifacts generated and passed validation!")
        return True
    else:
        failed_count = sum(1 for r in validation_results if not r["valid"])
        logger.error(f"❌ {failed_count}/{len(validation_results)} files failed validation")
        return False


def validate_file(file_path, specs):
    """Validate a single file based on its specifications."""
    result = {"file": file_path, "valid": False, "error": None}
    
    try:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            result["error"] = "File does not exist"
            return result
        
        # Check file size
        file_size = path.stat().st_size
        
        if specs["type"] == "image":
            if file_size < specs["min_size"]:
                result["error"] = f"File too small ({file_size} bytes < {specs['min_size']})"
                return result
            if file_size > specs["max_size"]:
                result["error"] = f"File too large ({file_size} bytes > {specs['max_size']})"
                return result
        
        elif specs["type"] == "table":
            # Validate CSV files
            try:
                df = pd.read_csv(path)
                if len(df) < specs["min_rows"]:
                    result["error"] = f"Too few rows ({len(df)} < {specs['min_rows']})"
                    return result
                if len(df.columns) < specs["min_cols"]:
                    result["error"] = f"Too few columns ({len(df.columns)} < {specs['min_cols']})"
                    return result
            except Exception as e:
                result["error"] = f"Failed to read CSV: {e}"
                return result
        
        elif specs["type"] == "markdown":
            if file_size < specs["min_size"]:
                result["error"] = f"File too small ({file_size} bytes < {specs['min_size']})"
                return result
            if file_size > specs["max_size"]:
                result["error"] = f"File too large ({file_size} bytes > {specs['max_size']})"
                return result
            
            # Check if it's readable text
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        result["error"] = "File is empty"
                        return result
            except Exception as e:
                result["error"] = f"Failed to read markdown: {e}"
                return result
        
        result["valid"] = True
        
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"
    
    return result


if __name__ == "__main__":
    success = validate_outputs()
    exit(0 if success else 1)