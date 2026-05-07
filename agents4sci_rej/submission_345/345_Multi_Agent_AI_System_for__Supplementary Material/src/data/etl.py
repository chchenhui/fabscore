"""
ETL module for loading and validating CSV data files.
Provides schema validation and clean data loading for the forecast pipeline.
"""
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected schemas for validation
SCHEMAS = {
    'epi': {
        'required_columns': ['year', 'prevalence', 'exacerbation_rate', 'eligible_neb_estimate'],
        'dtypes': {'year': 'int64', 'prevalence': 'float64', 'exacerbation_rate': 'float64', 'eligible_neb_estimate': 'float64'}
    },
    'analogs': {
        'required_columns': ['drug', 'launch_qtr', 'p_init', 'q_init', 'price_band', 'access_tier'],
        'dtypes': {'drug': 'object', 'launch_qtr': 'object', 'p_init': 'float64', 'q_init': 'float64', 'price_band': 'object', 'access_tier': 'object'}
    },
    'price_assumptions': {
        'required_columns': ['access_tier', 'list_price_month', 'gtn_pct', 'adoption_ceiling'],
        'dtypes': {'access_tier': 'object', 'list_price_month': 'float64', 'gtn_pct': 'float64', 'adoption_ceiling': 'float64'}
    },
    'payer_rules': {
        'required_columns': ['payer', 'rule_text', 'implied_tier'],
        'dtypes': {'payer': 'object', 'rule_text': 'object', 'implied_tier': 'object'}
    }
}

def load_config(config_path: str = "conf/params.yml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def validate_schema(df: pd.DataFrame, schema_name: str) -> bool:
    """Validate DataFrame against expected schema."""
    if schema_name not in SCHEMAS:
        raise ValueError(f"Unknown schema: {schema_name}")
    
    schema = SCHEMAS[schema_name]
    
    # Check required columns
    missing_cols = set(schema['required_columns']) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for {schema_name}: {missing_cols}")
    
    # Check data types (basic validation)
    for col, expected_dtype in schema['dtypes'].items():
        if col in df.columns:
            if expected_dtype == 'float64' and not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} should be numeric but got {df[col].dtype}")
            elif expected_dtype == 'int64' and not pd.api.types.is_integer_dtype(df[col]):
                raise ValueError(f"Column {col} should be integer but got {df[col].dtype}")
    
    logger.info(f"Schema validation passed for {schema_name}")
    return True

def load_csv_with_validation(file_path: str, schema_name: str) -> pd.DataFrame:
    """Load CSV file and validate against schema."""
    try:
        df = pd.read_csv(file_path)
        validate_schema(df, schema_name)
        logger.info(f"Successfully loaded and validated {file_path} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise

def load_epidemiology_data(data_dir: str = "data_raw") -> pd.DataFrame:
    """Load epidemiology data with schema validation."""
    # Find the most recent epi file
    data_path = Path(data_dir)
    epi_files = list(data_path.glob("*epi_us_copd.csv"))
    
    if not epi_files:
        raise FileNotFoundError(f"No epidemiology files found in {data_dir}")
    
    # Use the most recent file (assuming date prefix)
    latest_file = sorted(epi_files)[-1]
    return load_csv_with_validation(str(latest_file), 'epi')

def load_analog_data(data_dir: str = "data_raw") -> pd.DataFrame:
    """Load analog drug data with schema validation."""
    data_path = Path(data_dir)
    analog_files = list(data_path.glob("*analogs_resp.csv"))
    
    if not analog_files:
        raise FileNotFoundError(f"No analog files found in {data_dir}")
    
    latest_file = sorted(analog_files)[-1]
    return load_csv_with_validation(str(latest_file), 'analogs')

def load_price_assumptions(data_dir: str = "data_raw") -> pd.DataFrame:
    """Load price assumption data with schema validation."""
    data_path = Path(data_dir)
    price_files = list(data_path.glob("*price_assumptions.csv"))
    
    if not price_files:
        raise FileNotFoundError(f"No price assumption files found in {data_dir}")
    
    latest_file = sorted(price_files)[-1]
    return load_csv_with_validation(str(latest_file), 'price_assumptions')

def load_payer_rules(data_dir: str = "data_raw") -> pd.DataFrame:
    """Load payer rules data with schema validation."""
    data_path = Path(data_dir)
    payer_files = list(data_path.glob("*payer_rules_snippets.csv"))
    
    if not payer_files:
        raise FileNotFoundError(f"No payer rules files found in {data_dir}")
    
    latest_file = sorted(payer_files)[-1]
    return load_csv_with_validation(str(latest_file), 'payer_rules')

def save_processed_data(df: pd.DataFrame, filename: str, output_dir: str = "data_proc") -> None:
    """Save processed data to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filepath = output_path / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Saved processed data to {filepath} ({len(df)} rows)")

def load_all_data(data_dir: str = "data_raw") -> Dict[str, pd.DataFrame]:
    """Load all required data files."""
    data = {}
    
    try:
        data['epidemiology'] = load_epidemiology_data(data_dir)
        data['analogs'] = load_analog_data(data_dir)
        data['price_assumptions'] = load_price_assumptions(data_dir)
        data['payer_rules'] = load_payer_rules(data_dir)
        
        logger.info("Successfully loaded all data files")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

if __name__ == "__main__":
    # Demo usage
    config = load_config()
    data = load_all_data()
    
    print("Loaded data shapes:")
    for name, df in data.items():
        print(f"  {name}: {df.shape}")
