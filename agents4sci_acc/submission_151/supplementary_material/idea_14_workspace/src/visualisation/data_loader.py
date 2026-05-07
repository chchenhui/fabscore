"""
Data loader module for visualization pipeline.
Handles loading and validation of experiment result files.
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and validation of experiment data files."""
    
    def __init__(self, base_path: str = None):
        """
        Initialize the data loader.
        
        Args:
            base_path: Base path to the outputs directory
        """
        # Auto-detect path based on environment
        if base_path is None:
            if Path("/workspace/outputs").exists():
                # Modal environment
                base_path = "/workspace/outputs"
            elif Path("idea_14_workspace/outputs").exists():
                # Local environment
                base_path = "idea_14_workspace/outputs"
            else:
                # Default fallback
                base_path = "idea_14_workspace/outputs"
        
        self.base_path = Path(base_path)
        logger.info(f"Initialized DataLoader with base path: {self.base_path}")
        
        if not self.base_path.exists():
            logger.critical(f"Base path does not exist: {self.base_path}")
            sys.exit(1)
    
    def load_json(self, file_path: Union[str, Path], required_fields: List[str] = None) -> Dict[str, Any]:
        """
        Load and validate a JSON file.
        
        Args:
            file_path: Path to the JSON file
            required_fields: List of required top-level fields
            
        Returns:
            Dictionary containing the loaded JSON data
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        
        # Check if file exists
        if not file_path.exists():
            logger.critical(f"File not found: {file_path}")
            sys.exit(1)
        
        logger.info(f"Loading JSON file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.critical(f"Failed to parse JSON from {file_path}: {e}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Error reading file {file_path}: {e}")
            sys.exit(1)
        
        # Validate required fields
        if required_fields:
            missing_fields = []
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)
            
            if missing_fields:
                logger.warning(f"Missing required fields in {file_path}: {missing_fields}")
        
        logger.info(f"Successfully loaded {file_path} with {len(data)} top-level keys")
        return data
    
    def load_jsonl(self, file_path: Union[str, Path], required_fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        Load and validate a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            required_fields: List of required fields in each record
            
        Returns:
            List of dictionaries containing the loaded JSONL data
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        
        # Check if file exists
        if not file_path.exists():
            logger.critical(f"File not found: {file_path}")
            sys.exit(1)
        
        logger.info(f"Loading JSONL file: {file_path}")
        
        records = []
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        
                        # Validate required fields for this record
                        if required_fields:
                            missing_fields = [field for field in required_fields if field not in record]
                            if missing_fields:
                                logger.warning(f"Line {line_num}: Missing fields {missing_fields}")
                        
                        records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Failed to parse JSON: {e}")
                        continue
        except Exception as e:
            logger.critical(f"Error reading file {file_path}: {e}")
            sys.exit(1)
        
        logger.info(f"Successfully loaded {len(records)} records from {file_path}")
        return records
    
    def validate_metric_range(self, value: float, metric_name: str, valid_range: tuple = (0.0, 1.0)) -> bool:
        """
        Validate that a metric value is within expected range.
        
        Args:
            value: The metric value to validate
            metric_name: Name of the metric for logging
            valid_range: Tuple of (min, max) valid values
            
        Returns:
            True if valid, False otherwise
        """
        if value < valid_range[0] or value > valid_range[1]:
            logger.warning(f"{metric_name} value {value} outside valid range {valid_range}")
            return False
        return True
    
    def load_h1_results(self) -> Dict[str, Dict]:
        """Load H1 (JailbreakBench) evaluation results for both models."""
        logger.info("Loading H1 results...")
        
        h1_data = {}
        
        # Load Llama results
        llama_path = self.base_path / "h1/evaluation/llama4scout_120val_results.json"
        llama_data = self.load_json(llama_path, required_fields=[
            'avg_pairwise_bertscore', 'embedding_variance', 
            'levenshtein_variance', 'semantic_entropy'
        ])
        h1_data['llama'] = llama_data
        
        # Load Qwen results
        qwen_path = self.base_path / "h1/evaluation/qwen25_120val_results.json"
        qwen_data = self.load_json(qwen_path, required_fields=[
            'avg_pairwise_bertscore', 'embedding_variance',
            'levenshtein_variance', 'semantic_entropy'
        ])
        h1_data['qwen'] = qwen_data
        
        # Validate AUROC values
        for model_name, model_data in h1_data.items():
            for method in ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']:
                if method in model_data and 'auroc' in model_data[method]:
                    self.validate_metric_range(
                        model_data[method]['auroc'], 
                        f"{model_name}:{method}:auroc"
                    )
            
            # Validate SE tau results
            if 'semantic_entropy' in model_data and 'tau_results' in model_data['semantic_entropy']:
                for tau, tau_data in model_data['semantic_entropy']['tau_results'].items():
                    if 'auroc' in tau_data:
                        self.validate_metric_range(
                            tau_data['auroc'],
                            f"{model_name}:SE:tau_{tau}:auroc"
                        )
        
        return h1_data
    
    def load_h2_results(self) -> Dict[str, Dict]:
        """Load H2 (HarmBench) evaluation results for both models."""
        logger.info("Loading H2 results...")
        
        h2_data = {}
        
        # Load Llama results
        llama_path = self.base_path / "h2/evaluation/llama-4-scout-17b-16e-instruct_h2_results.json"
        llama_data = self.load_json(llama_path, required_fields=[
            'model', 'dataset_composition', 'semantic_entropy_results',
            'baseline_results'
        ])
        h2_data['llama'] = llama_data
        
        # Load Qwen results  
        qwen_path = self.base_path / "h2/evaluation/qwen2.5-7b-instruct_h2_results.json"
        qwen_data = self.load_json(qwen_path, required_fields=[
            'model', 'dataset_composition', 'semantic_entropy_results',
            'baseline_results'
        ])
        h2_data['qwen'] = qwen_data
        
        return h2_data
    
    def load_h3_prompt_analysis(self, model: str = 'llama') -> pd.DataFrame:
        """Load H3 per-prompt analysis data."""
        logger.info(f"Loading H3 prompt analysis for {model}...")
        
        if model == 'llama':
            file_path = self.base_path / "h3/per_prompt_analysis/llama-4-scout-17b-16e-instruct_H2_h3_prompt_analysis.jsonl"
        else:
            file_path = self.base_path / "h3/per_prompt_analysis/qwen2.5-7b-instruct_H2_h3_prompt_analysis.jsonl"
        
        records = self.load_jsonl(file_path, required_fields=['log_length', 'original_se_tau_0.1', 'label'])
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Validate data
        if len(df) != 162:
            logger.warning(f"Expected 162 records but got {len(df)}")
        
        # Validate label values
        if not df['label'].isin([0, 1]).all():
            logger.warning("Found labels outside of [0, 1]")
        
        # Validate log_length > 0
        if (df['log_length'] <= 0).any():
            logger.warning("Found non-positive log_length values")
        
        # Validate SE >= 0
        if (df['original_se_tau_0.1'] < 0).any():
            logger.warning("Found negative SE values")
        
        logger.info(f"Loaded {len(df)} prompt analysis records")
        return df
    
    def load_h4_brittleness_results(self) -> Dict[str, Any]:
        """Load H4 brittleness analysis results."""
        logger.info("Loading H4 brittleness results...")
        
        file_path = self.base_path / "h4/evaluation/h4_brittleness_results.json"
        data = self.load_json(file_path, required_fields=['performance_matrix'])
        
        # Validate performance matrix structure
        if 'performance_matrix' in data:
            perf_matrix = data['performance_matrix']
            expected_keys = []
            for tau in ['0.1', '0.2', '0.3', '0.4']:
                for n in ['5', '10']:
                    expected_keys.append(f'tau_{tau}_n_{n}')
            
            missing_keys = [key for key in expected_keys if key not in perf_matrix]
            if missing_keys:
                logger.warning(f"Missing keys in performance matrix: {missing_keys}")
        
        return data
    
    def load_h6_audit_results(self) -> Dict[str, Dict]:
        """Load H6 qualitative audit results."""
        logger.info("Loading H6 audit results...")
        
        h6_data = {}
        
        # Load Llama H1 JailbreakBench audit
        llama_path = self.base_path / "h6/llama-h1-jailbreakbench/llama-4-scout-17b-16e-instruct_H1_h6_qualitative_audit_results.json"
        llama_data = self.load_json(llama_path, required_fields=['tau_specific_results', 'false_negative_analysis'])
        h6_data['llama_jbb'] = llama_data
        
        # Load Qwen H2 HarmBench audit
        qwen_path = self.base_path / "h6/qwen-h2-harmbench/qwen-2.5-7b-instruct_H2_h6_qualitative_audit_results.json"
        qwen_data = self.load_json(qwen_path, required_fields=['tau_specific_results', 'false_negative_analysis'])
        h6_data['qwen_hbc'] = qwen_data
        
        return h6_data
    
    def load_h5_results(self) -> Dict[str, Dict]:
        """Load H5 (Paraphrase robustness) evaluation results."""
        logger.info("Loading H5 paraphrase results...")
        
        h5_path = self.base_path / "h5/evaluation/h5_robustness_evaluation.json"
        h5_data = self.load_json(h5_path, required_fields=[
            'all_model_results', 'evaluation_summary'
        ])
        
        logger.info("H5 data loaded successfully")
        return h5_data


def test_loader():
    """Test the data loader with all required files."""
    logger.info("Testing DataLoader...")
    
    loader = DataLoader()
    
    # Test H1 loading
    h1_data = loader.load_h1_results()
    logger.info(f"H1 data loaded: {list(h1_data.keys())}")
    
    # Test H2 loading
    h2_data = loader.load_h2_results()
    logger.info(f"H2 data loaded: {list(h2_data.keys())}")
    
    # Test H3 loading
    h3_df = loader.load_h3_prompt_analysis('llama')
    logger.info(f"H3 data shape: {h3_df.shape}")
    
    # Test H4 loading
    h4_data = loader.load_h4_brittleness_results()
    logger.info(f"H4 data keys: {list(h4_data.keys())}")
    
    # Test H6 loading
    h6_data = loader.load_h6_audit_results()
    logger.info(f"H6 data loaded: {list(h6_data.keys())}")
    
    logger.info("All data loading tests passed!")


if __name__ == "__main__":
    test_loader()