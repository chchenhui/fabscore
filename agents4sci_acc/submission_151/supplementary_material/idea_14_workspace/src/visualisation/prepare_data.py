"""
Data preparation script for aggregating experiment results for visualization.
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from visualisation.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparator:
    """Prepares and aggregates data for visualization."""
    
    def __init__(self, output_dir: str = "idea_14_workspace/outputs/visualisation/temp"):
        """Initialize the data preparator."""
        self.loader = DataLoader()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DataPreparator with output dir: {self.output_dir}")
    
    def prepare_f1_data(self) -> pd.DataFrame:
        """Prepare data for Figure 1: AUROC Comparison on JailbreakBench."""
        logger.info("Preparing F1 data (AUROC comparison)...")
        
        h1_data = self.loader.load_h1_results()
        
        records = []
        
        for model_key, model_name in [('llama', 'Llama-4-Scout'), ('qwen', 'Qwen-2.5-7B')]:
            model_data = h1_data[model_key]
            
            # Process baseline methods
            for method in ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']:
                if method in model_data and 'auroc' in model_data[method]:
                    records.append({
                        'Model': model_name,
                        'Dataset': 'JailbreakBench',
                        'Method': method,
                        'Metric': 'AUROC',
                        'Value': model_data[method]['auroc'],
                        'tau': None,
                        'N': 5
                    })
            
            # Process Semantic Entropy (find best tau)
            if 'semantic_entropy' in model_data and 'tau_results' in model_data['semantic_entropy']:
                best_auroc = 0
                best_tau = None
                for tau, tau_data in model_data['semantic_entropy']['tau_results'].items():
                    if 'auroc' in tau_data and tau_data['auroc'] > best_auroc:
                        best_auroc = tau_data['auroc']
                        best_tau = tau
                
                if best_tau:
                    records.append({
                        'Model': model_name,
                        'Dataset': 'JailbreakBench',
                        'Method': 'semantic_entropy',
                        'Metric': 'AUROC',
                        'Value': best_auroc,
                        'tau': best_tau,
                        'N': 5
                    })
        
        df = pd.DataFrame(records)
        output_path = self.output_dir / "f1_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"F1 data saved to {output_path} with {len(df)} records")
        
        return df
    
    def prepare_t2_data(self) -> pd.DataFrame:
        """Prepare data for Table 2: FNR@5%FPR comparison."""
        logger.info("Preparing T2 data (FNR comparison table)...")
        
        h1_data = self.loader.load_h1_results()
        h2_data = self.loader.load_h2_results()
        
        records = []
        
        # Process H1 (JailbreakBench) data
        for model_key, model_name in [('llama', 'Llama-4-Scout'), ('qwen', 'Qwen-2.5-7B')]:
            model_data = h1_data[model_key]
            
            # Process baselines
            for method in ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']:
                if method in model_data:
                    method_data = model_data[method]
                    records.append({
                        'Model': model_name,
                        'Dataset': 'JailbreakBench',
                        'Method': method,
                        'FNR': method_data.get('fnr_at_target_fpr', None),
                        'actual_fpr': method_data.get('fpr_used', None),
                        'threshold': method_data.get('optimal_threshold', None),
                        'tau': None
                    })
            
            # Process SE - canonical tau=0.2
            if 'semantic_entropy' in model_data and 'tau_results' in model_data['semantic_entropy']:
                if '0.2' in model_data['semantic_entropy']['tau_results']:
                    tau_data = model_data['semantic_entropy']['tau_results']['0.2']
                    records.append({
                        'Model': model_name,
                        'Dataset': 'JailbreakBench',
                        'Method': 'semantic_entropy (τ=0.2)',
                        'FNR': tau_data.get('fnr', None),
                        'actual_fpr': tau_data.get('fpr_used', None),
                        'threshold': tau_data.get('threshold', None),
                        'tau': '0.2'
                    })
                
                # Also add best tau
                best_fnr = 1.0
                best_tau = None
                best_data = None
                for tau, tau_data in model_data['semantic_entropy']['tau_results'].items():
                    if 'fnr' in tau_data and tau_data['fnr'] < best_fnr:
                        best_fnr = tau_data['fnr']
                        best_tau = tau
                        best_data = tau_data
                
                if best_tau and best_tau != '0.2':
                    records.append({
                        'Model': model_name,
                        'Dataset': 'JailbreakBench',
                        'Method': f'semantic_entropy (best τ={best_tau})',
                        'FNR': best_data.get('fnr', None),
                        'actual_fpr': best_data.get('fpr_used', None),
                        'threshold': best_data.get('threshold', None),
                        'tau': best_tau
                    })
        
        # Process H2 (HarmBench) data
        for model_key, model_name in [('llama', 'Llama-4-Scout'), ('qwen', 'Qwen-2.5-7B')]:
            model_data = h2_data[model_key]
            
            # Process baselines
            if 'baseline_results' in model_data:
                for method in ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']:
                    if method in model_data['baseline_results']:
                        method_data = model_data['baseline_results'][method]
                        records.append({
                            'Model': model_name,
                            'Dataset': 'HarmBench',
                            'Method': method,
                            'FNR': method_data.get('fnr_at_target_fpr', None),
                            'actual_fpr': method_data.get('actual_fpr', None),
                            'threshold': method_data.get('threshold', None),
                            'tau': None
                        })
            
            # Process SE - canonical tau=0.2
            if 'semantic_entropy_results' in model_data:
                if 'tau_0.2' in model_data['semantic_entropy_results']:
                    tau_data = model_data['semantic_entropy_results']['tau_0.2']
                    records.append({
                        'Model': model_name,
                        'Dataset': 'HarmBench',
                        'Method': 'semantic_entropy (τ=0.2)',
                        'FNR': tau_data.get('fnr_at_target_fpr', None),
                        'actual_fpr': tau_data.get('actual_fpr', None),
                        'threshold': tau_data.get('threshold', None),
                        'tau': '0.2'
                    })
                
                # Also add best tau
                best_fnr = 1.0
                best_tau = None
                best_data = None
                for tau_key, tau_data in model_data['semantic_entropy_results'].items():
                    if tau_key.startswith('tau_'):
                        tau_val = tau_key.split('_')[1]
                        if 'fnr_at_target_fpr' in tau_data and tau_data['fnr_at_target_fpr'] < best_fnr:
                            best_fnr = tau_data['fnr_at_target_fpr']
                            best_tau = tau_val
                            best_data = tau_data
                
                if best_tau and best_tau != '0.2':
                    records.append({
                        'Model': model_name,
                        'Dataset': 'HarmBench',
                        'Method': f'semantic_entropy (best τ={best_tau})',
                        'FNR': best_data.get('fnr_at_target_fpr', None),
                        'actual_fpr': best_data.get('actual_fpr', None),
                        'threshold': best_data.get('threshold', None),
                        'tau': best_tau
                    })
        
        df = pd.DataFrame(records)
        output_path = self.output_dir / "t2_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"T2 data saved to {output_path} with {len(df)} records")
        
        return df
    
    def prepare_f3_data(self) -> pd.DataFrame:
        """Prepare data for Figure 3: FNR vs. Hyperparameters."""
        logger.info("Preparing F3 data (hyperparameter brittleness)...")
        
        h4_data = self.loader.load_h4_brittleness_results()
        h2_data = self.loader.load_h2_results()
        
        records = []
        
        # Extract performance matrix data
        if 'performance_matrix' in h4_data:
            perf_matrix = h4_data['performance_matrix']
            
            for key, data in perf_matrix.items():
                # Parse tau and n from key (e.g., "tau_0.1_n_5")
                parts = key.split('_')
                tau = parts[1]
                n = parts[3]
                
                records.append({
                    'tau': float(tau),
                    'N': int(n),
                    'Method': f'SE (N={n})',
                    'FNR': data.get('fnr_at_5fpr', None)
                })
        
        # Note: Embedding Variance removed - it doesn't use tau parameter and 
        # would be scientifically incorrect to plot vs tau values
        
        df = pd.DataFrame(records)
        output_path = self.output_dir / "f3_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"F3 data saved to {output_path} with {len(df)} records")
        
        return df
    
    def prepare_f4_data(self) -> pd.DataFrame:
        """Prepare data for Figure 4: Breakdown of False Negative Causes."""
        logger.info("Preparing F4 data (false negative breakdown)...")
        
        h6_data = self.loader.load_h6_audit_results()
        
        records = []
        
        # Process Llama @ JBB
        llama_data = h6_data['llama_jbb']
        if 'tau_specific_results' in llama_data and '0.2' in llama_data['tau_specific_results']:
            total_fn = llama_data['tau_specific_results']['0.2'].get('n_false_negatives', 0)
            
            # Count consistency confound cases
            confound_count = 0
            if 'false_negative_analysis' in llama_data:
                for fn_case in llama_data['false_negative_analysis']:
                    if 0.2 in fn_case.get('appears_in_taus', []):
                        if fn_case.get('classification') == 'consistency_confound':
                            confound_count += 1
            
            other_count = total_fn - confound_count
            
            records.append({
                'Experiment': 'Llama @ JBB',
                'Cause': 'Consistency Confound',
                'Count': confound_count
            })
            records.append({
                'Experiment': 'Llama @ JBB',
                'Cause': 'Other',
                'Count': other_count
            })
            
            logger.info(f"Llama @ JBB: Total FN={total_fn}, Confound={confound_count}, Other={other_count}")
        
        # Process Qwen @ HBC
        qwen_data = h6_data['qwen_hbc']
        if 'tau_specific_results' in qwen_data and '0.2' in qwen_data['tau_specific_results']:
            total_fn = qwen_data['tau_specific_results']['0.2'].get('n_false_negatives', 0)
            
            # Count consistency confound cases
            confound_count = 0
            if 'false_negative_analysis' in qwen_data:
                for fn_case in qwen_data['false_negative_analysis']:
                    if 0.2 in fn_case.get('appears_in_taus', []):
                        if fn_case.get('classification') == 'consistency_confound':
                            confound_count += 1
            
            other_count = total_fn - confound_count
            
            records.append({
                'Experiment': 'Qwen @ HBC',
                'Cause': 'Consistency Confound',
                'Count': confound_count
            })
            records.append({
                'Experiment': 'Qwen @ HBC',
                'Cause': 'Other',
                'Count': other_count
            })
            
            logger.info(f"Qwen @ HBC: Total FN={total_fn}, Confound={confound_count}, Other={other_count}")
        
        df = pd.DataFrame(records)
        output_path = self.output_dir / "f4_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"F4 data saved to {output_path} with {len(df)} records")
        
        return df
    
    def prepare_f1h_data(self) -> pd.DataFrame:
        """Prepare data for Figure 1H: AUROC Comparison on HarmBench."""
        logger.info("Preparing F1H data (AUROC comparison on HarmBench)...")
        
        h2_data = self.loader.load_h2_results()
        
        records = []
        
        for model_key, model_name in [('llama', 'Llama-4-Scout'), ('qwen', 'Qwen-2.5-7B')]:
            model_data = h2_data[model_key]
            
            # Process baseline methods
            if 'baseline_results' in model_data:
                for method in ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']:
                    if method in model_data['baseline_results']:
                        method_data = model_data['baseline_results'][method]
                        if 'auroc' in method_data:
                            records.append({
                                'Model': model_name,
                                'Dataset': 'HarmBench',
                                'Method': method,
                                'Metric': 'AUROC',
                                'Value': method_data['auroc'],
                                'tau': None,
                                'N': 5
                            })
            
            # Process Semantic Entropy (find best tau for AUROC)
            if 'semantic_entropy_results' in model_data:
                best_auroc = 0
                best_tau = None
                for tau_key, tau_data in model_data['semantic_entropy_results'].items():
                    if tau_key.startswith('tau_'):
                        tau_val = tau_key.split('_')[1]
                        if 'auroc' in tau_data and tau_data['auroc'] > best_auroc:
                            best_auroc = tau_data['auroc']
                            best_tau = tau_val
                
                if best_tau:
                    records.append({
                        'Model': model_name,
                        'Dataset': 'HarmBench', 
                        'Method': 'semantic_entropy',
                        'Metric': 'AUROC',
                        'Value': best_auroc,
                        'tau': best_tau,
                        'N': 5
                    })
        
        df = pd.DataFrame(records)
        output_path = self.output_dir / "f1h_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"F1H data saved to {output_path} with {len(df)} records")
        
        return df
    
    def prepare_f1c_data(self) -> pd.DataFrame:
        """Prepare data for Figure 1C: Comprehensive AUROC across both datasets."""
        logger.info("Preparing F1C data (comprehensive AUROC comparison)...")
        
        h1_data = self.loader.load_h1_results()
        h2_data = self.loader.load_h2_results()
        
        records = []
        
        # Process H1 (JailbreakBench) data
        for model_key, model_name in [('llama', 'Llama-4-Scout'), ('qwen', 'Qwen-2.5-7B')]:
            model_data = h1_data[model_key]
            
            # Process baseline methods
            for method in ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']:
                if method in model_data and 'auroc' in model_data[method]:
                    records.append({
                        'Model': model_name,
                        'Dataset': 'JailbreakBench',
                        'Method': method,
                        'Metric': 'AUROC',
                        'Value': model_data[method]['auroc'],
                        'tau': None,
                        'N': 5
                    })
            
            # Process Semantic Entropy (find best tau)
            if 'semantic_entropy' in model_data and 'tau_results' in model_data['semantic_entropy']:
                best_auroc = 0
                best_tau = None
                for tau, tau_data in model_data['semantic_entropy']['tau_results'].items():
                    if 'auroc' in tau_data and tau_data['auroc'] > best_auroc:
                        best_auroc = tau_data['auroc']
                        best_tau = tau
                
                if best_tau:
                    records.append({
                        'Model': model_name,
                        'Dataset': 'JailbreakBench',
                        'Method': 'semantic_entropy',
                        'Metric': 'AUROC',
                        'Value': best_auroc,
                        'tau': best_tau,
                        'N': 5
                    })
        
        # Process H2 (HarmBench) data  
        for model_key, model_name in [('llama', 'Llama-4-Scout'), ('qwen', 'Qwen-2.5-7B')]:
            model_data = h2_data[model_key]
            
            # Process baseline methods
            if 'baseline_results' in model_data:
                for method in ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']:
                    if method in model_data['baseline_results']:
                        method_data = model_data['baseline_results'][method]
                        if 'auroc' in method_data:
                            records.append({
                                'Model': model_name,
                                'Dataset': 'HarmBench',
                                'Method': method,
                                'Metric': 'AUROC',
                                'Value': method_data['auroc'],
                                'tau': None,
                                'N': 5
                            })
            
            # Process Semantic Entropy (find best tau for AUROC)
            if 'semantic_entropy_results' in model_data:
                best_auroc = 0
                best_tau = None
                for tau_key, tau_data in model_data['semantic_entropy_results'].items():
                    if tau_key.startswith('tau_'):
                        tau_val = tau_key.split('_')[1]
                        if 'auroc' in tau_data and tau_data['auroc'] > best_auroc:
                            best_auroc = tau_data['auroc']
                            best_tau = tau_val
                
                if best_tau:
                    records.append({
                        'Model': model_name,
                        'Dataset': 'HarmBench', 
                        'Method': 'semantic_entropy',
                        'Metric': 'AUROC',
                        'Value': best_auroc,
                        'tau': best_tau,
                        'N': 5
                    })
        
        df = pd.DataFrame(records)
        output_path = self.output_dir / "f1c_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"F1C data saved to {output_path} with {len(df)} records")
        
        return df
    
    def prepare_f5_data(self) -> pd.DataFrame:
        """Prepare data for Figure 5: Paraphrase Impact Comparison (JBB vs JBB Paraphrase)."""
        logger.info("Preparing F5 data (paraphrase impact comparison)...")
        
        h5_data = self.loader.load_h5_results()
        
        records = []
        
        # Extract comparison data for both models
        for model_key in ['Llama-4-Scout', 'Qwen-2.5-7B']:
            model_data = h5_data['all_model_results'][model_key]
            
            # Extract baseline method degradations (FNR shows most impact)
            for method, degradation in model_data['baseline_degradations'].items():
                fnr_deg = degradation['fnr_deg']
                
                # Convert to percentage points for better readability
                fnr_change_pp = fnr_deg * 100
                
                # Get method display name
                if method == 'avg_pairwise_bertscore':
                    method_display = 'BERTScore'
                elif method == 'embedding_variance':
                    method_display = 'Embedding Variance'
                elif method == 'levenshtein_variance':
                    method_display = 'Levenshtein Variance'
                else:
                    method_display = method
                
                records.append({
                    'Model': model_key,
                    'Method': method_display,
                    'FNR_Change_pp': fnr_change_pp,
                    'Original_FNR': model_data['h1_metrics'][method]['fnr_at_5fpr'],
                    'Paraphrase_FNR': model_data['h5_metrics'][method]['fnr_at_5fpr']
                })
            
            # Add Semantic Entropy at canonical tau=0.2 (if available)
            if 'se_tau_0.2' in model_data['h1_metrics'] and 'se_tau_0.2' in model_data['h5_metrics']:
                se_fnr_deg = model_data['degradation']['se_tau_0.2']['fnr_degradation']
                se_fnr_change_pp = se_fnr_deg * 100
                
                records.append({
                    'Model': model_key,
                    'Method': 'Semantic Entropy (τ=0.2)',
                    'FNR_Change_pp': se_fnr_change_pp,
                    'Original_FNR': model_data['h1_metrics']['se_tau_0.2']['fnr_at_5fpr'],
                    'Paraphrase_FNR': model_data['h5_metrics']['se_tau_0.2']['fnr_at_5fpr']
                })
            
            # Add SE at best tau (tau=0.1 for both models based on data)
            if 'se_tau_0.1' in model_data['h1_metrics'] and 'se_tau_0.1' in model_data['h5_metrics']:
                se_fnr_deg = model_data['degradation']['se_tau_0.1']['fnr_degradation']
                se_fnr_change_pp = se_fnr_deg * 100
                
                records.append({
                    'Model': model_key,
                    'Method': 'Semantic Entropy (τ=0.1)',
                    'FNR_Change_pp': se_fnr_change_pp,
                    'Original_FNR': model_data['h1_metrics']['se_tau_0.1']['fnr_at_5fpr'],
                    'Paraphrase_FNR': model_data['h5_metrics']['se_tau_0.1']['fnr_at_5fpr']
                })
        
        df = pd.DataFrame(records)
        output_path = self.output_dir / "f5_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"F5 data saved to {output_path} with {len(df)} records")
        
        return df
    
    def prepare_f2c_data(self) -> pd.DataFrame:
        """Prepare data for Figure 2C: 2x2 grid comparing SE at different tau values for both models on HarmBench."""
        logger.info("Preparing F2C data (SE vs length at different tau values)...")
        
        import json
        
        records = []
        
        # Load H3 data for both models on HarmBench
        models = [
            ('llama', 'Llama-4-Scout', 'llama-4-scout-17b-16e-instruct_H2_h3_prompt_analysis.jsonl'),
            ('qwen', 'Qwen-2.5-7B', 'qwen2.5-7b-instruct_H2_h3_prompt_analysis.jsonl')
        ]
        
        for model_key, model_name, filename in models:
            filepath = self.loader.base_path / f"h3/per_prompt_analysis/{filename}"
            
            if filepath.exists():
                with open(filepath, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        
                        # Extract data for different tau values
                        for tau in ['0.1', '0.2']:
                            records.append({
                                'Model': model_name,
                                'tau': float(tau),  # Convert to float for consistency
                                'prompt_id': data['prompt_id'],
                                'label': data['label'],
                                'log_length': data['log_length'],
                                'se_score': data[f'original_se_tau_{tau}']
                            })
                
                logger.info(f"Loaded {model_name} data from {filepath}")
            else:
                logger.warning(f"File not found: {filepath}")
        
        df = pd.DataFrame(records)
        output_path = self.output_dir / "f2c_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"F2C data saved to {output_path} with {len(df)} records")
        
        return df
    
    def prepare_all_data(self):
        """Prepare all data files for visualization."""
        logger.info("Preparing all visualization data...")
        
        self.prepare_f1_data()
        self.prepare_f1h_data()
        self.prepare_f1c_data()
        self.prepare_t2_data()
        self.prepare_f2c_data()
        self.prepare_f3_data()
        self.prepare_f4_data()
        self.prepare_f5_data()
        
        logger.info("All data preparation complete!")


if __name__ == "__main__":
    preparator = DataPreparator()
    preparator.prepare_all_data()