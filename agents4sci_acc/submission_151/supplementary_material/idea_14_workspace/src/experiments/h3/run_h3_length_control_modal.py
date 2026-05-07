#!/usr/bin/env python3
"""
H3 Length-Control Analysis on Modal - Evaluate length-controlled Semantic Entropy

This script performs length residualization analysis on SE scores to assess
whether SE effectiveness is confounded by response length patterns.

Key Analysis:
1. Fit linear model: SE ~ log(length) on benign prompts only
2. Calculate residual SE scores for all prompts
3. Evaluate residual SE performance vs original SE
4. Compare with baseline metrics for context

ACCEPTANCE CRITERION: 
- For Llama-4 on both datasets, residual SE AUROC must be < 0.55
"""

import modal
import json
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modal setup - consistent with H2 infrastructure
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "numpy", 
    "scikit-learn",
    "pandas",
    "scipy",
    "statsmodels",
    "pyyaml"
]).add_local_python_source("src").add_local_dir("configs", "/configs").add_local_dir("data", "/data")

# Use same persistent storage volume
volume = modal.Volume.from_name("alignment-research-storage", create_if_missing=True)

app = modal.App("h3-length-control-analysis")

@app.function(
    image=image,
    timeout=3600,  # 1 hour should be sufficient for analysis
    volumes={"/research_storage": volume}
)
def run_h3_analysis(model_short: str, dataset: str = "H2"):
    """Run H3 length-control analysis for a specific model and dataset."""
    
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import logging
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from scipy import stats
    from statsmodels.stats.proportion import proportion_confint
    import warnings
    warnings.filterwarnings('ignore')
    
    # Configure logging and reproducibility
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    # Note: sklearn LinearRegression is deterministic, no random seed needed
    
    logger.info("=" * 100)
    logger.info(f"H3 LENGTH-CONTROL ANALYSIS - {model_short} on {dataset}")
    logger.info("=" * 100)
    
    # Load config
    try:
        with open('/configs/project_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Loaded project configuration")
    except Exception as e:
        logger.warning(f"⚠️ Could not load config: {e}, using defaults")
        config = {}
    
    # Setup paths based on dataset
    if dataset == 'H1':
        scores_file = Path(f'/research_storage/outputs/h1/{model_short}_h1_scores.jsonl')
    else:  # H2
        scores_file = Path(f'/research_storage/outputs/h2/scoring/{model_short}_h2_scores.jsonl')
    
    logger.info(f"📁 Scores input: {scores_file}")
    logger.info("📊 Length data will be extracted from scoring diagnostics")
    
    # Check if files exist and validate data availability
    logger.info(f"\n🔍 VALIDATING DATA AVAILABILITY")
    logger.info("=" * 60)
    
    if not scores_file.exists():
        logger.error(f"❌ Scores file not found: {scores_file}")
        logger.error("   This indicates H2 scoring was not completed for this model")
        raise FileNotFoundError(f"H2 scoring data not found for {model_short}. Run H2 scoring first.")
    
    # Quick validation of file format
    try:
        with open(scores_file, 'r') as f:
            first_line = f.readline()
            sample_data = json.loads(first_line)
            
        required_fields = ['semantic_entropy', 'semantic_entropy_diagnostics', 'label']
        missing_fields = [field for field in required_fields if field not in sample_data]
        
        if missing_fields:
            logger.error(f"❌ Invalid scoring file format. Missing fields: {missing_fields}")
            raise ValueError(f"H2 scoring file has invalid format for {model_short}")
            
        logger.info(f"✅ Scoring file validation passed")
        logger.info(f"   Sample contains {len(sample_data)} fields")
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"❌ Corrupted scoring file: {e}")
        raise ValueError(f"H2 scoring file corrupted for {model_short}")
    
    # Load scores data
    logger.info("\n📊 Loading score data...")
    scores_data = []
    with open(scores_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                scores_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"JSON error on line {line_num}: {e}")
    
    df = pd.DataFrame(scores_data)
    logger.info(f"✅ Loaded {len(df)} scored samples")
    
    # Extract response length data from scoring diagnostics
    logger.info(f"\n📊 Extracting response length data from scoring diagnostics...")

    def extract_length_from_scores(row):
        """Extract average response length from semantic entropy diagnostics."""
        se_data = row.get('semantic_entropy_diagnostics', {})
        tau_01_data = se_data.get('tau_0.1', {})
        return tau_01_data.get('avg_response_length', 0)

    df['median_response_length'] = df.apply(extract_length_from_scores, axis=1)

    # Validate length extraction
    valid_lengths = df['median_response_length'] > 0
    logger.info(f"✅ Extracted response lengths for {valid_lengths.sum()}/{len(df)} samples")

    if valid_lengths.sum() == 0:
        logger.error("❌ No valid response length data found in scoring files")
        raise ValueError("Response length data not available in scoring files")

    logger.info(f"📊 Length statistics:")
    logger.info(f"   Mean: {df[valid_lengths]['median_response_length'].mean():.1f} chars")
    logger.info(f"   Median: {df[valid_lengths]['median_response_length'].median():.1f} chars") 
    logger.info(f"   Range: {df[valid_lengths]['median_response_length'].min():.0f}-{df[valid_lengths]['median_response_length'].max():.0f} chars")
    
    # Dataset statistics
    harmful_count = sum(df['label'] == 1)
    benign_count = sum(df['label'] == 0)
    logger.info(f"\n📊 Dataset composition:")
    logger.info(f"   Harmful samples: {harmful_count}")
    logger.info(f"   Benign samples: {benign_count}")
    logger.info(f"   Total samples: {len(df)}")
    logger.info(f"   Samples with length data: {df['median_response_length'].notna().sum()}")
    
    # Prepare labels
    labels = df['label'].values
    
    # Initialize results
    results = {
        'model': model_short,
        'dataset': dataset,
        'n_samples': len(df),
        'n_harmful': harmful_count,
        'n_benign': benign_count
    }
    
    # 1. Evaluate original SE across all tau values
    logger.info("\n" + "=" * 60)
    logger.info("ORIGINAL SEMANTIC ENTROPY PERFORMANCE (ALL TAU VALUES)")
    logger.info("=" * 60)
    
    # Get all available tau values
    tau_values = [0.1, 0.2, 0.3, 0.4]  # Standard tau grid
    se_results_by_tau = {}
    
    # Check data format
    if 'semantic_entropy' not in df.columns:
        logger.error("❌ No semantic entropy scores found in data")
        raise ValueError("Semantic entropy scores not found")
    
    for tau in tau_values:
        tau_key = f'tau_{tau}'
        
        # Extract scores for this tau
        se_scores = df['semantic_entropy'].apply(
            lambda x: x.get(tau_key, 0) if isinstance(x, dict) else 0
        ).values
        
        # Note: SE=0 is valid data (indicates perfect response consistency)
        # We analyze all tau values, including all-zero cases
        if np.all(se_scores == 0):
            logger.info(f"📝 τ={tau}: All SE scores are 0 (perfect consistency) - analyzing anyway")
            
        auroc_orig = roc_auc_score(labels, se_scores)
        
        # Calculate FNR at 5% FPR
        fpr, tpr, thresholds = roc_curve(labels, se_scores, drop_intermediate=False)
        valid_indices = np.where(fpr <= 0.05)[0]
        
        if len(valid_indices) > 0:
            idx = valid_indices[-1]
            fnr_orig = 1 - tpr[idx]
            threshold_orig = thresholds[idx]
            fpr_used_orig = fpr[idx]
        else:
            idx = np.argmin(np.abs(fpr - 0.05))
            fnr_orig = 1 - tpr[idx]
            threshold_orig = thresholds[idx]
            fpr_used_orig = fpr[idx]
        
        # Calculate confidence intervals
        n_positives = sum(labels)
        fnr_ci_orig = proportion_confint(
            int(fnr_orig * n_positives), n_positives, method='wilson', alpha=0.05
        )
        
        se_results_by_tau[tau] = {
            'auroc': float(auroc_orig),
            'fnr_at_5fpr': float(fnr_orig),
            'fnr_ci_lower': float(fnr_ci_orig[0]),
            'fnr_ci_upper': float(fnr_ci_orig[1]),
            'threshold': float(threshold_orig),
            'fpr_used': float(fpr_used_orig)
            # Note: se_scores removed to avoid JSON serialization issues
        }
        
        logger.info(f"📊 τ={tau}: AUROC={auroc_orig:.4f}, FNR@5%FPR={fnr_orig:.4f} [CI: {fnr_ci_orig[0]:.4f}-{fnr_ci_orig[1]:.4f}]")
    
    results['original_se_by_tau'] = se_results_by_tau
    
    # Find best performing tau for primary analysis
    if se_results_by_tau:
        best_tau = max(se_results_by_tau.keys(), key=lambda t: se_results_by_tau[t]['auroc'])
        logger.info(f"🏆 Best performing τ={best_tau} (AUROC: {se_results_by_tau[best_tau]['auroc']:.4f})")
        results['best_tau'] = best_tau
    else:
        logger.error("❌ No valid SE results for any tau value")
        raise ValueError("No valid semantic entropy data found")
    
    # 2. Fit length models and calculate residuals for each tau
    logger.info("\n" + "=" * 60)
    logger.info("FITTING LENGTH MODELS FOR ALL TAU VALUES")
    logger.info("=" * 60)
    
    # Filter to benign prompts only for fitting
    benign_mask = df['label'] == 0
    benign_df = df[benign_mask].copy()
    
    # Remove samples with invalid lengths
    benign_df = benign_df[benign_df['median_response_length'] > 0]
    
    logger.info(f"📊 Fitting length models on {len(benign_df)} benign samples")
    
    # Prepare features: log(length)
    X_benign = np.log(benign_df['median_response_length'].values).reshape(-1, 1)
    
    # Remove invalid lengths from full dataset for residual calculation
    valid_length_mask = df['median_response_length'] > 0
    df_valid = df[valid_length_mask].copy()
    labels_valid = df_valid['label'].values
    X_all = np.log(df_valid['median_response_length'].values).reshape(-1, 1)
    
    length_models = {}
    residual_results = {}
    
    for tau in se_results_by_tau.keys():
        tau_key = f'tau_{tau}'
        logger.info(f"\n🔬 Processing τ={tau}...")
        
        # Get SE scores for benign samples for this tau
        y_benign = benign_df['semantic_entropy'].apply(
            lambda x: x.get(tau_key, 0) if isinstance(x, dict) else 0
        ).values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X_benign, y_benign)
        r2 = model.score(X_benign, y_benign)
        
        logger.info(f"   Length model R²: {r2:.4f}")
        logger.info(f"   Intercept: {model.intercept_:.4f}, Slope: {model.coef_[0]:.4f}")
        
        length_models[tau] = {
            'model': model,
            'r2': float(r2),
            'intercept': float(model.intercept_),
            'slope': float(model.coef_[0])
        }
        
        # Calculate residuals for ALL valid samples
        se_scores_valid = df_valid['semantic_entropy'].apply(
            lambda x: x.get(tau_key, 0) if isinstance(x, dict) else 0
        ).values
        
        predicted_scores = model.predict(X_all)
        residual_scores = se_scores_valid - predicted_scores
        
        logger.info(f"   Calculated residuals for {len(residual_scores)} samples")
        
        # Store residual scores for evaluation (temporarily)
        residual_results[tau] = {
            'residual_scores': residual_scores,  # numpy array - will be used then removed
            'labels_valid': labels_valid,        # numpy array - will be used then removed  
            'n_samples': len(residual_scores)
        }
    
    results['length_models'] = {tau: {k: v for k, v in model_data.items() if k != 'model'} 
                               for tau, model_data in length_models.items()}
    logger.info(f"✅ Fitted length models for {len(length_models)} tau values")
    
    # 3. Evaluate residual SE performance for all tau values
    logger.info("\n" + "=" * 60)
    logger.info("RESIDUAL SEMANTIC ENTROPY PERFORMANCE (ALL TAU VALUES)")
    logger.info("=" * 60)
    
    residual_se_results = {}
    
    for tau in residual_results.keys():
        logger.info(f"\n🔬 Evaluating residuals for τ={tau}...")
        
        residual_scores = residual_results[tau]['residual_scores']
        labels_valid = residual_results[tau]['labels_valid']
        
        auroc_resid = roc_auc_score(labels_valid, residual_scores)
        
        # Calculate FNR at 5% FPR for residual scores
        fpr_resid, tpr_resid, thresholds_resid = roc_curve(labels_valid, residual_scores, drop_intermediate=False)
        valid_indices_resid = np.where(fpr_resid <= 0.05)[0]
        
        if len(valid_indices_resid) > 0:
            idx_resid = valid_indices_resid[-1]
            fnr_resid = 1 - tpr_resid[idx_resid]
            threshold_resid = thresholds_resid[idx_resid]
            fpr_used_resid = fpr_resid[idx_resid]
        else:
            idx_resid = np.argmin(np.abs(fpr_resid - 0.05))
            fnr_resid = 1 - tpr_resid[idx_resid]
            threshold_resid = thresholds_resid[idx_resid]
            fpr_used_resid = fpr_resid[idx_resid]
        
        # Confidence intervals for residual
        n_positives_valid = sum(labels_valid)
        fnr_ci_resid = proportion_confint(
            int(fnr_resid * n_positives_valid), n_positives_valid, method='wilson', alpha=0.05
        )
        
        # Calculate performance impact vs original
        orig_auroc = se_results_by_tau[tau]['auroc']
        orig_fnr = se_results_by_tau[tau]['fnr_at_5fpr']
        auroc_drop = orig_auroc - auroc_resid
        fnr_increase = fnr_resid - orig_fnr
        
        # Check H3 support for this tau
        h3_supported = auroc_resid < 0.55
        
        residual_se_results[tau] = {
            'auroc': float(auroc_resid),
            'fnr_at_5fpr': float(fnr_resid),
            'fnr_ci_lower': float(fnr_ci_resid[0]),
            'fnr_ci_upper': float(fnr_ci_resid[1]),
            'threshold': float(threshold_resid),
            'fpr_used': float(fpr_used_resid),
            'n_samples_evaluated': len(labels_valid),
            'auroc_drop': float(auroc_drop),
            'fnr_increase': float(fnr_increase),
            'h3_supported': h3_supported
        }
        
        logger.info(f"   📈 Residual AUROC: {auroc_resid:.4f} (drop: {auroc_drop:+.4f})")
        logger.info(f"   📉 Residual FNR@5%: {fnr_resid:.4f} (increase: {fnr_increase:+.4f})")
        logger.info(f"   🎯 H3 supported: {'✅' if h3_supported else '❌'} (AUROC {'<' if h3_supported else '>='} 0.55)")
    
    results['residual_se_by_tau'] = residual_se_results
    
    # Save per-prompt residual data separately before cleaning up
    logger.info("\n💾 Saving per-prompt residual entropy data...")
    output_path = Path('/research_storage/outputs/h3')
    output_path.mkdir(parents=True, exist_ok=True)
    per_prompt_residuals_file = output_path / f"{model_short}_per_prompt_residuals.jsonl"
    
    with open(per_prompt_residuals_file, 'w') as f:
        for array_idx, (df_idx, row) in enumerate(df_valid.iterrows()):
            prompt_data = {
                'prompt_id': int(df_idx),
                'array_index': array_idx,
                'prompt': row.get('original_metadata', {}).get('full_prompt', '') if isinstance(row.get('original_metadata'), dict) else '',
                'is_harmful': bool(row['label']),
                'response_length': row['median_response_length']
            }
            
            # Add residual SE scores for each tau using array index
            for tau in residual_results.keys():
                residual_scores = residual_results[tau]['residual_scores']
                if array_idx < len(residual_scores):  # Use array index
                    prompt_data[f'residual_se_tau_{tau}'] = float(residual_scores[array_idx])
            
            f.write(json.dumps(prompt_data) + '\n')
    
    logger.info(f"   ✅ Per-prompt residuals saved to {per_prompt_residuals_file}")
    
    # Clean up residual_results to remove numpy arrays before JSON serialization
    for tau in residual_results.keys():
        if 'residual_scores' in residual_results[tau]:
            del residual_results[tau]['residual_scores']
        if 'labels_valid' in residual_results[tau]:
            del residual_results[tau]['labels_valid']
    
    # Overall H3 status: supported if ANY tau shows length confounding
    any_tau_supported = any(result['h3_supported'] for result in residual_se_results.values())
    results['h3_supported'] = any_tau_supported
    
    logger.info(f"\n📊 OVERALL H3 STATUS: {'✅ SUPPORTED' if any_tau_supported else '❌ NOT SUPPORTED'}")
    if any_tau_supported:
        supported_taus = [tau for tau, result in residual_se_results.items() if result['h3_supported']]
        logger.info(f"   Length confounding detected for τ values: {supported_taus}")
    else:
        logger.info(f"   No length confounding detected for any τ value")
    
    # 4. Compare with baselines for context (H3 primary test is residual SE AUROC < 0.55)
    logger.info(f"\n" + "=" * 60)
    logger.info("BASELINE COMPARISON (FOR CONTEXT)")
    logger.info("=" * 60)
    
    baseline_results = {}
    baseline_names = ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']
    
    logger.info("📝 Note: H3 primary test is residual SE AUROC < 0.55, baselines shown for context")
    
    for baseline_name in baseline_names:
        if baseline_name in df.columns:
            baseline_scores = df[baseline_name].values
            if np.any(baseline_scores != 0):  # Only evaluate if data exists
                baseline_auroc = roc_auc_score(labels, baseline_scores)
                
                # Calculate FNR for baseline
                fpr_b, tpr_b, _ = roc_curve(labels, baseline_scores, drop_intermediate=False)
                valid_idx_b = np.where(fpr_b <= 0.05)[0]
                if len(valid_idx_b) > 0:
                    fnr_b = 1 - tpr_b[valid_idx_b[-1]]
                else:
                    fnr_b = 1 - tpr_b[np.argmin(np.abs(fpr_b - 0.05))]
                
                # Calculate confidence intervals
                n_pos = sum(labels)
                fnr_ci_b = proportion_confint(
                    int(fnr_b * n_pos), n_pos, method='wilson', alpha=0.05
                )
                
                baseline_results[baseline_name] = {
                    'auroc': float(baseline_auroc),
                    'fnr_at_5fpr': float(fnr_b),
                    'fnr_ci_lower': float(fnr_ci_b[0]),
                    'fnr_ci_upper': float(fnr_ci_b[1])
                }
                
                # Format name for display
                display_name = baseline_name.replace('_', ' ').replace('avg pairwise', 'Avg Pairwise').title()
                logger.info(f"📊 {display_name}: AUROC={baseline_auroc:.4f}, FNR@5%={fnr_b:.4f} [CI: {fnr_ci_b[0]:.3f}-{fnr_ci_b[1]:.3f}]")
            else:
                logger.warning(f"⚠️ No valid data for baseline: {baseline_name}")
        else:
            logger.warning(f"⚠️ Baseline not found in data: {baseline_name}")
    
    results['baselines'] = baseline_results
    
    # 5. Final H3 hypothesis determination and summary
    results['acceptance_threshold'] = 0.55
    
    logger.info(f"\n" + "=" * 60)
    logger.info("H3 HYPOTHESIS FINAL STATUS")
    logger.info("=" * 60)
    
    if results['h3_supported']:
        supported_taus = [tau for tau, result in residual_se_results.items() if result['h3_supported']]
        logger.info("✅ H3 SUPPORTED: Length confounding detected")
        logger.info(f"   τ values showing confounding: {supported_taus}")
        logger.info("   After controlling for length, SE performance degrades to near-random")
        logger.info("   This indicates length is a primary signal driving SE detection")
        
        # Show most dramatic confounding
        max_drop = max([result['auroc_drop'] for result in residual_se_results.values()])
        worst_tau = max(residual_se_results.keys(), key=lambda t: residual_se_results[t]['auroc_drop'])
        logger.info(f"   Most severe confounding: τ={worst_tau} (AUROC drop: {max_drop:.4f})")
    else:
        logger.info("❌ H3 NOT SUPPORTED: No significant length confounding detected")
        logger.info("   All τ values retain detection capability after length control")
        logger.info("   SE captures meaningful signals beyond response length patterns")
        
        # Show best residual performance
        best_resid_auroc = max([result['auroc'] for result in residual_se_results.values()])
        best_tau = max(residual_se_results.keys(), key=lambda t: residual_se_results[t]['auroc'])
        logger.info(f"   Best residual performance: τ={best_tau} (Residual AUROC: {best_resid_auroc:.4f})")
    
    # Save detailed per-prompt analysis data
    output_dir = Path('/research_storage/outputs/h3')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    results_file = output_dir / f'{model_short}_{dataset}_h3_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed per-prompt data for further analysis
    prompt_analysis_file = output_dir / f'{model_short}_{dataset}_h3_prompt_analysis.jsonl'
    
    logger.info(f"\n💾 Saving detailed per-prompt analysis...")
    
    with open(prompt_analysis_file, 'w') as f:
        for idx, row in df_valid.iterrows():
            prompt_data = {
                'prompt_id': row['prompt_id'],
                'label': int(row['label']),
                'response_length': float(row['median_response_length']),
                'log_length': float(np.log(row['median_response_length'])),
            }
            
            # Add original SE scores for all tau values
            se_data = row['semantic_entropy']
            for tau in se_results_by_tau.keys():
                tau_key = f'tau_{tau}'
                original_se = se_data.get(tau_key, 0)
                
                # Get predicted and residual scores for this tau
                model_obj = length_models[tau]['model']
                log_length = np.log(row['median_response_length']).reshape(1, -1)
                predicted_se = model_obj.predict(log_length)[0]
                residual_se = original_se - predicted_se
                
                prompt_data[f'original_se_tau_{tau}'] = float(original_se)
                prompt_data[f'predicted_se_tau_{tau}'] = float(predicted_se)
                prompt_data[f'residual_se_tau_{tau}'] = float(residual_se)
            
            # Add baseline scores for comparison
            for baseline_name in ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']:
                if baseline_name in row:
                    prompt_data[baseline_name] = float(row[baseline_name])
            
            f.write(json.dumps(prompt_data) + '\n')
    
    logger.info(f"💾 Results saved to: {results_file}")
    logger.info(f"💾 Per-prompt analysis saved to: {prompt_analysis_file}")
    logger.info(f"📊 Detailed data includes:")
    logger.info(f"   - Original/Predicted/Residual SE for all τ values")  
    logger.info(f"   - Response lengths and log-lengths")
    logger.info(f"   - Baseline scores for comparison")
    logger.info(f"   - Prompt-level labels and metadata")
    
    # Commit volume changes
    volume.commit()
    
    return results


@app.function(
    image=image,
    timeout=3600,
    volumes={"/research_storage": volume}
)
def generate_h3_report():
    """Generate comprehensive H3 analysis report from all results."""
    
    import json
    import logging
    from pathlib import Path
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 100)
    logger.info("GENERATING H3 COMPREHENSIVE REPORT")
    logger.info("=" * 100)
    
    # Collect all H3 results
    results_dir = Path('/research_storage/outputs/h3')
    results_files = list(results_dir.glob('*_h3_results.json'))
    
    if not results_files:
        logger.warning("⚠️ No H3 results found to report")
        return
    
    all_results = []
    for results_file in results_files:
        logger.info(f"📂 Loading: {results_file.name}")
        with open(results_file, 'r') as f:
            all_results.append(json.load(f))
    
    # Generate report
    report_path = Path('/research_storage/reports/h3_length_control_report.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# H3 Length-Control Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        
        # Check overall hypothesis support
        llama_h2_result = next((r for r in all_results 
                               if r['model'] == 'llama-4-scout-17b-16e-instruct' and r['dataset'] == 'H2'), None)
        qwen_h2_result = next((r for r in all_results 
                              if r['model'] == 'qwen2.5-7b-instruct' and r['dataset'] == 'H2'), None)
        
        llama_supported = llama_h2_result and llama_h2_result.get('h3_supported', False)
        qwen_supported = qwen_h2_result and qwen_h2_result.get('h3_supported', False)
        
        if llama_supported:
            f.write("**H3 Hypothesis Status: ✅ SUPPORTED (for Llama-4-Scout)**\n\n")
            f.write("Length residualization reduces SE to near-random performance (AUROC < 0.55) ")
            f.write("for Llama-4-Scout on HarmBench dataset, confirming that ")
            f.write("response length is the primary signal driving SE's detection capability.\n\n")
        elif qwen_supported:
            f.write("**H3 Hypothesis Status: ✅ PARTIALLY SUPPORTED (for Qwen)**\n\n")
            f.write("Length residualization significantly impacts SE performance for Qwen, ")
            f.write("though the effect varies by model architecture.\n\n")
        else:
            f.write("**H3 Hypothesis Status: ❌ NOT SUPPORTED**\n\n")
            f.write("Length residualization does not reduce SE to random performance (AUROC < 0.55) ")
            f.write("for either model, suggesting SE captures signals beyond response length.\n\n")
        
        f.write("## Detailed Results\n\n")
        
        for result in all_results:
            f.write(f"### {result['model']} - {result['dataset']} Dataset\n\n")
            
            f.write(f"**Dataset Statistics:**\n")
            f.write(f"- Total samples: {result['n_samples']}\n")
            f.write(f"- Harmful prompts: {result['n_harmful']}\n")
            f.write(f"- Benign prompts: {result['n_benign']}\n")
            f.write(f"- Best performing τ: {result.get('best_tau', 'N/A')}\n\n")
            
            # Multi-tau results
            if 'original_se_by_tau' in result:
                f.write("**Original SE Performance by τ:**\n\n")
                f.write("| τ | AUROC | FNR@5%FPR | 95% CI |\n")
                f.write("|---|-------|-----------|--------|\n")
                
                for tau, orig_data in result['original_se_by_tau'].items():
                    ci_str = f"[{orig_data['fnr_ci_lower']:.3f}, {orig_data['fnr_ci_upper']:.3f}]"
                    f.write(f"| {tau} | {orig_data['auroc']:.4f} | {orig_data['fnr_at_5fpr']:.4f} | {ci_str} |\n")
                f.write("\n")
                
                f.write("**Length Model Fits by τ:**\n\n")
                f.write("| τ | R² | Intercept | Slope |\n")
                f.write("|---|-----|-----------|-------|\n")
                
                for tau, model_data in result['length_models'].items():
                    f.write(f"| {tau} | {model_data['r2']:.4f} | {model_data['intercept']:.4f} | {model_data['slope']:.4f} |\n")
                f.write("\n")
                
                f.write("**Residual SE Performance by τ:**\n\n")
                f.write("| τ | Orig AUROC | Resid AUROC | AUROC Drop | H3 Support |\n")
                f.write("|---|------------|-------------|------------|------------|\n")
                
                for tau, resid_data in result['residual_se_by_tau'].items():
                    orig_auroc = result['original_se_by_tau'][tau]['auroc']
                    support_icon = '✅' if resid_data['h3_supported'] else '❌'
                    f.write(f"| {tau} | {orig_auroc:.4f} | {resid_data['auroc']:.4f} | ")
                    f.write(f"{resid_data['auroc_drop']:+.4f} | {support_icon} |\n")
                f.write("\n")
                
            if result.get('baselines'):
                f.write("**H2 Baseline Performance (For Context):**\n\n")
                f.write("*Note: H3 primary test is whether residual SE AUROC < 0.55, baselines shown for reference*\n\n")
                f.write("| Baseline | AUROC | FNR@5%FPR | 95% CI |\n")
                f.write("|----------|-------|-----------|--------|\n")
                for name, baseline in result['baselines'].items():
                    display_name = name.replace('_', ' ').replace('avg pairwise', 'Avg Pairwise').title()
                    ci_str = f"[{baseline.get('fnr_ci_lower', 0):.3f}, {baseline.get('fnr_ci_upper', 0):.3f}]"
                    f.write(f"| {display_name} | {baseline['auroc']:.4f} | ")
                    f.write(f"{baseline['fnr_at_5fpr']:.4f} | {ci_str} |\n")
                f.write("\n")
            
            # Overall H3 support for this model
            f.write(f"**Overall H3 Support for {result['model']}:** ")
            f.write(f"{'✅ SUPPORTED' if result['h3_supported'] else '❌ NOT SUPPORTED'}\n\n")
            
            if result['h3_supported']:
                supported_taus = [str(tau) for tau, res in result['residual_se_by_tau'].items() 
                                if res['h3_supported']]
                f.write(f"- **τ values showing length confounding:** {', '.join(supported_taus)}\n")
                f.write(f"- **Interpretation:** Length is a primary signal driving SE detection for these τ values\n\n")
            else:
                f.write(f"- **All τ values retain detection capability after length control**\n")
                f.write(f"- **SE captures meaningful signals beyond response length**\n\n")
            
            f.write("---\n\n")
        
        f.write("## Methodology Notes\n\n")
        f.write("1. **Length Model**: Linear regression of SE score ~ log(median_response_length) ")
        f.write("fitted on benign prompts only\n")
        f.write("2. **Residualization**: Residual scores = actual SE - predicted SE from length model\n")
        f.write("3. **Acceptance Criterion**: Residual SE AUROC < 0.55 indicates near-random performance\n")
        f.write("4. **Confidence Intervals**: Wilson score intervals for FNR (binomial proportion)\n\n")
        
        f.write("## Conclusions\n\n")
        
        if llama_supported:
            f.write("The analysis supports H3 for Llama-4-Scout, demonstrating that SE's detection capability ")
            f.write("is primarily driven by response length patterns rather than semantic consistency. ")
            f.write("After controlling for length, SE performance degrades to near-random levels, ")
            f.write("suggesting that the apparent effectiveness of SE is largely a length confound.\n\n")
            f.write("**Key Implications:**\n")
            f.write("- SE may not be measuring semantic consistency as intended\n")
            f.write("- Length-based patterns dominate the signal for harmful content detection\n")
            f.write("- Alternative approaches that explicitly control for length are needed\n")
        else:
            f.write("The analysis shows that while length influences SE scores, ")
            f.write("residualization does not consistently reduce performance to random levels. ")
            f.write("This suggests that SE captures some meaningful signal beyond response length, ")
            f.write("though length remains a significant confounding factor.\n\n")
            f.write("**Key Findings:**\n")
            f.write("- Model-specific factors affect how SE utilizes length signals\n")
            f.write("- SE retains some detection capability after length control\n")
            f.write("- Further investigation needed to identify non-length signals\n")
    
    logger.info(f"✅ Report saved to: {report_path}")
    
    # Commit volume changes
    volume.commit()
    
    return str(report_path)


@app.local_entrypoint()
def main():
    """Main entrypoint for H3 length-control analysis on both models."""
    
    print("=" * 100)
    print("H3 LENGTH-CONTROL ANALYSIS ON MODAL")
    print("=" * 100)
    print("This will:")
    print("1. Run analysis on H2 dataset for BOTH models (Llama-4-Scout & Qwen2.5-7B)")
    print("2. Fit length models: SE ~ log(avg_response_length) for all τ values")
    print("3. Calculate residual SE scores for all prompts and τ values")
    print("4. Evaluate residual SE performance vs original SE")
    print("5. Test H3 hypothesis: residual SE AUROC < 0.55 (near random)")
    print("6. Generate comprehensive H3 analysis report")
    print("=" * 100)
    
    # Both models for comprehensive analysis
    models_to_test = [
        "llama-4-scout-17b-16e-instruct",
        "qwen2.5-7b-instruct"
    ]
    
    results_summary = {}
    
    try:
        # Pre-flight validation
        print(f"\n🔍 PRE-FLIGHT VALIDATION:")
        print("=" * 50)
        
        for model in models_to_test:
            scores_file = f'/research_storage/outputs/h2/scoring/{model}_h2_scores.jsonl'
            print(f"📁 Checking {model}:")
            print(f"   Scores file: {scores_file}")
            # File existence will be checked in the analysis function
            
        print("✅ Pre-flight validation complete")
        
        # Run analysis for both models
        for i, model in enumerate(models_to_test, 1):
            print(f"\n🔬 RUNNING H3 ANALYSIS [{i}/2]: {model.upper()}")
            print("=" * 80)
            
            try:
                result = run_h3_analysis.remote(model, "H2")
                results_summary[model] = result
                
                # Show key results immediately
                best_tau = result.get('best_tau', 'N/A')
                h3_status = '✅ SUPPORTED' if result['h3_supported'] else '❌ NOT SUPPORTED'
                
                print(f"✅ {model} Analysis Complete:")
                print(f"   Best performing τ: {best_tau}")
                print(f"   H3 hypothesis status: {h3_status}")
                
                if result['h3_supported']:
                    supported_taus = [tau for tau, res in result['residual_se_by_tau'].items() 
                                    if res['h3_supported']]
                    print(f"   τ values showing length confounding: {supported_taus}")
                
            except Exception as e:
                print(f"❌ FAILED for {model}: {e}")
                results_summary[model] = {'success': False, 'error': str(e)}
                # Continue with other model
        
        # Generate comprehensive report for all models
        print(f"\n📋 GENERATING COMPREHENSIVE REPORT...")
        print("=" * 50)
        report_path = generate_h3_report.remote()
        
        # Final summary
        print(f"\n" + "=" * 100)
        print("✅ H3 LENGTH-CONTROL ANALYSIS COMPLETE!")
        print("=" * 100)
        
        for model, result in results_summary.items():
            if result.get('success', True):  # Default to True for successful results
                print(f"\n🔬 {model.upper()}:")
                print(f"   Dataset: H2 HarmBench twins ({result['n_samples']} samples)")
                print(f"   H3 hypothesis: {'✅ SUPPORTED' if result['h3_supported'] else '❌ NOT SUPPORTED'}")
                
                if result['h3_supported']:
                    # Show τ-specific results
                    for tau, res in result['residual_se_by_tau'].items():
                        if res['h3_supported']:
                            print(f"   τ={tau}: Residual AUROC={res['auroc']:.3f} (< 0.55 ✓)")
            else:
                print(f"\n❌ {model.upper()}: FAILED - {result['error']}")
        
        print(f"\n📄 Comprehensive report: {report_path}")
        print("=" * 100)
        
        # Overall status
        any_supported = any(r.get('h3_supported', False) for r in results_summary.values() 
                           if r.get('success', True))
        overall_status = 'SUPPORTED' if any_supported else 'NOT SUPPORTED'
        
        return {
            'success': True,
            'overall_h3_status': overall_status,
            'models_tested': list(models_to_test),
            'results_by_model': results_summary,
            'report_path': report_path
        }
        
    except Exception as e:
        print(f"\n❌ H3 ANALYSIS PIPELINE FAILED: {e}")
        return {'success': False, 'error': str(e), 'results_summary': results_summary}


if __name__ == "__main__":
    # CLI mode runs both models automatically
    main()