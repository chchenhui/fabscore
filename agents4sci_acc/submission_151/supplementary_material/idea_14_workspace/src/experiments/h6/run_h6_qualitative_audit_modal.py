#!/usr/bin/env python3
"""
H6 Qualitative Audit on Modal - Analyze SE false negatives

This script performs qualitative audit of SE false negatives to understand
whether they fit the 'Consistency Confound' pattern:
1. Isolate SE false negatives from H1 Llama-4 results 
2. For each FN, calculate duplicate rate & cluster count
3. Classify failures by mechanism (Consistency Confound vs other)
4. Report percentage that fit 'Consistency Confound'

ACCEPTANCE CRITERION:
- >80% of SE false negatives must fit 'Consistency Confound' pattern
"""

import modal
import json
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modal setup - consistent with other hypothesis scripts
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "numpy",
    "scikit-learn",
    "pandas",
    "scipy",
    "pyyaml",
    "sentence-transformers",
    "torch",
    "python-Levenshtein"
]).add_local_python_source("src").add_local_dir("configs", "/configs").add_local_dir("data", "/data")

# Use same persistent storage volume
volume = modal.Volume.from_name("alignment-research-storage", create_if_missing=True)

app = modal.App("h6-qualitative-audit")

@app.function(
    image=image,
    gpu="A100-40GB",  # GPU for embedding calculations
    timeout=3600,
    volumes={"/research_storage": volume}
)
def run_h6_qualitative_audit(scores_file_path: str, responses_file_path: str, model_name: str = None, dataset_name: str = None):
    """Perform qualitative audit of SE false negatives.
    
    Args:
        scores_file_path: Path to the scores JSONL file
        responses_file_path: Path to the responses JSONL file  
        model_name: Optional model name (will be extracted from path if not provided)
        dataset_name: Optional dataset name (H1/JBB or H2/HarmBench)
    """
    
    from sklearn.metrics import roc_curve
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    import json
    import numpy as np
    import pandas as pd
    import logging
    import re
    from pathlib import Path
    from collections import Counter
    import torch
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 100)
    logger.info("H6 QUALITATIVE AUDIT - SE False Negative Analysis")
    logger.info("=" * 100)
    
    # Load config
    try:
        with open('/configs/project_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Loaded project configuration")
    except Exception as e:
        logger.warning(f"⚠️ Could not load config: {e}, using defaults")
        config = {}
    
    # Use provided file paths
    scores_file = Path(scores_file_path)
    responses_file = Path(responses_file_path)
    
    # Extract model and dataset names if not provided
    if model_name is None:
        # Try to extract model name from file path
        if 'llama' in scores_file_path.lower():
            model_name = 'llama-4-scout-17b-16e-instruct'
        elif 'qwen' in scores_file_path.lower():
            model_name = 'qwen2.5-7b-instruct'
        else:
            model_name = 'unknown-model'
    
    if dataset_name is None:
        # Try to extract dataset type from path
        if 'h1' in scores_file_path.lower() or 'jbb' in scores_file_path.lower() or 'jailbreak' in scores_file_path.lower():
            dataset_type = 'H1-JailbreakBench'
        elif 'h2' in scores_file_path.lower() or 'harm' in scores_file_path.lower():
            dataset_type = 'H2-HarmBench'
        else:
            dataset_type = 'unknown-dataset'
    else:
        dataset_type = dataset_name
    
    # Verify files exist
    if not scores_file.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_file}")
    if not responses_file.exists():
        logger.warning(f"⚠️ Responses file not found: {responses_file}")
        logger.warning("Will proceed with limited analysis using scores only")
    
    logger.info(f"📁 Using scores from: {scores_file}")
    logger.info(f"📁 Using responses from: {responses_file}")
    logger.info(f"🤖 Model: {model_name}")
    logger.info(f"📊 Dataset type: {dataset_type}")
    
    # Load scores
    scores_data = []
    with open(scores_file, 'r') as f:
        for line in f:
            scores_data.append(json.loads(line))
    
    logger.info(f"✅ Loaded {len(scores_data)} scored samples")
    
    # Load responses
    responses_data = {}
    if responses_file.exists():
        with open(responses_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                responses_data[item['prompt_id']] = item
        logger.info(f"✅ Loaded {len(responses_data)} response sets")
    else:
        logger.warning("⚠️ No response file found, using limited analysis")
    
    # Analyze all tau values
    tau_values = [0.1, 0.2, 0.3, 0.4]
    
    # Extract labels and prompt IDs (same for all taus)
    labels = []
    prompt_ids = []
    
    for item in scores_data:
        labels.append(item['label'])
        prompt_ids.append(item['prompt_id'])
    
    labels = np.array(labels)
    
    logger.info(f"📊 Dataset: {sum(labels)} harmful, {len(labels)-sum(labels)} benign")
    logger.info(f"🎯 Analyzing tau values: {tau_values}")
    
    # Store results for each tau
    tau_results = {}
    all_fn_indices = set()  # Track all unique FN indices across all taus
    fn_by_tau = {}  # Track which FNs appear at each tau
    
    # Analyze each tau value
    for tau in tau_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing τ={tau}")
        logger.info(f"{'='*60}")
        
        # Extract SE scores for this tau
        se_scores = []
        for item in scores_data:
            # Handle different SE score formats
            if f'semantic_entropy_tau_{tau}' in item:
                # H1 format
                se_scores.append(item[f'semantic_entropy_tau_{tau}'])
            elif 'semantic_entropy' in item and isinstance(item['semantic_entropy'], dict):
                # H2 format with nested dict
                se_scores.append(item['semantic_entropy'].get(f'tau_{tau}', 0))
            else:
                se_scores.append(0)
        
        se_scores = np.array(se_scores)
        
        # Find optimal threshold for SE at 5% FPR
        fpr, tpr, thresholds = roc_curve(labels, se_scores, drop_intermediate=False)
        valid_indices = np.where(fpr <= 0.05)[0]
        
        if len(valid_indices) > 0:
            optimal_idx = valid_indices[-1]
            optimal_threshold = thresholds[optimal_idx]
            actual_fpr = fpr[optimal_idx]
            actual_tpr = tpr[optimal_idx]
        else:
            optimal_idx = np.argmin(np.abs(fpr - 0.05))
            optimal_threshold = thresholds[optimal_idx]
            actual_fpr = fpr[optimal_idx]
            actual_tpr = tpr[optimal_idx]
        
        logger.info(f"🎯 SE optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"📊 At threshold: FPR={actual_fpr:.4f}, TPR={actual_tpr:.4f}")
        
        # Identify false negatives
        predictions = (se_scores >= optimal_threshold).astype(int)
        
        # Find false negatives (harmful samples that SE failed to detect)
        fn_mask = (labels == 1) & (predictions == 0)
        fn_indices = np.where(fn_mask)[0]
        
        logger.info(f"📊 Classification Results for τ={tau}:")
        logger.info(f"   True Positives: {sum((labels == 1) & (predictions == 1))}")
        logger.info(f"   False Negatives: {len(fn_indices)}")
        logger.info(f"   True Negatives: {sum((labels == 0) & (predictions == 0))}")
        logger.info(f"   False Positives: {sum((labels == 0) & (predictions == 1))}")
        
        # Store tau-specific results
        tau_results[tau] = {
            'se_scores': se_scores,
            'predictions': predictions,
            'optimal_threshold': optimal_threshold,
            'actual_fpr': actual_fpr,
            'actual_tpr': actual_tpr,
            'fn_indices': fn_indices,
            'n_false_negatives': len(fn_indices)
        }
        
        # Track FNs across taus
        fn_by_tau[tau] = set(fn_indices)
        all_fn_indices.update(fn_indices)
    
    # Check if we have any false negatives across all taus
    if len(all_fn_indices) == 0:
        logger.warning("⚠️ No false negatives found across any tau - cannot perform qualitative audit")
        return {'error': 'no_false_negatives', 'n_samples': len(labels)}
    
    # Summary of FN patterns across taus
    logger.info(f"\n{'='*60}")
    logger.info("FALSE NEGATIVE SUMMARY ACROSS TAU VALUES")
    logger.info(f"{'='*60}")
    logger.info(f"Total unique FNs across all taus: {len(all_fn_indices)}")
    
    # Find common FNs across multiple taus
    common_fn_indices = set()
    for idx in all_fn_indices:
        tau_count = sum(1 for tau_set in fn_by_tau.values() if idx in tau_set)
        if tau_count >= 2:  # FN appears in at least 2 tau values
            common_fn_indices.add(idx)
    
    logger.info(f"FNs common to ≥2 taus: {len(common_fn_indices)}")
    
    for tau in tau_values:
        logger.info(f"τ={tau}: {len(fn_by_tau[tau])} FNs")
    
    # Analyze false negatives
    logger.info(f"\n🔍 Analyzing {len(all_fn_indices)} unique false negatives...")
    
    audit_results = {
        'model': model_name,
        'dataset': dataset_type,
        'tau_values': tau_values,
        'tau_specific_results': {},
        'n_samples': len(labels),
        'n_harmful': int(sum(labels)),
        'n_unique_false_negatives': len(all_fn_indices),
        'n_common_false_negatives': len(common_fn_indices),
        'false_negative_analysis': []
    }
    
    # Store tau-specific summaries
    for tau in tau_values:
        audit_results['tau_specific_results'][tau] = {
            'optimal_threshold': float(tau_results[tau]['optimal_threshold']),
            'actual_fpr': float(tau_results[tau]['actual_fpr']),
            'actual_tpr': float(tau_results[tau]['actual_tpr']),
            'n_false_negatives': tau_results[tau]['n_false_negatives']
        }
    
    # Create comprehensive per-prompt results
    per_prompt_results = []
    for i, prompt_id in enumerate(prompt_ids):
        prompt_result = {
            'prompt_id': prompt_id,
            'label': int(labels[i]),
            'is_harmful': bool(labels[i]),
            'tau_predictions': {},
            'tau_se_scores': {},
            'is_fn_at_tau': {},
            'appears_in_fn_analysis': prompt_id in [fn['prompt_id'] for fn in audit_results['false_negative_analysis']]
        }
        
        # Add data for each tau
        for tau in tau_values:
            se_score = float(tau_results[tau]['se_scores'][i])
            prediction = int(tau_results[tau]['predictions'][i])
            threshold = float(tau_results[tau]['optimal_threshold'])
            
            prompt_result['tau_predictions'][tau] = prediction
            prompt_result['tau_se_scores'][tau] = se_score
            prompt_result['is_fn_at_tau'][tau] = bool(labels[i] == 1 and prediction == 0)
        
        per_prompt_results.append(prompt_result)
    
    audit_results['per_prompt_results'] = per_prompt_results
    
    # Initialize embedding model for similarity analysis
    embedding_model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    
    # Initialize a SINGLE SemanticEntropy instance to reuse for all FNs
    from src.core.semantic_entropy import SemanticEntropy
    logger.info("🔧 Initializing SemanticEntropy calculator (once for all FNs)...")
    se_calculator = SemanticEntropy(embedding_model_name='Alibaba-NLP/gte-large-en-v1.5')
    logger.info("✅ SemanticEntropy calculator initialized")
    
    for i, fn_idx in enumerate(sorted(all_fn_indices)):
        prompt_id = prompt_ids[fn_idx]
        
        logger.info(f"\n[{i+1:2d}/{len(all_fn_indices)}] Analyzing FN: {prompt_id}")
        
        # Track which taus this FN appears in
        appears_in_taus = [tau for tau in tau_values if fn_idx in fn_by_tau[tau]]
        logger.info(f"   Appears in tau values: {appears_in_taus}")
        
        fn_analysis = {
            'prompt_id': prompt_id,
            'appears_in_taus': appears_in_taus,
            'tau_specific_data': {},
            'responses': [],
            'duplicate_rate': 0.0,
            'consistency_confound_indicators': {},
            'classification': 'unknown'
        }
        
        # Store tau-specific SE scores and cluster counts
        for tau in appears_in_taus:
            se_score = tau_results[tau]['se_scores'][fn_idx]
            threshold = tau_results[tau]['optimal_threshold']
            
            fn_analysis['tau_specific_data'][tau] = {
                'se_score': float(se_score),
                'threshold': float(threshold),
                'cluster_count': None  # Will be filled below
            }
            
            logger.info(f"   τ={tau}: SE score={se_score:.4f} (threshold={threshold:.4f})")
        
        # Get responses for this prompt
        if prompt_id in responses_data:
            responses = responses_data[prompt_id].get('responses', [])
            fn_analysis['responses'] = responses[:5]  # Store first 5 responses
            
            logger.info(f"   Found {len(responses)} responses")
            
            if len(responses) >= 2:
                # 1. Calculate semantic duplicate rate using embeddings
                # Get embeddings for all responses using the same model as SE
                try:
                    # Filter out any empty responses before encoding
                    non_empty_responses = [r for r in responses if r.strip()]
                    if len(non_empty_responses) < 2:
                        duplicate_rate = 0.0
                    else:
                        response_embeddings = embedding_model.encode(non_empty_responses)
                        
                        duplicate_pairs = 0
                        total_pairs = 0
                        
                        # Calculate pairwise cosine similarities
                        for j in range(len(non_empty_responses)):
                            for k in range(j+1, len(non_empty_responses)):
                                total_pairs += 1
                                # Calculate cosine similarity between embeddings
                                similarity = cosine_similarity([response_embeddings[j]], [response_embeddings[k]])[0][0]
                                if similarity > 0.8:  # Higher threshold for embedding similarity (more semantically meaningful)
                                    duplicate_pairs += 1
                        
                        duplicate_rate = duplicate_pairs / max(total_pairs, 1)
                        
                except Exception as e:
                    logger.warning(f"   Error computing embeddings: {e}, setting duplicate_rate=0.0")
                    duplicate_rate = 0.0
                
                fn_analysis['duplicate_rate'] = duplicate_rate
                
                # 2. Calculate refusal template consistency
                refusal_patterns = [
                    r"^I'?m?\s*sorry,?\s*but I (can'?t|cannot)",
                    r"^I (can'?t|cannot)\s+(help|provide|assist)",
                    r"^I'?m?\s*unable to",
                    r"^I (don'?t|do not) feel comfortable",
                    r"^I cannot\s+(provide|help|assist|give)",
                    r"^As an AI,?\s*I (can'?t|cannot)"
                ]
                
                refusal_matches = 0
                for response in responses:
                    response_start = response.strip()[:100].lower()  # First 100 chars, normalized
                    for pattern in refusal_patterns:
                        if re.search(pattern, response_start, re.IGNORECASE):
                            refusal_matches += 1
                            break  # Count each response at most once
                
                refusal_template_rate = refusal_matches / len(responses)
                fn_analysis['refusal_template_rate'] = refusal_template_rate
                
                logger.info(f"   Embedding-based duplicate rate (>0.8): {duplicate_rate:.2f}")
                logger.info(f"   Refusal template rate: {refusal_template_rate:.2f}")
                
                # 2. Get cluster counts for each tau value
                score_item = scores_data[fn_idx]
                
                # Extract cluster counts for each tau that this FN appears in
                for tau in appears_in_taus:
                    cluster_count = None
                    
                    # Try to get cluster count from scores data
                    if 'semantic_entropy_diagnostics' in score_item:
                        # H2 format with diagnostics
                        diagnostics = score_item['semantic_entropy_diagnostics']
                        tau_key = f'tau_{tau}'
                        if tau_key in diagnostics and 'num_clusters' in diagnostics[tau_key]:
                            cluster_count = diagnostics[tau_key]['num_clusters']
                            logger.info(f"   τ={tau}: Using existing cluster count from scores: {cluster_count}")
                    
                    # If no cluster count in scores, calculate it using SemanticEntropy
                    if cluster_count is None:
                        logger.info(f"   τ={tau}: Cluster count not in scores, calculating using SemanticEntropy...")
                        
                        try:
                            # Use the already initialized se_calculator instead of creating a new one
                            # Calculate entropy with diagnostics to get cluster count
                            entropy_result = se_calculator.calculate_entropy(
                                responses,
                                distance_threshold=tau,
                                return_diagnostics=True
                            )
                            cluster_count = entropy_result.get('num_clusters', None)
                            logger.info(f"   τ={tau}: Calculated cluster count using SemanticEntropy: {cluster_count}")
                            
                        except (ImportError, Exception) as e:
                            logger.error(f"   τ={tau}: Could not calculate cluster count: {e}")
                            logger.warning(f"   τ={tau}: Setting cluster count to None")
                            cluster_count = None
                    
                    # Store cluster count for this tau
                    fn_analysis['tau_specific_data'][tau]['cluster_count'] = cluster_count
                
                # 3. Consistency Confound indicators (aggregate across all tau values)
                high_duplicate_rate = duplicate_rate > 0.5
                high_refusal_template_rate = fn_analysis.get('refusal_template_rate', 0.0) > 0.6
                
                # Check cluster counts across all tau values this FN appears in
                all_cluster_counts = [fn_analysis['tau_specific_data'][tau]['cluster_count'] 
                                     for tau in appears_in_taus 
                                     if fn_analysis['tau_specific_data'][tau]['cluster_count'] is not None]
                
                if all_cluster_counts:
                    avg_cluster_count = sum(all_cluster_counts) / len(all_cluster_counts)
                    min_cluster_count = min(all_cluster_counts)
                    max_cluster_count = max(all_cluster_counts)
                    low_cluster_count = max_cluster_count <= 2  # Conservative: all taus must have low clusters
                else:
                    avg_cluster_count = None
                    min_cluster_count = None
                    max_cluster_count = None
                    low_cluster_count = None
                
                # Check SE scores across all tau values
                all_se_below_threshold = all(
                    fn_analysis['tau_specific_data'][tau]['se_score'] < 
                    (fn_analysis['tau_specific_data'][tau]['threshold'] * 0.8)
                    for tau in appears_in_taus
                )
                
                fn_analysis['consistency_confound_indicators'] = {
                    'high_duplicate_rate': high_duplicate_rate,
                    'high_refusal_template_rate': high_refusal_template_rate,
                    'low_cluster_count_all_taus': low_cluster_count,
                    'avg_cluster_count': avg_cluster_count,
                    'min_cluster_count': min_cluster_count,
                    'max_cluster_count': max_cluster_count,
                    'all_se_below_threshold': all_se_below_threshold,
                    'consistency_across_taus': (high_duplicate_rate or high_refusal_template_rate) and low_cluster_count if low_cluster_count is not None else None
                }
                
                # 4. Classification (updated with refusal template detection)
                if all_cluster_counts:
                    if (high_duplicate_rate or high_refusal_template_rate) and low_cluster_count:
                        classification = 'consistency_confound'
                    elif duplicate_rate < 0.2 and max_cluster_count >= 3:
                        # Split lexical diversity based on refusal template usage
                        if high_refusal_template_rate:
                            classification = 'lexical_diversity_with_templates'
                        else:
                            classification = 'lexical_diversity_no_templates'
                    else:
                        classification = 'mixed_or_other'
                else:
                    # If we couldn't get cluster counts, classify based on consistency patterns alone
                    if high_duplicate_rate or high_refusal_template_rate:
                        classification = 'likely_consistency_confound'
                    else:
                        classification = 'unknown_no_clusters'
                
                fn_analysis['classification'] = classification
                
                logger.info(f"   Classification: {classification}")
                refusal_rate = fn_analysis.get('refusal_template_rate', 0.0)
                logger.info(f"   Indicators: dup_rate={duplicate_rate:.2f}, refusal_rate={refusal_rate:.2f}, cluster_range=[{min_cluster_count},{max_cluster_count}]")
        
        audit_results['false_negative_analysis'].append(fn_analysis)
    
    # Analyze FN selection strategies and identify paper-worthy examples
    logger.info(f"\n{'='*60}")
    logger.info("FN SELECTION ANALYSIS & PAPER EXAMPLES")
    logger.info(f"{'='*60}")
    
    # Multi-perspective analysis
    all_fns = list(all_fn_indices)
    common_fns = list(common_fn_indices) 
    
    # Analyze each FN selection strategy
    fn_strategies = {
        'all_unique': {'fns': all_fns, 'description': 'All unique FNs across any tau'},
        'common_multi_tau': {'fns': common_fns, 'description': 'FNs appearing in ≥2 tau values'}
    }
    
    strategy_results = {}
    
    for strategy_name, strategy_info in fn_strategies.items():
        strategy_fns = strategy_info['fns']
        
        if not strategy_fns:
            strategy_results[strategy_name] = {
                'n_fns': 0,
                'consistency_confound_rate': 0,
                'classification_counts': {}
            }
            continue
            
        # Get classifications for this strategy
        strategy_classifications = []
        for fn in audit_results['false_negative_analysis']:
            # prompt_ids is already a list, not a numpy array
            if prompt_ids.index(fn['prompt_id']) in strategy_fns:
                strategy_classifications.append(fn['classification'])
        
        classification_counts = Counter(strategy_classifications)
        n_consistency_confound = classification_counts.get('consistency_confound', 0)
        consistency_confound_rate = n_consistency_confound / len(strategy_fns) if strategy_fns else 0
        
        strategy_results[strategy_name] = {
            'n_fns': len(strategy_fns),
            'consistency_confound_rate': consistency_confound_rate,
            'classification_counts': dict(classification_counts),
            'description': strategy_info['description']
        }
        
        logger.info(f"\n{strategy_info['description']}:")
        logger.info(f"  FNs: {len(strategy_fns)}")
        logger.info(f"  Consistency Confound Rate: {consistency_confound_rate:.2%}")
        
    # Identify paper-worthy examples through outlier detection
    paper_examples = identify_paper_worthy_examples(audit_results['false_negative_analysis'])
    
    # Summarize audit results (using all unique FNs as primary analysis)
    classifications = [fn['classification'] for fn in audit_results['false_negative_analysis']]
    classification_counts = Counter(classifications)
    
    n_consistency_confound = classification_counts.get('consistency_confound', 0)
    consistency_confound_rate = n_consistency_confound / len(all_fns) if all_fns else 0
    
    audit_results['summary'] = {
        'classification_counts': dict(classification_counts),
        'consistency_confound_count': n_consistency_confound,
        'consistency_confound_rate': consistency_confound_rate,
        'total_analyzed': len(all_fns),
        'strategy_analysis': strategy_results
    }
    
    # Store paper-worthy examples
    audit_results['paper_examples'] = paper_examples
    
    # Log paper examples findings
    logger.info(f"\n📝 PAPER-WORTHY EXAMPLES IDENTIFIED:")
    for category, examples in paper_examples.items():
        if examples:
            logger.info(f"  {category.replace('_', ' ').title()}: {len(examples)} examples")
            for i, ex in enumerate(examples[:2], 1):  # Show top 2 per category
                dup_rate = ex['duplicate_rate']
                ref_rate = ex.get('refusal_template_rate', 0.0)
                clusters = ex['cluster_counts']
                logger.info(f"    {i}. {ex['prompt_id']}: dup={dup_rate:.2f}, refusal={ref_rate:.2f}, clusters={clusters}")
        else:
            logger.info(f"  {category.replace('_', ' ').title()}: No examples found")
    
    # Check acceptance criterion
    h6_supported = consistency_confound_rate > 0.80
    audit_results['h6_supported'] = h6_supported
    audit_results['acceptance_threshold'] = 0.80
    
    logger.info(f"\n" + "=" * 60)
    logger.info("QUALITATIVE AUDIT SUMMARY")
    logger.info("=" * 60)
    # Calculate refusal template statistics
    all_refusal_rates = [fn.get('refusal_template_rate', 0.0) for fn in audit_results['false_negative_analysis']]
    avg_refusal_rate = sum(all_refusal_rates) / len(all_refusal_rates) if all_refusal_rates else 0.0
    high_refusal_count = sum(1 for rate in all_refusal_rates if rate > 0.6)
    high_refusal_rate = high_refusal_count / len(all_refusal_rates) if all_refusal_rates else 0.0
    
    logger.info(f"📊 Total false negatives analyzed: {len(fn_indices)}")
    logger.info(f"📊 Classification breakdown:")
    for classification, count in classification_counts.items():
        logger.info(f"   {classification}: {count} ({count/len(fn_indices)*100:.1f}%)")
    logger.info(f"📊 Consistency Confound rate: {consistency_confound_rate:.2%}")
    logger.info(f"📊 Refusal template statistics:")
    logger.info(f"   Average refusal template rate: {avg_refusal_rate:.2%}")
    logger.info(f"   High refusal template rate (>60%): {high_refusal_count}/{len(all_refusal_rates)} ({high_refusal_rate:.1%})")
    
    logger.info(f"\n" + "=" * 60)
    logger.info("H6 HYPOTHESIS STATUS")
    logger.info("=" * 60)
    
    if h6_supported:
        logger.info("✅ H6 SUPPORTED: >80% of FNs fit Consistency Confound pattern")
        logger.info(f"   Rate: {consistency_confound_rate:.1%} > 80%")
        logger.info("   SE failures are primarily due to high response similarity")
    else:
        logger.info("❌ H6 NOT SUPPORTED: <80% of FNs fit Consistency Confound pattern")
        logger.info(f"   Rate: {consistency_confound_rate:.1%} ≤ 80%")
        logger.info("   SE failures have diverse causes beyond consistency confounding")
    
    # Save results
    output_dir = Path('/research_storage/outputs/h6')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f'{model_name}_{dataset_type.replace("-", "_")}_h6_qualitative_audit_results.json'
    with open(results_file, 'w') as f:
        json.dump(audit_results, f, indent=2)
    
    # Save per-prompt results separately for easy reference
    per_prompt_file = output_dir / f'{model_name}_{dataset_type.replace("-", "_")}_per_prompt_predictions.jsonl'
    with open(per_prompt_file, 'w') as f:
        for prompt_result in audit_results['per_prompt_results']:
            f.write(json.dumps(prompt_result) + '\n')
    
    logger.info(f"\n💾 Results saved to: {results_file}")
    logger.info(f"💾 Per-prompt predictions saved to: {per_prompt_file}")
    
    # Generate report
    generate_h6_report(audit_results)
    
    # Commit volume changes
    volume.commit()
    
    return audit_results


def identify_paper_worthy_examples(fn_analysis_list):
    """Identify paper-worthy examples through outlier detection and pattern analysis."""
    
    from collections import defaultdict
    
    paper_examples = {
        'perfect_consistency_confound': [],
        'perfect_lexical_diversity': [],  
        'tau_dependent_behavior': [],
        'cluster_volatility': [],
        'threshold_sensitivity': [],
        'mixed_anomalous': []
    }
    
    for fn in fn_analysis_list:
        if not fn.get('tau_specific_data'):
            continue
            
        # Extract metrics across all taus for this FN
        tau_data = fn['tau_specific_data']
        appears_in_taus = fn['appears_in_taus']
        duplicate_rate = fn['duplicate_rate']
        refusal_template_rate = fn.get('refusal_template_rate', 0.0)
        
        # Get cluster counts and SE scores across taus
        cluster_counts = [tau_data[tau]['cluster_count'] for tau in appears_in_taus 
                         if tau_data[tau]['cluster_count'] is not None]
        se_scores = [tau_data[tau]['se_score'] for tau in appears_in_taus]
        thresholds = [tau_data[tau]['threshold'] for tau in appears_in_taus]
        
        if not cluster_counts:
            continue
            
        # Calculate metrics
        min_clusters = min(cluster_counts)
        max_clusters = max(cluster_counts) 
        cluster_range = max_clusters - min_clusters
        avg_clusters = sum(cluster_counts) / len(cluster_counts)
        
        # Distance from threshold (normalized)
        threshold_distances = [(score - thresh) / thresh for score, thresh in zip(se_scores, thresholds)]
        min_threshold_distance = min(threshold_distances)
        
        # Categorize examples
        example_data = {
            'prompt_id': fn['prompt_id'],
            'duplicate_rate': duplicate_rate,
            'refusal_template_rate': refusal_template_rate,
            'cluster_counts': cluster_counts,
            'min_clusters': min_clusters,
            'max_clusters': max_clusters,
            'cluster_range': cluster_range,
            'avg_clusters': avg_clusters,
            'appears_in_taus': appears_in_taus,
            'se_scores': se_scores,
            'threshold_distances': threshold_distances,
            'min_threshold_distance': min_threshold_distance,
            'classification': fn['classification'],
            'responses': fn.get('responses', [])[:3],  # First 3 responses for examples
            'score': 0  # Will be set based on category
        }
        
        # 1. Perfect Consistency Confound: High consistency (dup rate OR refusal templates) + low clusters
        if (duplicate_rate > 0.8 or refusal_template_rate > 0.8) and max_clusters <= 2 and len(appears_in_taus) >= 2:
            # Score based on both types of consistency + low clusters
            consistency_score = max(duplicate_rate, refusal_template_rate)
            example_data['score'] = consistency_score + (1.0 / (avg_clusters + 0.1))  # Higher score for lower clusters
            paper_examples['perfect_consistency_confound'].append(example_data)
            
        # 2. Perfect Lexical Diversity: Low dup rate + high clusters, split by template usage  
        elif duplicate_rate < 0.2 and min_clusters >= 3:
            diversity_score = (1.0 - duplicate_rate) + (avg_clusters / 5.0)  # Higher score for more clusters
            
            # Bonus for interesting template patterns in diverse responses
            if refusal_template_rate > 0.8:  # High template usage despite diversity
                example_data['score'] = diversity_score + 0.5  # Bonus for templates + diversity
                paper_examples['perfect_lexical_diversity'].append(example_data)
            elif refusal_template_rate < 0.2:  # True diversity without templates
                example_data['score'] = diversity_score + 0.3  # Bonus for genuine diversity
                paper_examples['perfect_lexical_diversity'].append(example_data)
            else:
                example_data['score'] = diversity_score
                paper_examples['perfect_lexical_diversity'].append(example_data)
            
        # 3. Tau-Dependent Behavior: Different patterns at different taus
        elif len(appears_in_taus) >= 3 and cluster_range >= 2:
            example_data['score'] = cluster_range + len(appears_in_taus) * 0.5
            paper_examples['tau_dependent_behavior'].append(example_data)
            
        # 4. Cluster Volatility: Dramatic cluster differences across taus
        elif cluster_range >= 3:
            example_data['score'] = cluster_range + (1.0 if len(appears_in_taus) >= 3 else 0)
            paper_examples['cluster_volatility'].append(example_data)
            
        # 5. Threshold Sensitivity: Very close to decision boundary
        elif abs(min_threshold_distance) < 0.1:  # Within 10% of threshold
            example_data['score'] = 1.0 / (abs(min_threshold_distance) + 0.01)  # Closer = higher score
            paper_examples['threshold_sensitivity'].append(example_data)
            
        # 6. Mixed/Anomalous: Everything else interesting
        else:
            # Score based on how "unusual" the pattern is, including refusal template inconsistency
            template_inconsistency = abs(refusal_template_rate - 0.5)  # Distance from 50% templates
            unusualness = (cluster_range * 0.5 + 
                          abs(duplicate_rate - 0.5) +  # Distance from middle duplicate rate
                          template_inconsistency * 0.3 +  # Template pattern unusualness 
                          len(appears_in_taus) * 0.2)   # More taus = more interesting
            if unusualness > 1.0:  # Only include if sufficiently unusual
                example_data['score'] = unusualness
                paper_examples['mixed_anomalous'].append(example_data)
    
    # Sort each category by score and keep top examples
    for category in paper_examples:
        paper_examples[category].sort(key=lambda x: x['score'], reverse=True)
        paper_examples[category] = paper_examples[category][:3]  # Top 3 per category
        
    return paper_examples


def generate_h6_report(results: dict):
    """Generate H6 qualitative audit report with multi-tau analysis."""
    
    import logging
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    model_name = results.get('model', 'unknown')
    dataset_name = results.get('dataset', 'unknown')
    
    report_path = Path(f'/research_storage/reports/{model_name}_{dataset_name.replace("-", "_")}_h6_qualitative_audit.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# H6 Qualitative Audit Report\n\n")
        f.write(f"**Model**: {model_name}  \n")
        f.write(f"**Dataset**: {dataset_name}  \n")
        f.write(f"**Analysis Date**: {results.get('analysis_date', 'Unknown')}  \n\n")
        
        f.write("## Executive Summary\n\n")
        
        summary = results.get('summary', {})
        consistency_rate = summary.get('consistency_confound_rate', 0)
        
        if results.get('h6_supported', False):
            f.write("**H6 Hypothesis Status: ✅ SUPPORTED**\n\n")
            f.write(f"More than 80% ({consistency_rate:.1%}) of SE false negatives exhibit the ")
            f.write("'Consistency Confound' pattern, confirming that SE failures are primarily ")
            f.write("due to high similarity between model responses rather than genuine ")
            f.write("semantic inconsistency detection.\n\n")
        else:
            f.write("**H6 Hypothesis Status: ❌ NOT SUPPORTED**\n\n")
            f.write(f"Only {consistency_rate:.1%} of SE false negatives fit the Consistency Confound ")
            f.write("pattern, suggesting that SE failures have diverse causes beyond response similarity.\n\n")
        
        # Multi-tau methodology
        f.write("## Audit Methodology\n\n")
        tau_values = results.get('tau_values', [0.1, 0.2, 0.3, 0.4])
        f.write(f"1. **Multi-tau Analysis**: Analyzed τ values {tau_values}\n")
        f.write(f"2. **Dataset**: {results.get('n_samples', 0)} samples ({results.get('n_harmful', 0)} harmful)\n")
        f.write("3. **FN Selection Strategies**:\n")
        f.write("   - **All Unique**: All FNs appearing at any τ value\n")
        f.write("   - **Common Multi-tau**: FNs appearing at ≥2 τ values\n")
        f.write("4. **Classification Criteria**:\n")
        f.write("   - **Consistency Confound**: High embedding duplicate rate (>50%) OR high refusal templates (>60%) + Low clusters (≤2) across all τ\n")
        f.write("   - **Lexical Diversity**: Low embedding duplicates (<20%) + High clusters (≥3)\n")
        f.write("   - **Mixed/Other**: Cases that don't fit clear patterns\n\n")
        
        # Tau-specific results
        f.write("## Multi-Tau Analysis Results\n\n")
        tau_specific = results.get('tau_specific_results', {})
        
        f.write("| τ | Threshold | FPR | TPR | False Negatives |\n")
        f.write("|---|-----------|-----|-----|----------------|\n")
        
        for tau in tau_values:
            if tau in tau_specific:
                data = tau_specific[tau]
                f.write(f"| {tau} | {data['optimal_threshold']:.4f} | {data['actual_fpr']:.3f} | {data['actual_tpr']:.3f} | {data['n_false_negatives']} |\n")
        
        f.write("\n**Key Findings**:\n")
        f.write(f"- **Total Unique FNs**: {results.get('n_unique_false_negatives', 0)}\n")
        f.write(f"- **Common Multi-tau FNs**: {results.get('n_common_false_negatives', 0)}\n")
        
        # FN Selection Strategy Analysis
        strategy_analysis = summary.get('strategy_analysis', {})
        if strategy_analysis:
            f.write("\n## FN Selection Strategy Comparison\n\n")
            
            f.write("| Strategy | FNs | Consistency Confound Rate | Description |\n")
            f.write("|----------|-----|--------------------------|-------------|\n")
            
            for strategy, data in strategy_analysis.items():
                f.write(f"| {strategy.replace('_', ' ').title()} | {data['n_fns']} | {data['consistency_confound_rate']:.1%} | {data.get('description', 'N/A')} |\n")
        
        f.write("## Detailed Results\n\n")
        
        f.write(f"**Dataset Statistics:**\n")
        f.write(f"- Total samples: {results.get('n_samples', 0)}\n")
        f.write(f"- Harmful samples: {results.get('n_harmful', 0)}\n")
        f.write(f"- Unique FNs analyzed: {results.get('n_unique_false_negatives', 0)}\n\n")
        
        f.write("**Classification Breakdown:**\n\n")
        classification_counts = summary.get('classification_counts', {})
        total_analyzed = summary.get('total_analyzed', 1)
        
        f.write("| Classification | Count | Percentage |\n")
        f.write("|---------------|-------|------------|\n")
        
        for classification, count in classification_counts.items():
            percentage = count / total_analyzed * 100
            f.write(f"| {classification.replace('_', ' ').title()} | {count} | {percentage:.1f}% |\n")
        
        f.write("\n**Key Findings:**\n")
        f.write(f"- **Consistency Confound Rate**: {consistency_rate:.1%}\n")
        f.write(f"- **Acceptance Threshold**: 80%\n")
        f.write(f"- **Hypothesis Result**: {'✅ Supported' if results.get('h6_supported') else '❌ Not Supported'}\n\n")
        
        # Paper-worthy examples section
        paper_examples = results.get('paper_examples', {})
        if paper_examples:
            f.write("## Paper-Worthy Examples & Outlier Analysis\n\n")
            f.write("These examples represent the most illustrative cases for understanding SE failure modes:\n\n")
            
            for category, examples in paper_examples.items():
                if examples:
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    
                    category_descriptions = {
                        'perfect_consistency_confound': 'High duplicate rate + consistently low clusters across all τ values',
                        'perfect_lexical_diversity': 'Low duplicate rate + consistently high clusters - SE should work here',
                        'tau_dependent_behavior': 'Different clustering patterns at different τ values',
                        'cluster_volatility': 'Dramatic cluster count differences across τ values',
                        'threshold_sensitivity': 'SE scores very close to decision boundaries',
                        'mixed_anomalous': 'Unusual patterns that don\'t fit standard categories'
                    }
                    
                    f.write(f"*{category_descriptions.get(category, 'Miscellaneous patterns')}*\n\n")
                    
                    for i, example in enumerate(examples, 1):
                        f.write(f"**{i}. {example['prompt_id']}** (Score: {example['score']:.2f})\n")
                        f.write(f"- Duplicate rate: {example['duplicate_rate']:.2f}\n")
                        f.write(f"- Refusal template rate: {example.get('refusal_template_rate', 0.0):.2f}\n")
                        f.write(f"- Cluster counts: {example['cluster_counts']}\n")
                        f.write(f"- Appears in τ values: {example['appears_in_taus']}\n")
                        f.write(f"- Classification: {example['classification'].replace('_', ' ').title()}\n")
                        
                        # Show sample responses for top examples
                        if i <= 2 and example.get('responses'):
                            f.write(f"- **Sample responses** (first 3):\n")
                            for j, resp in enumerate(example['responses'][:3], 1):
                                truncated = resp[:100] + "..." if len(resp) > 100 else resp
                                f.write(f"  {j}. {truncated}\n")
                        
                        f.write("\n")
        
        # Implications with multi-tau insights
        f.write("## Scientific Implications\n\n")
        
        if results.get('h6_supported'):
            f.write("### Consistency Confound is Dominant\n")
            f.write("The dominance of Consistency Confound failures across multiple τ values reveals:\n\n")
            f.write("- **Systematic Nature**: SE failures are predictable, not random measurement errors\n")
            f.write("- **Threshold Independence**: Pattern holds across different clustering thresholds\n")
            f.write("- **Fundamental Limitation**: SE measures response similarity, not semantic inconsistency\n")
            f.write("- **Detection vs. Measurement**: SE detects output diversity, not internal conflict\n\n")
            
            f.write("### Actionable Recommendations\n")
            f.write("1. **Preprocessing Filter**: Use response diversity metrics before SE calculation\n")
            f.write("2. **Hybrid Approaches**: Combine SE with orthogonal detection methods\n")
            f.write("3. **Threshold Adaptation**: Develop τ values that account for response similarity\n")
            f.write("4. **Model-Specific Calibration**: Adjust detection thresholds per model alignment level\n\n")
        else:
            f.write("### Complex Failure Landscape\n")
            f.write("The diversity of SE failure modes across τ values suggests:\n\n")
            f.write("- **Multiple Mechanisms**: SE failures have varied underlying causes\n")
            f.write("- **Threshold Sensitivity**: Performance varies significantly with τ selection\n")
            f.write("- **Context Dependency**: Effectiveness depends on prompt and response characteristics\n")
            f.write("- **Research Opportunity**: SE may capture genuine but complex semantic patterns\n\n")
            
            f.write("### Future Research Directions\n")
            f.write("1. **Mechanism Identification**: Investigate non-consistency-confound failure modes\n")
            f.write("2. **Adaptive Thresholding**: Develop context-aware τ selection methods\n")
            f.write("3. **Feature Engineering**: Extract additional signals from response patterns\n")
            f.write("4. **Comparative Studies**: Evaluate SE variants and alternative approaches\n\n")
    
    logger.info(f"✅ Report saved to: {report_path}")


@app.local_entrypoint()
def main(scores_file: str = None, responses_file: str = None, model_name: str = None, dataset_name: str = None):
    """Main entrypoint for H6 qualitative audit.
    
    Args:
        scores_file: Path to scores JSONL file
        responses_file: Path to responses JSONL file
        model_name: Optional model name
        dataset_name: Optional dataset name (H1/H2)
    """
    
    import sys
    
    # Parse command line arguments if not provided as function args
    if scores_file is None:
        if len(sys.argv) < 3:
            print("Usage: python run_h6_qualitative_audit_modal.py <scores_file> <responses_file> [model_name] [dataset_name]")
            print("\nExample:")
            print("  python run_h6_qualitative_audit_modal.py outputs/h1/llama4scout_120val_N5_temp0.7_top0.95_tokens1024_scores.jsonl outputs/h1/llama4scout_120val_N5_temp0.7_top0.95_tokens1024_responses.jsonl")
            return {'success': False, 'error': 'Missing required arguments'}
        
        scores_file = sys.argv[1]
        responses_file = sys.argv[2]
        model_name = sys.argv[3] if len(sys.argv) > 3 else None
        dataset_name = sys.argv[4] if len(sys.argv) > 4 else None
    
    print("=" * 100)
    print("H6 QUALITATIVE AUDIT ON MODAL")
    print("=" * 100)
    print(f"Scores file: {scores_file}")
    print(f"Responses file: {responses_file}")
    print(f"Model: {model_name or 'auto-detect'}")
    print(f"Dataset: {dataset_name or 'auto-detect'}")
    print("\nThis will:")
    print("1. Identify SE false negatives using τ=0.3 threshold")
    print("2. For each FN, calculate embedding-based duplicate rate & semantic cluster count")
    print("3. Classify failures: Consistency Confound vs Lexical Diversity vs Mixed")
    print("4. Test H6 hypothesis: >80% of FNs fit 'Consistency Confound' pattern")
    print("5. Generate comprehensive qualitative audit report")
    print("=" * 100)
    
    try:
        # Run audit analysis
        print("\n🔍 Running qualitative audit of SE false negatives...")
        results = run_h6_qualitative_audit.remote(scores_file, responses_file, model_name, dataset_name)
        
        if results.get('error') == 'no_false_negatives':
            print("\n⚠️ No false negatives found - cannot perform audit")
            print(f"Dataset had {results.get('n_samples', 0)} samples but SE performed perfectly")
            return {'success': True, 'no_false_negatives': True}
        
        summary = results.get('summary', {})
        
        print("\n" + "=" * 100)
        print("✅ H6 ANALYSIS COMPLETE!")
        print("=" * 100)
        print(f"Model: {results.get('model', 'llama-4-scout-17b-16e-instruct')}")
        print(f"Dataset: {results.get('dataset', 'H2')} ({results.get('n_samples', 0)} samples)")
        print(f"False negatives analyzed: {results.get('n_false_negatives', 0)}")
        print(f"Consistency Confound rate: {summary.get('consistency_confound_rate', 0):.1%}")
        
        classification_counts = summary.get('classification_counts', {})
        for classification, count in classification_counts.items():
            print(f"  {classification.replace('_', ' ').title()}: {count}")
        
        print(f"H6 hypothesis supported: {'✅' if results.get('h6_supported') else '❌'}")
        print("=" * 100)
        
        return {
            'success': True,
            'model': results.get('model'),
            'dataset': results.get('dataset'),
            'n_false_negatives': results.get('n_false_negatives'),
            'h6_supported': results.get('h6_supported'),
            'consistency_confound_rate': summary.get('consistency_confound_rate', 0),
            'classification_counts': classification_counts
        }
        
    except Exception as e:
        print(f"\n❌ H6 ANALYSIS FAILED: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    main()