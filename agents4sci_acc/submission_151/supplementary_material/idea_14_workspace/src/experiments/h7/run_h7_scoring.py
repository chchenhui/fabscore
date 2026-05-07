"""
H7 Scoring Script - Score responses from Qwen2.5-72B-Instruct
Adapted from H1 scoring to compute SE and baseline metrics
"""

import argparse
import json
import logging
from pathlib import Path
import modal
import yaml
import os
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "openai", "requests", "pyyaml", "numpy", "scikit-learn", 
    "sentence-transformers", "torch", "bert-score", "python-Levenshtein", "tqdm"
]).add_local_python_source("src").add_local_dir("configs", "/configs")

# Persistent storage volume for research outputs
volume = modal.Volume.from_name("alignment-research-storage", create_if_missing=True)

app = modal.App("h7-scoring")

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,  # 2 hours
    volumes={"/research_storage": volume}
)
def compute_h7_scores(model_short: str, test_mode=False):
    """Compute H7 scores using consistent configuration
    
    Args:
        model_short: Short model name (e.g., 'qwen-2.5-72b-instruct')
        test_mode: If True, process test file instead of full dataset
    """
    from src.core.semantic_entropy import SemanticEntropy
    from src.core.baseline_metrics import BaselineMetrics
    import yaml
    import json
    import logging
    import os
    import numpy as np
    from tqdm import tqdm
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    with open('/configs/project_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get H7 configuration
    h7_config = config['hypotheses']['h7']
    embedding_model = h7_config['embedding_model']
    tau_grid = h7_config['tau_grid']
    
    logging.info("="*80)
    logging.info("H7 SCORING - SOTA MODEL CHECK")
    logging.info("="*80)
    logging.info(f"📊 Configuration:")
    logging.info(f"   - Embedding model: {embedding_model}")
    logging.info(f"   - Tau grid: {tau_grid}")
    logging.info("="*80)
    
    # Setup input/output paths following H2 convention
    if test_mode:
        input_file = f"/research_storage/outputs/h7/{model_short}_h7_TEST_responses.jsonl"
        output_file = f"/research_storage/outputs/h7/{model_short}_h7_TEST_scores.jsonl"
    else:
        input_file = f"/research_storage/outputs/h7/{model_short}_h7_responses.jsonl"
        output_file = f"/research_storage/outputs/h7/{model_short}_h7_scores.jsonl"
    
    # Initialize scorers
    se_scorer = SemanticEntropy(embedding_model)
    baseline_scorer = BaselineMetrics(embedding_model)
    
    # Load responses
    responses_data = []
    with open(input_file, 'r') as f:
        for line in f:
            responses_data.append(json.loads(line))
    
    logging.info(f"📁 Loaded {len(responses_data)} samples from {input_file}")
    
    # Process each sample with detailed logging (following H2 pattern)
    scored_data = []
    successful_scores = 0
    failed_scores = []
    
    for idx, data in enumerate(responses_data):
        prompt_id = data['prompt_id']
        prompt = data['prompt']
        label = data['label']
        responses = data['responses']
        
        logging.info(f"\n[{idx+1:3d}/{len(responses_data)}] Scoring {prompt_id}")
        logging.info(f"   Label: {'harmful' if label == 1 else 'benign'}")
        logging.info(f"   Responses: {len(responses)} samples")
        
        # Response validation (following H2 pattern)
        valid_responses = [r for r in responses if r and r.strip()]
        if len(valid_responses) < 2:
            logging.warning(f"   ⚠️ Only {len(valid_responses)} valid responses, skipping")
            failed_scores.append({
                'prompt_id': prompt_id,
                'reason': f'insufficient_responses_{len(valid_responses)}'
            })
            continue
        
        try:
            scores = {
                'prompt_id': prompt_id,
                'label': label,
                'n_responses': len(valid_responses),
            }
            
            # Compute Semantic Entropy for all τ values with diagnostics
            logging.info(f"   🧠 Computing Semantic Entropy...")
            se_scores = {}
            se_diagnostics = {}
            for tau in tau_grid:
                entropy_result = se_scorer.calculate_entropy(
                    valid_responses, 
                    distance_threshold=tau,
                    return_diagnostics=True
                )
                
                if isinstance(entropy_result, dict):
                    se_score = entropy_result.get('semantic_entropy', 0.0)
                    diagnostics = entropy_result
                else:
                    se_score = entropy_result
                    diagnostics = {}
                
                se_scores[f'tau_{tau}'] = se_score
                se_diagnostics[f'tau_{tau}'] = diagnostics
                
                # Enhanced diagnostic logging
                clusters = diagnostics.get('num_clusters', 'N/A')
                logging.info(f"      τ={tau}: SE={se_score:.6f}, clusters={clusters}")
            
            scores['semantic_entropy'] = se_scores
            scores['semantic_entropy_diagnostics'] = se_diagnostics
            
            # Compute baseline metrics (following H2 pattern)
            logging.info(f"   📏 Computing baseline metrics...")
            baseline_scores = baseline_scorer.calculate_metrics(valid_responses)
            
            # Extract individual baseline scores
            scores['avg_pairwise_bertscore'] = baseline_scores['avg_pairwise_bertscore']
            scores['embedding_variance'] = baseline_scores['embedding_variance']
            scores['levenshtein_variance'] = baseline_scores['levenshtein_variance']
            
            logging.info(f"      Avg BERTScore: {baseline_scores['avg_pairwise_bertscore']:.6f}")
            logging.info(f"      Embedding Variance: {baseline_scores['embedding_variance']:.6f}")
            logging.info(f"      Levenshtein Variance: {baseline_scores['levenshtein_variance']:.6f}")
            
            # Add response metadata for analysis (following H2 pattern)
            response_lengths = [len(r) for r in valid_responses]
            avg_length = sum(response_lengths) / len(response_lengths)
            scores['response_metadata'] = {
                'avg_length': avg_length,
                'min_length': min(response_lengths),
                'max_length': max(response_lengths),
                'length_variance': sum((l - avg_length)**2 for l in response_lengths) / len(response_lengths),
                'empty_responses': sum(1 for r in responses if not r or not r.strip())
            }
            
            # Preserve original metadata (following H2 pattern)
            scores['original_metadata'] = {
                k: v for k, v in data.items() 
                if k not in ['prompt_id', 'responses', 'label']
            }
            
            scored_data.append(scores)
            successful_scores += 1
            
            logging.info(f"   ✅ Successfully scored all metrics")
            
        except Exception as e:
            error_msg = f"Scoring failed: {str(e)}"
            logging.error(f"   ❌ {error_msg}")
            
            failed_scores.append({
                'prompt_id': prompt_id,
                'reason': error_msg,
                'n_responses': len(valid_responses)
            })
        
        # Progress logging every 20 prompts (following H2 pattern)
        if (idx + 1) % 20 == 0:
            success_rate = successful_scores / (idx + 1) * 100
            logging.info(f"\n📊 PROGRESS UPDATE: {idx+1}/{len(responses_data)} processed")
            logging.info(f"   Success rate: {success_rate:.1f}% ({successful_scores} successful)")
            logging.info(f"   Failed scores: {len(failed_scores)}")
    
    # Save scored data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for item in scored_data:
            f.write(json.dumps(item) + '\n')
    
    # Final results summary (following H2 pattern)
    logging.info("\n" + "="*80)
    logging.info("H7 SCORING COMPLETE")
    logging.info("="*80)
    
    total_success_rate = successful_scores / len(responses_data) * 100
    
    logging.info(f"📊 FINAL STATISTICS:")
    logging.info(f"   Total response sets: {len(responses_data)}")
    logging.info(f"   Successfully scored: {successful_scores}")
    logging.info(f"   Failed scores: {len(failed_scores)}")
    logging.info(f"   Success rate: {total_success_rate:.1f}%")
    
    # Report any failures
    if failed_scores:
        logging.warning(f"\n⚠️ FAILED SCORES ({len(failed_scores)}):")
        for failure in failed_scores:
            logging.warning(f"   {failure['prompt_id']}: {failure['reason']}")
    
    # Extract scores for meaningful analysis (fix data structure access)
    if scored_data:
        harmful_scores = {'tau_0.1': [], 'tau_0.2': [], 'avg_pairwise_bertscore': [], 
                         'embedding_variance': [], 'levenshtein_variance': []}
        benign_scores = {'tau_0.1': [], 'tau_0.2': [], 'avg_pairwise_bertscore': [], 
                        'embedding_variance': [], 'levenshtein_variance': []}
        
        for item in scored_data:
            scores_dict = harmful_scores if item['label'] == 1 else benign_scores
            
            # Extract SE scores
            if 'semantic_entropy' in item:
                for tau_key in ['tau_0.1', 'tau_0.2']:
                    if tau_key in item['semantic_entropy'] and item['semantic_entropy'][tau_key] is not None:
                        scores_dict[tau_key].append(item['semantic_entropy'][tau_key])
            
            # Extract baseline scores
            for metric in ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']:
                if metric in item and item[metric] is not None:
                    scores_dict[metric].append(item[metric])
        
        # Display statistics
        logging.info(f"\n📈 SCORE ANALYSIS:")
        for metric in ['tau_0.1', 'tau_0.2', 'avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']:
            if harmful_scores[metric] and benign_scores[metric]:
                harmful_mean = np.mean(harmful_scores[metric])
                benign_mean = np.mean(benign_scores[metric])
                logging.info(f"{metric}:")
                logging.info(f"  Harmful mean: {harmful_mean:.6f} (n={len(harmful_scores[metric])})")
                logging.info(f"  Benign mean: {benign_mean:.6f} (n={len(benign_scores[metric])})")
                logging.info(f"  Difference: {abs(harmful_mean - benign_mean):.6f}")
    
    logging.info("="*80)
    logging.info(f"✅ H7 Scoring completed! Results saved to: {output_file}")
    logging.info(f"📊 Total samples scored: {len(scored_data)}")
    logging.info("="*80)
    
    # Generate comprehensive scoring report (following H2 pattern)
    logging.info("📄 Generating H7 scoring report...")
    report_file = output_file.replace('_scores.jsonl', '_scoring_report.md')
    
    # Calculate detailed statistics for report
    harmful_count = sum(1 for item in scored_data if item['label'] == 1)
    benign_count = sum(1 for item in scored_data if item['label'] == 0)
    
    # Calculate score averages for report
    score_stats = {}
    if scored_data:
        # SE averages
        for tau_key in ['tau_0.1', 'tau_0.2', 'tau_0.3', 'tau_0.4']:
            values = [item['semantic_entropy'].get(tau_key, 0) for item in scored_data 
                     if 'semantic_entropy' in item and item['semantic_entropy'].get(tau_key) is not None]
            if values:
                score_stats[f'se_{tau_key}'] = sum(values) / len(values)
        
        # Baseline averages
        for metric in ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']:
            values = [item[metric] for item in scored_data if item.get(metric) is not None]
            if values:
                score_stats[metric] = sum(values) / len(values)
    
    # Generate report content
    report_content = f"""# H7 Scoring Report - {model_short}

## Summary

- **Model:** {model_short}
- **Input file:** {input_file}
- **Total response sets:** {len(responses_data)}
- **Successfully scored:** {successful_scores}
- **Failed scores:** {len(failed_scores)}
- **Success rate:** {total_success_rate:.1f}%
- **Output file:** {output_file}

## Scoring Methods

### Semantic Entropy
- **τ grid:** {tau_grid}
- **Embedding model:** {embedding_model}
- **Method:** Agglomerative clustering with cosine distance
- **Diagnostics captured:** Cluster counts, embedding matrices, distance thresholds

### Baseline Metrics
- **Avg Pairwise BERTScore:** Mean pairwise BERTScore F1 across response sets
- **Embedding Variance:** Variance of sentence embeddings within response sets
- **Levenshtein Variance:** Variance of edit distances within response sets

## Dataset Composition

- **Harmful samples:** {harmful_count}
- **Benign samples:** {benign_count}
- **Total samples:** {len(scored_data)}

## Score Statistics (Averages)

"""

    # Add score statistics to report
    for metric, avg_value in score_stats.items():
        if metric.startswith('se_'):
            tau_val = metric.replace('se_tau_', '')
            report_content += f"- **SE (τ={tau_val}) average:** {avg_value:.6f}\\n"
        else:
            metric_name = metric.replace('_', ' ').title()
            report_content += f"- **{metric_name} average:** {avg_value:.6f}\\n"
    
    # Add failure analysis if any
    if failed_scores:
        report_content += f"""
## Failed Scores Analysis

- **Total failures:** {len(failed_scores)}

### Failure Breakdown:
"""
        failure_reasons = {}
        for failure in failed_scores:
            reason = failure['reason']
            if reason not in failure_reasons:
                failure_reasons[reason] = []
            failure_reasons[reason].append(failure['prompt_id'])
        
        for reason, prompt_ids in failure_reasons.items():
            report_content += f"- **{reason}:** {len(prompt_ids)} samples\\n"
            if len(prompt_ids) <= 5:
                report_content += f"  - Affected prompts: {', '.join(prompt_ids)}\\n"
            else:
                report_content += f"  - Affected prompts: {', '.join(prompt_ids[:5])}, ... and {len(prompt_ids)-5} more\\n"
    
    report_content += f"""
## Technical Details

- **Processing time:** Detailed per-prompt timing logged
- **Response validation:** Minimum 2 valid responses required per sample
- **Diagnostic data:** Complete SE clustering information preserved
- **Metadata preservation:** All original generation metadata retained

## Output Structure

The scoring output file contains:
- Individual prompt scores with diagnostics
- Semantic entropy values for each τ threshold
- Complete baseline metric calculations
- Response metadata (length statistics, quality metrics)
- Original generation metadata preservation

## Next Steps

1. Run H7 evaluation pipeline: `modal run src/experiments/h7/run_h7_evaluation.py::main --model={model_short}`
2. Compare results with H1 baseline experiments
3. Validate H7 success criteria for SOTA model performance
"""

    # Write report to Modal volume
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    logging.info(f"📄 Scoring report generated: {report_file}")
    
    # Volume commit for persistence
    volume.commit()
    
    return output_file

@app.local_entrypoint()
def main(model: str, test: bool = False):
    """Entry point for H7 scoring
    
    Args:
        model: Model short name (e.g., qwen-2.5-72b-instruct, llama-3.3-70b-instruct)
        test: Process test file (10 samples) if True
    """
    logging.info("🚀 Starting H7 Scoring (SOTA Model Check)")
    logging.info(f"Model: {model}")
    logging.info(f"Mode: {'TEST (10 samples)' if test else 'FULL (120 samples)'}")
    
    # Run the scoring
    output_file = compute_h7_scores.remote(model_short=model, test_mode=test)
    
    logging.info(f"✅ H7 Scoring completed!")
    logging.info(f"📁 Results saved to: {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_h7_scoring.py <model_name> [--test]")
        sys.exit(1)
    
    model_name = sys.argv[1]
    test_mode = "--test" in sys.argv
    main(model_name, test_mode)