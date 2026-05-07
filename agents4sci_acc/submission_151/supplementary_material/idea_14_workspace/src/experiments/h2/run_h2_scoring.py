
#!/usr/bin/env python3
"""
H2 Scoring - Score H2 twins responses with SE and baseline metrics on Modal.
Updated for new methodology: test τ grid instead of frozen parameters, include baselines.
"""

import modal
import json
import logging
import os
from pathlib import Path
from typing import Dict, List
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modal setup - consistent with existing infrastructure
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "sentence-transformers", 
    "scikit-learn", 
    "numpy", 
    "torch",
    "bert-score",
    "python-Levenshtein",
    "pyyaml"
]).add_local_python_source("src").add_local_dir("configs", "/configs").add_local_dir("data", "/data")

# Use same persistent storage volume
volume = modal.Volume.from_name("alignment-research-storage", create_if_missing=True)

app = modal.App("h2-scoring")

@app.function(
    image=image,
    gpu="A100-40GB",  # GPU for embedding calculations
    timeout=7200,  # 2 hours
    volumes={"/research_storage": volume}
)
def score_h2_responses(model_short: str):
    """Score H2 responses with SE (multiple τ) and baseline metrics."""
    from src.core.semantic_entropy import SemanticEntropy
    from src.core.baseline_metrics import BaselineMetrics
    import json
    import logging
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 100)
    logger.info(f"H2 SCORING - {model_short}")
    logger.info("=" * 100)
    
    # Load config
    try:
        with open('/configs/project_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Loaded project configuration")
    except Exception as e:
        logger.warning(f"⚠️ Could not load config: {e}, using defaults")
        config = {}
    
    # Define paths
    input_path = Path(f'/research_storage/outputs/h2/{model_short}_h2_responses.jsonl')
    output_dir = Path('/research_storage/outputs/h2/scoring')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_short}_h2_scores.jsonl"
    
    logger.info(f"📁 Input: {input_path}")
    logger.info(f"📁 Output: {output_path}")
    
    # Check if response file exists
    if not input_path.exists():
        raise FileNotFoundError(f"H2 response file not found at {input_path}. "
                               "Run run_h2_response_generation.py first.")
    
    # Load responses
    responses_data = []
    try:
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    responses_data.append(data)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON error on line {line_num}: {e}")
                    
        logger.info(f"✅ Loaded {len(responses_data)} response sets")
    except Exception as e:
        logger.error(f"❌ Failed to load responses: {e}")
        raise
    
    # Analyze response data
    harmful_count = sum(1 for item in responses_data if item['label'] == 1)
    benign_count = sum(1 for item in responses_data if item['label'] == 0)
    
    logger.info(f"📊 Response composition: {harmful_count} harmful + {benign_count} benign = {len(responses_data)} total")
    
    # Setup scoring parameters (per new methodology - no frozen params)
    # Try multiple config locations for tau_grid
    tau_grid = None
    if 'methods' in config and 'semantic_entropy' in config['methods']:
        tau_grid = config['methods']['semantic_entropy'].get('tau_grid')
    elif 'tuning' in config:
        tau_grid = config['tuning'].get('tau_grid')
    elif 'scoring' in config:
        tau_grid = config['scoring'].get('semantic_entropy_tau_grid')
    
    if tau_grid is None:
        tau_grid = [0.1, 0.2, 0.3, 0.4]  # Default fallback
        
    embedding_model = config.get('hypotheses', {}).get('h2', {}).get('embedding_model', 'Alibaba-NLP/gte-large-en-v1.5')
    
    logger.info(f"⚙️ Scoring parameters:")
    logger.info(f"   Semantic Entropy τ grid: {tau_grid}")
    logger.info(f"   Embedding model: {embedding_model}")
    
    # Initialize scoring calculators
    logger.info("🔧 Initializing scoring calculators...")
    se_calculator = SemanticEntropy(embedding_model)
    baseline_calculator = BaselineMetrics(embedding_model)
    logger.info("✅ Calculators initialized")
    
    # Score all responses
    logger.info("🚀 Starting scoring process...")
    
    scored_responses = []
    successful_scores = 0
    failed_scores = []
    
    for idx, item in enumerate(responses_data):
        prompt_id = item['prompt_id']
        responses = item['responses']
        label = item['label']
        
        logger.info(f"\n[{idx+1:3d}/{len(responses_data)}] Scoring {prompt_id}")
        logger.info(f"   Label: {'harmful' if label == 1 else 'benign'}")
        logger.info(f"   Responses: {len(responses)} samples")
        
        # Check response quality
        valid_responses = [r for r in responses if r and r.strip()]
        if len(valid_responses) < 2:
            logger.warning(f"   ⚠️ Only {len(valid_responses)} valid responses, skipping")
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
            
            # Calculate Semantic Entropy for all τ values with diagnostics
            logger.info(f"   🧠 Computing Semantic Entropy...")
            se_scores = {}
            se_diagnostics = {}
            for tau in tau_grid:
                entropy_result = se_calculator.calculate_entropy(
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
                
                clusters = diagnostics.get('num_clusters', 'N/A')
                logger.info(f"      τ={tau}: SE={se_score:.6f}, clusters={clusters}")
            
            scores['semantic_entropy'] = se_scores
            scores['semantic_entropy_diagnostics'] = se_diagnostics
            
            # Calculate baseline metrics (matching H1 scoring approach)
            logger.info(f"   📏 Computing baseline metrics...")
            baseline_scores = baseline_calculator.calculate_metrics(valid_responses)
            
            # Extract individual scores
            scores['avg_pairwise_bertscore'] = baseline_scores['avg_pairwise_bertscore']
            scores['embedding_variance'] = baseline_scores['embedding_variance'] 
            scores['levenshtein_variance'] = baseline_scores['levenshtein_variance']
            
            logger.info(f"      Avg BERTScore: {baseline_scores['avg_pairwise_bertscore']:.6f}")
            logger.info(f"      Embedding Variance: {baseline_scores['embedding_variance']:.6f}")
            logger.info(f"      Levenshtein Variance: {baseline_scores['levenshtein_variance']:.6f}")
            
            # Add response metadata for analysis
            response_lengths = [len(r) for r in valid_responses]
            avg_length = sum(response_lengths) / len(response_lengths)
            scores['response_metadata'] = {
                'avg_length': avg_length,
                'min_length': min(response_lengths),
                'max_length': max(response_lengths),
                'length_variance': sum((l - avg_length)**2 for l in response_lengths) / len(response_lengths)
            }
            
            # Preserve original metadata
            scores['original_metadata'] = {
                k: v for k, v in item.items() 
                if k not in ['prompt_id', 'responses', 'label']
            }
            
            scored_responses.append(scores)
            successful_scores += 1
            
            logger.info(f"   ✅ Successfully scored all metrics")
            
        except Exception as e:
            error_msg = f"Scoring failed: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            
            failed_scores.append({
                'prompt_id': prompt_id,
                'reason': error_msg,
                'n_responses': len(valid_responses)
            })
        
        # Progress logging every 20 prompts
        if (idx + 1) % 20 == 0:
            success_rate = successful_scores / (idx + 1) * 100
            logger.info(f"\n📊 PROGRESS UPDATE: {idx+1}/{len(responses_data)} processed")
            logger.info(f"   Success rate: {success_rate:.1f}% ({successful_scores} successful)")
            logger.info(f"   Failed scores: {len(failed_scores)}")
    
    # Final results
    logger.info("\n" + "=" * 100)
    logger.info("H2 SCORING COMPLETE")
    logger.info("=" * 100)
    
    total_success_rate = successful_scores / len(responses_data) * 100
    
    logger.info(f"📊 FINAL STATISTICS:")
    logger.info(f"   Total response sets: {len(responses_data)}")
    logger.info(f"   Successfully scored: {successful_scores}")
    logger.info(f"   Failed scores: {len(failed_scores)}")
    logger.info(f"   Success rate: {total_success_rate:.1f}%")
    logger.info(f"   Output samples: {len(scored_responses)}")
    
    # Save scores
    try:
        with open(output_path, 'w') as f:
            for record in scored_responses:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"✅ Scores saved to {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save scores: {e}")
        raise
    
    # Generate detailed scoring report
    report_path = output_dir / f"{model_short}_h2_scoring_report.md"
    try:
        with open(report_path, 'w') as f:
            f.write(f"# H2 Scoring Report - {model_short}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Model:** {model_short}\n")
            f.write(f"- **Input file:** {input_path}\n")
            f.write(f"- **Total response sets:** {len(responses_data)}\n")
            f.write(f"- **Successfully scored:** {successful_scores}\n")
            f.write(f"- **Success rate:** {total_success_rate:.1f}%\n")
            f.write(f"- **Output file:** {output_path}\n\n")
            
            f.write("## Scoring Methods\n\n")
            f.write("### Semantic Entropy\n")
            f.write(f"- **τ grid:** {tau_grid}\n")
            f.write(f"- **Embedding model:** {embedding_model}\n")
            f.write("- **Method:** Agglomerative clustering with cosine distance\n\n")
            
            f.write("### Baseline Metrics\n")
            f.write("- **Avg Pairwise BERTScore:** Mean pairwise BERTScore F1\n")
            f.write("- **Embedding Variance:** Variance of sentence embeddings\n")
            f.write("- **Levenshtein Variance:** Variance of edit distances\n\n")
            
            f.write(f"## Dataset Composition\n\n")
            f.write(f"- **Harmful samples:** {harmful_count}\n")
            f.write(f"- **Benign samples:** {benign_count}\n")
            
            if failed_scores:
                f.write(f"\n## Failed Scores ({len(failed_scores)})\n\n")
                f.write("| Prompt ID | Failure Reason |\n")
                f.write("|-----------|----------------|\n")
                for fail in failed_scores[:20]:  # First 20
                    f.write(f"| {fail['prompt_id']} | {fail['reason']} |\n")
            
            # Generate sample score statistics
            if scored_responses:
                f.write(f"\n## Score Statistics (Sample)\n\n")
                
                # SE stats for first τ
                first_tau = tau_grid[0]
                se_scores = [s['semantic_entropy'][f'tau_{first_tau}'] for s in scored_responses]
                avg_se = sum(se_scores) / len(se_scores)
                f.write(f"- **SE (τ={first_tau}) average:** {avg_se:.6f}\n")
                
                # Baseline stats
                bertscore_scores = [s['avg_pairwise_bertscore'] for s in scored_responses]
                avg_bert = sum(bertscore_scores) / len(bertscore_scores)
                f.write(f"- **Avg BERTScore average:** {avg_bert:.6f}\n")
                
                emb_var_scores = [s['embedding_variance'] for s in scored_responses]
                avg_emb_var = sum(emb_var_scores) / len(emb_var_scores)
                f.write(f"- **Embedding Variance average:** {avg_emb_var:.6f}\n")
            
        logger.info(f"✅ Scoring report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save scoring report: {e}")
    
    # Commit volume changes
    try:
        volume.commit()
        logger.info("✅ Volume changes committed")
    except Exception as e:
        logger.error(f"❌ Volume commit failed: {e}")
    
    return {
        'success': True,
        'model_short': model_short,
        'total_response_sets': len(responses_data),
        'successful_scores': successful_scores,
        'success_rate': total_success_rate,
        'output_path': str(output_path),
        'tau_grid': tau_grid
    }

@app.local_entrypoint()
def main(model_short: str = "qwen-2.5-7b-instruct"):
    """Main entrypoint for H2 scoring."""
    
    print("=" * 100)
    print("H2 SCORING ON MODAL")
    print("=" * 100)
    print(f"Model: {model_short}")
    print("This will:")
    print("1. Load H2 response data for the specified model")
    print("2. Score with Semantic Entropy (τ ∈ {0.1,0.2,0.3,0.4})")
    print("3. Score with baseline metrics (BERTScore, Embedding Variance)")
    print("4. Save comprehensive scoring data with metadata")
    print("5. Generate detailed scoring report")
    print("=" * 100)
    
    try:
        result = score_h2_responses.remote(model_short)
        
        print("\n" + "=" * 100)
        print("✅ H2 SCORING COMPLETE!")
        print("=" * 100)
        print(f"Model: {result['model_short']}")
        print(f"Success rate: {result['success_rate']:.1f}%")
        print(f"Scored: {result['successful_scores']}/{result['total_response_sets']} response sets")
        print(f"τ grid: {result['tau_grid']}")
        print(f"Output: {result['output_path']}")
        print("=" * 100)
        
        return result
        
    except Exception as e:
        print(f"\n❌ H2 SCORING FAILED: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen-2.5-7b-instruct"
    main(model)
