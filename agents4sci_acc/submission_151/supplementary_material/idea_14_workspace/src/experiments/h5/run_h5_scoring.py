#!/usr/bin/env python3
"""
H5 Scoring - Score H5 paraphrased responses with SE and baseline metrics on Modal.
Following the exact pattern from H2 scoring but adapted for H5 robustness analysis.
"""

import modal
import json
import logging
import os
from pathlib import Path
from typing import Dict, List
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modal setup - consistent with H1/H2 infrastructure
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

app = modal.App("h5-scoring")

@app.function(
    image=image,
    gpu="A100-40GB",  # GPU for embedding calculations
    timeout=7200,  # 2 hours
    volumes={"/research_storage": volume}
)
def score_h5_responses(model_short: str, test_mode: bool = False):
    """Score H5 paraphrased responses with SE (multiple τ) and baseline metrics."""
    from src.core.semantic_entropy import SemanticEntropy
    from src.core.baseline_metrics import BaselineMetrics
    import json
    import logging
    from pathlib import Path
    import time
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 100)
    logger.info(f"H5 SCORING - {model_short.upper()} - PARAPHRASED RESPONSES")
    logger.info("=" * 100)
    
    # Load configuration
    with open('/configs/project_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get H5 and method configuration
    h5_config = config['hypotheses']['h5']
    se_config = config['methods']['semantic_entropy']
    baseline_config = config['methods']['baselines']
    paths_config = h5_config['paths']
    
    logger.info("🔧 H5 SCORING CONFIGURATION")
    logger.info(f"📂 Input responses: {paths_config['responses_dir']}")
    logger.info(f"📂 Score output: {paths_config['scores_dir']}")
    logger.info(f"📊 Semantic Entropy:")
    logger.info(f"   - τ grid: {se_config['tau_grid']}")
    logger.info(f"   - Embedding model: {se_config['embedding_model']}")
    logger.info(f"📊 Baseline Methods:")
    for method, method_config in baseline_config.items():
        logger.info(f"   - {method}: {method_config['method']}")
    
    # Set up paths from config
    input_dir = Path(paths_config['responses_dir'])
    output_dir = Path(paths_config['scores_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    
    # Fix model naming to match actual generated files
    test_prefix = "TEST_" if test_mode else ""
    if model_short == "qwen2.5-7b-instruct":
        input_filename = f"{test_prefix}qwen-qwen2.5-7b-instruct_h5_responses.jsonl"
        output_filename = f"{test_prefix}qwen-qwen2.5-7b-instruct_h5_scores.jsonl"
    elif model_short == "llama-4-scout-17b-16e-instruct":
        input_filename = f"{test_prefix}meta-llama-llama-4-scout-17b-16e-instruct_h5_responses.jsonl"
        output_filename = f"{test_prefix}meta-llama-llama-4-scout-17b-16e-instruct_h5_scores.jsonl"
    else:
        input_filename = f"{test_prefix}{model_short}_h5_responses.jsonl"
        output_filename = f"{test_prefix}{model_short}_h5_scores.jsonl"
    
    input_file = input_dir / input_filename
    output_file = output_dir / output_filename
    
    logger.info(f"📁 Input responses: {input_file}")
    logger.info(f"📁 Output scores: {output_file}")
    
    # Load H5 responses
    if not input_file.exists():
        raise FileNotFoundError(f"H5 responses not found: {input_file}")
    
    responses_data = []
    with open(input_file, 'r') as f:
        for line in f:
            responses_data.append(json.loads(line))
    
    logger.info(f"✅ Loaded {len(responses_data)} response records")
    
    # Apply test mode if requested
    if test_mode:
        responses_data = responses_data[:10]  # Take first 10 samples for testing
        logger.info(f"🧪 TEST MODE ACTIVATED: Processing first 10 samples only")
        logger.info(f"   Test subset: {len(responses_data)} samples")
    
    # Analyze dataset composition
    harmful_count = sum(1 for item in responses_data if item['label'] == 1)
    benign_count = sum(1 for item in responses_data if item['label'] == 0)
    
    logger.info(f"   Harmful: {harmful_count}, Benign: {benign_count}")
    
    # Check for existing scores (resume capability)
    already_scored = set()
    if output_file.exists():
        logger.info("📋 Found existing scores file, checking for resume capability...")
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    already_scored.add(item['prompt_id'])
                except:
                    continue
        if already_scored:
            logger.info(f"   Found {len(already_scored)} already scored samples")
            logger.info(f"   Will resume from where we left off")
    
    # Initialize scoring methods (same as H1/H2)
    logger.info("\n🔧 Initializing scoring methods...")
    
    # Semantic Entropy with τ grid testing (following H2 pattern exactly)
    se_calculator = SemanticEntropy(
        embedding_model_name=se_config['embedding_model']  # Fix parameter name
    )
    logger.info(f"✅ Semantic Entropy calculator initialized with model: {se_config['embedding_model']}")
    
    # Baseline methods (same as H1/H2)
    baseline_calculator = BaselineMetrics(
        embedding_model_name=se_config['embedding_model']  # Fix parameter name
    )
    logger.info("✅ Baseline metrics calculator initialized")
    
    # Process each response record
    samples_to_process = [item for item in responses_data if item['prompt_id'] not in already_scored]
    logger.info(f"\n🚀 Starting scoring process...")
    logger.info(f"   Total samples: {len(responses_data)}")
    logger.info(f"   Already scored: {len(already_scored)}")
    logger.info(f"   To process: {len(samples_to_process)}")
    
    total_processing_time = 0
    successful_scores = 0
    failed_scores = 0
    
    # Process with append mode for incremental saving (H2 pattern)
    mode = 'a' if already_scored else 'w'
    
    with open(output_file, mode) as outf:
        for i, response_record in enumerate(responses_data):
            prompt_id = response_record['prompt_id']
            
            # Skip if already scored
            if prompt_id in already_scored:
                logger.info(f"[{i+1:3d}/{len(responses_data)}] ⏭️  Skipping {prompt_id} (already scored)")
                continue
            
            logger.info(f"\n[{i+1:3d}/{len(responses_data)}] 🔄 Scoring {prompt_id}")
            logger.info(f"   Label: {'harmful' if response_record['label'] == 1 else 'benign'}")
            logger.info(f"   Responses: {len(response_record['responses'])}")
            
            start_time = time.time()
            
            try:
                responses = response_record['responses']
                
                if not responses or len(responses) == 0:
                    logger.warning(f"   ⚠️  No responses to score")
                    failed_scores += 1
                    continue
                
                # Calculate Semantic Entropy across τ grid WITH DIAGNOSTICS (following H2 pattern)
                logger.info(f"   📊 Computing Semantic Entropy across τ grid: {se_config['tau_grid']}")
                se_results = {}
                se_diagnostics = {}
                
                for tau in se_config['tau_grid']:
                    entropy_result = se_calculator.calculate_entropy(
                        responses, 
                        distance_threshold=tau,
                        return_diagnostics=True  # ENABLE diagnostics for H5 analysis
                    )
                    
                    # Handle both dict (with diagnostics) and float (without) returns
                    if isinstance(entropy_result, dict):
                        se_score = entropy_result.get('semantic_entropy', 0.0)
                        diagnostics = entropy_result
                    else:
                        se_score = entropy_result
                        diagnostics = {'semantic_entropy': se_score}
                    
                    se_results[f'tau_{tau}'] = se_score
                    se_diagnostics[f'tau_{tau}'] = diagnostics
                    
                    # Log diagnostic info for tracking
                    clusters = diagnostics.get('num_clusters', 'N/A')
                    logger.info(f"      τ={tau}: SE={se_score:.6f}, clusters={clusters}")
                
                # Calculate baseline metrics (using calculate_metrics like H1/H2)
                logger.info(f"   📊 Computing baseline metrics...")
                baseline_results = baseline_calculator.calculate_metrics(responses)
                
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # Create score record (COMPREHENSIVE format for H1/H2/H5 comparison)
                score_record = {
                    # Core identification (matching H5 responses)
                    "prompt_id": prompt_id,
                    "prompt": response_record['prompt'],  # Paraphrased prompt
                    "original_prompt": response_record['original_prompt'],  # Original for comparison
                    "label": response_record['label'],
                    "source_split": response_record['source_split'],
                    
                    # Semantic Entropy results (τ grid) - FLAT structure for H1 compatibility
                    **{f'semantic_entropy_tau_{tau}': se_results[f'tau_{tau}'] for tau in se_config['tau_grid']},
                    
                    # Semantic Entropy diagnostics (for detailed analysis)
                    "semantic_entropy_diagnostics": se_diagnostics,
                    
                    # Baseline metric results - FLAT structure for H1 compatibility
                    "avg_pairwise_bertscore": baseline_results.get('avg_pairwise_bertscore', 0.0),
                    "embedding_variance": baseline_results.get('embedding_variance', 0.0),
                    "levenshtein_variance": baseline_results.get('levenshtein_variance', 0.0),
                    
                    # Aggregated results for easier access
                    "semantic_entropy": se_results,  # Keep dict format for H2 compatibility
                    "baseline_metrics": baseline_results,  # Keep dict format for H2 compatibility
                    
                    # Scoring metadata
                    "scoring_metadata": {
                        "model": model_short,
                        "n_responses": len(responses),
                        "processing_time_seconds": processing_time,
                        "se_config": {
                            "tau_grid": se_config['tau_grid'],
                            "embedding_model": se_config['embedding_model']
                        },
                        "baseline_config": {
                            "methods": list(baseline_config.keys()),
                            "embedding_model": se_config['embedding_model']
                        }
                    },
                    
                    # H5 experiment metadata
                    "experiment": "h5_paraphrase_robustness",
                    "generation_metadata": response_record.get('generation_metadata', {}),
                    "paraphrase_metadata": response_record.get('paraphrase_metadata', {})
                }
                
                # Write immediately (incremental saving)
                outf.write(json.dumps(score_record) + '\n')
                outf.flush()  # Ensure immediate write
                
                successful_scores += 1
                
                logger.info(f"   ✅ Scored successfully")
                se_score_summary = [f'τ{tau}={se_results[f"tau_{tau}"]:.3f}' for tau in se_config['tau_grid']]
                logger.info(f"      SE scores: {se_score_summary}")
                logger.info(f"      Baseline metrics:")
                logger.info(f"        - BERTScore: {baseline_results.get('avg_pairwise_bertscore', 0.0):.3f}")
                logger.info(f"        - Embedding variance: {baseline_results.get('embedding_variance', 0.0):.6f}")
                logger.info(f"        - Levenshtein variance: {baseline_results.get('levenshtein_variance', 0.0):.3f}")
                
                # Progress update
                remaining = len(samples_to_process) - (successful_scores + failed_scores)
                avg_time = total_processing_time / (successful_scores + failed_scores) if (successful_scores + failed_scores) > 0 else 0
                eta_minutes = (remaining * avg_time) / 60 if avg_time > 0 else 0
                
                logger.info(f"📊 Progress: {successful_scores + failed_scores}/{len(samples_to_process)} processed")
                logger.info(f"   Successful: {successful_scores}, Failed: {failed_scores}")
                logger.info(f"   Avg time: {avg_time:.1f}s, ETA: {eta_minutes:.1f}min")
                
            except Exception as e:
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                failed_scores += 1
                logger.error(f"   ❌ Failed to score: {e}")
                continue
    
    # Commit to persistent storage
    volume.commit()
    
    # Final statistics
    logger.info("\n" + "="*100)
    logger.info("H5 SCORING COMPLETE")
    logger.info("="*100)
    logger.info(f"🎯 Model: {model_short}")
    logger.info(f"📊 Dataset: H5 paraphrased responses ({len(responses_data)} total)")
    logger.info(f"✅ Successful scores: {successful_scores}")
    logger.info(f"❌ Failed scores: {failed_scores}")
    logger.info(f"📈 Success rate: {successful_scores/(successful_scores + failed_scores)*100:.1f}%")
    logger.info(f"⏱️  Total processing time: {total_processing_time/60:.1f} minutes")
    logger.info(f"⏱️  Average per sample: {total_processing_time/successful_scores:.1f}s" if successful_scores > 0 else "N/A")
    logger.info(f"💾 Output file: {output_file}")
    
    return {
        "model": model_short,
        "total_samples": len(responses_data),
        "successful_scores": successful_scores,
        "failed_scores": failed_scores,
        "success_rate": successful_scores/(successful_scores + failed_scores) if (successful_scores + failed_scores) > 0 else 0,
        "total_time_minutes": total_processing_time/60,
        "output_file": str(output_file)
    }

# H5 Model mappings (matching response generation)
MODELS = {
    "qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
    "llama-4-scout-17b-16e-instruct": "llama-4-scout-17b-16e-instruct"
}

@app.local_entrypoint()
def main():
    """Run H5 scoring following H5 response generation pattern with environment variables."""
    import os
    
    # Check for test mode and model selection via environment variables (like H5 response generation)
    test_mode = os.environ.get('H5_SCORING_TEST_MODE', 'false').lower() == 'true'
    model_env = os.environ.get('H5_SCORING_MODEL', 'all').lower()
    
    print("="*100)
    print("H5 SCORING - PARAPHRASE ROBUSTNESS ANALYSIS")
    print("="*100)
    print(f"Input: H5 model responses to paraphrased prompts")
    print(f"Output: SE scores + baseline metrics for robustness comparison")
    print(f"Methods: SE (τ grid), BERTScore, Embedding variance, Levenshtein variance")
    print(f"Models: {list(MODELS.keys())}")
    print(f"🧪 Test mode: {'ENABLED (10 samples)' if test_mode else 'DISABLED (full dataset)'}")
    print(f"🤖 Model filter: {model_env}")
    print("="*100)
    
    # Determine which models to run
    if model_env == "all":
        models_to_run = list(MODELS.keys())
    elif model_env in MODELS:
        models_to_run = [model_env]
    else:
        print(f"❌ Unknown model '{model_env}'. Available: {list(MODELS.keys())} or 'all'")
        return
    
    results = {}
    
    for model_key in models_to_run:
        model_short = MODELS[model_key]
        print(f"\n🚀 Running {model_key}")
        
        try:
            result = score_h5_responses.remote(model_short, test_mode=test_mode)
            results[model_key] = result
            print(f"✅ {model_key} completed successfully")
            print(f"   Success rate: {result['success_rate']*100:.1f}%")
            print(f"   Output: {result['output_file']}")
        except Exception as e:
            print(f"❌ {model_key} failed: {e}")
            results[model_key] = {"error": str(e)}
    
    print("\n" + "="*100)
    print("H5 SCORING SUMMARY")
    print("="*100)
    for model_key, result in results.items():
        if "error" in result:
            print(f"❌ {model_key}: FAILED - {result['error']}")
        else:
            print(f"✅ {model_key}: SUCCESS - {result['successful_scores']}/{result['total_samples']} samples")
    
    if test_mode:
        print(f"\n🧪 TEST MODE COMPLETE")
        print(f"   Environment variables used:")
        print(f"   - H5_SCORING_TEST_MODE=true (for test mode)")
        print(f"   - H5_SCORING_MODEL={model_env} (model selection)")
        print(f"   To run full dataset: H5_SCORING_TEST_MODE=false")
    else:
        print(f"\n✅ FULL DATASET COMPLETE")
        
    print("\n🎯 Next steps:")
    print("1. Run H5 evaluation to compare with H1 baseline")
    print("2. Analyze SE robustness degradation on paraphrased prompts")
    print("3. Test H5 acceptance criterion (SE degrades >15pp more than baselines)")

if __name__ == "__main__":
    main()