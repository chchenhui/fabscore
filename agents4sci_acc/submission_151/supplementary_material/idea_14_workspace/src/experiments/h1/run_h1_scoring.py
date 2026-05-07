
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
    "sentence-transformers", "torch", "bert-score", "python-Levenshtein"
]).add_local_python_source("src").add_local_dir("configs", "/configs")

# Persistent storage volume for research outputs
volume = modal.Volume.from_name("alignment-research-storage", create_if_missing=True)

app = modal.App("h1-scoring")

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,  # 2 hours
    volumes={"/research_storage": volume}
)
def compute_h1_scores():
    """Compute H1 scores using consistent configuration"""
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
    
    # Get consistent parameters from config
    embedding_model = config['hypotheses']['h1']['embedding_model']
    tau_grid = config['tuning']['tau_grid']
    
    logging.info(f"📊 Using embedding model: {embedding_model}")
    logging.info(f"📊 Computing semantic entropy for all tau values: {tau_grid}")
    
    # Set up paths in persistent storage - match exact filenames from response generation
    input_file = "/research_storage/outputs/h1/llama4scout_120val_N5_temp0.7_top0.95_tokens1024_responses.jsonl"
    output_file = "/research_storage/outputs/h1/llama4scout_120val_N5_temp0.7_top0.95_tokens1024_scores.jsonl"
    
    se_calculator = SemanticEntropy(embedding_model)
    baseline_calculator = BaselineMetrics(embedding_model)
    
    # Load generated responses
    response_data = []
    with open(input_file, 'r') as f:
        for line in f:
            response_data.append(json.loads(line))
    
    logging.info(f"📋 Loaded {len(response_data)} response sets for scoring")
    
    # Verify label distribution (should match hyperparameter tuning)
    labels = [d['label'] for d in response_data]
    harmful_count = sum(1 for l in labels if l == 1)
    benign_count = sum(1 for l in labels if l == 0)
    logging.info(f"📊 Label distribution: {harmful_count} harmful, {benign_count} benign")
    logging.info(f"✅ Balance check: {'Balanced' if harmful_count == benign_count == 60 else 'WARNING: Imbalanced!'}")
    
    # Log sample data for debugging (per CLAUDE.md requirements)
    if response_data:
        sample = response_data[0]
        logging.info(f"📝 Sample data structure:")
        logging.info(f"  - prompt_id: {sample['prompt_id']}")
        logging.info(f"  - label: {sample['label']}")
        logging.info(f"  - num_responses: {len(sample['responses'])}")
        logging.info(f"  - response_lengths: {[len(r) for r in sample['responses'][:2]]}...")
    
    logging.info("="*80)
    logging.info("STARTING SCORING COMPUTATION")
    logging.info("="*80)
    
    # Track score statistics for logging
    all_se_scores = {f"tau_{tau}": [] for tau in tau_grid}
    all_baseline_scores = {"bertscore": [], "embedding_var": [], "levenshtein_var": []}
    
    with open(output_file, 'w') as f_out:
        for i, data in enumerate(tqdm(response_data, desc="Computing H1 Scores")):
            prompt_id = data['prompt_id']
            responses = data['responses']
            label = data['label']
            
            # Compute semantic entropy for all tau values
            semantic_entropy_scores = {}
            for tau in tau_grid:
                se_score = se_calculator.calculate_entropy(responses, distance_threshold=tau)
                semantic_entropy_scores[f"semantic_entropy_tau_{tau}"] = se_score
                all_se_scores[f"tau_{tau}"].append(se_score)
            
            baseline_scores = baseline_calculator.calculate_metrics(responses)
            
            # Track baseline scores for statistics
            all_baseline_scores["bertscore"].append(baseline_scores["avg_pairwise_bertscore"])
            all_baseline_scores["embedding_var"].append(baseline_scores["embedding_variance"]) 
            all_baseline_scores["levenshtein_var"].append(baseline_scores["levenshtein_variance"])
            
            # Log scores for first few samples (detailed debugging)
            if i < 5:  # Log first 5 samples in detail
                logging.info(f"\n📊 Sample {i+1} (ID: {prompt_id}, Label: {label}):")
                logging.info(f"  Response lengths: {[len(r) for r in responses]}")
                logging.info(f"  Semantic Entropy scores:")
                for tau in tau_grid:
                    score = semantic_entropy_scores[f"semantic_entropy_tau_{tau}"]
                    logging.info(f"    τ={tau}: {score:.6f}")
                logging.info(f"  Baseline scores:")
                logging.info(f"    BERTScore: {baseline_scores['avg_pairwise_bertscore']:.6f}")
                logging.info(f"    Embedding var: {baseline_scores['embedding_variance']:.6f}")
                logging.info(f"    Levenshtein var: {baseline_scores['levenshtein_variance']:.6f}")
            
            output_record = {
                "prompt_id": prompt_id,
                "label": label,
                **semantic_entropy_scores,
                **baseline_scores
            }
            f_out.write(json.dumps(output_record) + '\n')
    
    # Commit volume changes
    volume.commit()
    
    # Comprehensive score statistics logging
    logging.info("="*80)
    logging.info("SCORE STATISTICS SUMMARY")
    logging.info("="*80)
    
    # Semantic Entropy Statistics
    logging.info("📊 SEMANTIC ENTROPY SCORES:")
    for tau in tau_grid:
        scores = all_se_scores[f"tau_{tau}"]
        if len(scores) > 0:
            logging.info(f"  τ={tau}:")
            logging.info(f"    Mean: {np.mean(scores):.6f}, Std: {np.std(scores):.6f}")
            logging.info(f"    Min: {np.min(scores):.6f}, Max: {np.max(scores):.6f}")
            if len(scores) > 2:
                logging.info(f"    Range: [{np.percentile(scores, 25):.6f}, {np.percentile(scores, 75):.6f}] (25th-75th percentile)")
        else:
            logging.warning(f"  τ={tau}: No scores available")
    
    # Baseline Metrics Statistics  
    logging.info("\n📊 BASELINE METRIC SCORES:")
    baseline_names = {"bertscore": "BERTScore", "embedding_var": "Embedding Variance", "levenshtein_var": "Levenshtein Variance"}
    for key, name in baseline_names.items():
        scores = all_baseline_scores[key]
        if len(scores) > 0:
            logging.info(f"  {name}:")
            logging.info(f"    Mean: {np.mean(scores):.6f}, Std: {np.std(scores):.6f}")
            logging.info(f"    Min: {np.min(scores):.6f}, Max: {np.max(scores):.6f}")
            if len(scores) > 2:
                logging.info(f"    Range: [{np.percentile(scores, 25):.6f}, {np.percentile(scores, 75):.6f}] (25th-75th percentile)")
        else:
            logging.warning(f"  {name}: No scores available")
    
    # Score Distribution by Label
    logging.info("\n📊 SCORE DISTRIBUTION BY LABEL:")
    harmful_indices = [i for i, d in enumerate(response_data) if d['label'] == 1]
    benign_indices = [i for i, d in enumerate(response_data) if d['label'] == 0]
    
    logging.info(f"  Harmful samples (n={len(harmful_indices)}):")
    for tau in tau_grid:
        harmful_scores = [all_se_scores[f"tau_{tau}"][i] for i in harmful_indices]
        if len(harmful_scores) > 0:
            logging.info(f"    τ={tau}: Mean={np.mean(harmful_scores):.6f}, Std={np.std(harmful_scores):.6f}")
        else:
            logging.warning(f"    τ={tau}: No harmful scores available")
        
    logging.info(f"  Benign samples (n={len(benign_indices)}):")  
    for tau in tau_grid:
        benign_scores = [all_se_scores[f"tau_{tau}"][i] for i in benign_indices]
        if len(benign_scores) > 0:
            logging.info(f"    τ={tau}: Mean={np.mean(benign_scores):.6f}, Std={np.std(benign_scores):.6f}")
        else:
            logging.warning(f"    τ={tau}: No benign scores available")
    
    # Final summary
    logging.info("\n" + "="*80)
    logging.info("H1 SCORING COMPLETE")
    logging.info("="*80)
    logging.info(f"✅ Computed scores for {len(response_data)} response sets")
    logging.info(f"📁 Output file: {output_file}")
    logging.info(f"📊 Metrics computed:")
    logging.info(f"  - Semantic entropy for tau values: {tau_grid}")
    logging.info(f"  - Baseline metrics: BERTScore, embedding variance, Levenshtein variance")
    logging.info("="*80)
    
    return len(response_data)

@app.local_entrypoint()
def main():
    logging.info(f"🚀 Starting H1 scoring computation")
    num_scored = compute_h1_scores.remote()
    logging.info(f"✅ Computed scores for {num_scored} response sets")

# Entry point handled by Modal
