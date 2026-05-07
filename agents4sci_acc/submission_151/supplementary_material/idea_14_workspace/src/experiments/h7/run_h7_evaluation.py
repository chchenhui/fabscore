"""
H7 Evaluation Script - Evaluate Qwen2.5-72B-Instruct performance
Adapted from H1 evaluation to check if SE fails on SOTA models
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import yaml
import modal
import os
import numpy as np
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "openai", "requests", "pyyaml", "numpy", "scikit-learn", 
    "sentence-transformers", "torch", "bert-score", "python-Levenshtein", "pandas", "tqdm"
]).add_local_python_source("src").add_local_dir("configs", "/configs")

# Persistent storage volume for research outputs
volume = modal.Volume.from_name("alignment-research-storage", create_if_missing=True)

app = modal.App("h7-evaluation")

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,  # 2 hours (consistent with H2)
    volumes={"/research_storage": volume}
)
def evaluate_h7_results(model_short: str, test_mode=False):
    """Evaluate H7 results to check if SE fails on SOTA model
    
    Args:
        model_short: Short model name (e.g., 'qwen-2.5-72b-instruct')
        test_mode: If True, process test file instead of full dataset
    """
    from src.core.evaluation import calculate_auroc, calculate_fnr_at_fpr
    import yaml
    import json
    import logging
    import pandas as pd
    import os
    import numpy as np
    from datetime import datetime
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration with error handling
    try:
        with open('/configs/project_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Loaded project configuration")
    except Exception as e:
        logger.warning(f"⚠️ Could not load config: {e}, using defaults")
        config = {}
    
    # Get H7 configuration
    h7_config = config['hypotheses']['h7']
    tau_grid = h7_config['tau_grid']
    target_fpr = h7_config['target_fpr']
    acceptance_criteria = h7_config['acceptance_criteria']
    
    # Map short name to full model name
    model_mapping = {
        'qwen-2.5-72b-instruct': 'Qwen/Qwen2.5-72B-Instruct',
        'llama-3.3-70b-instruct': 'meta-llama/Llama-3.3-70B-Instruct'
    }
    full_model_name = model_mapping.get(model_short, model_short)
    
    logging.info("="*80)
    logging.info(f"H7 EVALUATION - SOTA MODEL CHECK ({full_model_name})")
    logging.info("="*80)
    logging.info(f"📊 Configuration:")
    logging.info(f"   - Tau grid: {tau_grid}")
    logging.info(f"   - Target FPR: {target_fpr}")
    logging.info(f"   - Success criteria:")
    logging.info(f"     - SE AUROC must be below best baseline")
    logging.info(f"     - SE FNR must exceed {acceptance_criteria['fnr_threshold']} for at least one tau")
    logging.info("="*80)
    
    # Setup input/output paths matching H7 scoring convention
    if test_mode:
        scores_file = f"/research_storage/outputs/h7/{model_short}_h7_TEST_scores.jsonl"
        output_dir = Path('/research_storage/outputs/h7/evaluation')
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{model_short}_h7_TEST_results.json"
        report_file = Path('/research_storage/reports') / f"h7_{model_short}_TEST_evaluation_report.md"
    else:
        scores_file = f"/research_storage/outputs/h7/{model_short}_h7_scores.jsonl"
        output_dir = Path('/research_storage/outputs/h7/evaluation')
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{model_short}_h7_results.json"
        report_file = Path('/research_storage/reports') / f"h7_{model_short}_evaluation_report.md"
    
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Scores input: {scores_file}")
    logger.info(f"📁 Results output: {results_file}")
    logger.info(f"📁 Report output: {report_file}")
    
    # Load scored data
    scored_data = []
    with open(scores_file, 'r') as f:
        for line in f:
            scored_data.append(json.loads(line))
    
    logging.info(f"📁 Loaded {len(scored_data)} scored samples")
    
    # Prepare data for evaluation
    labels = [item['label'] for item in scored_data]
    
    # Evaluate each method
    results = {
        'metadata': {
            'model': full_model_name,
            'model_short': model_short,
            'dataset_size': len(scored_data),
            'harmful_count': sum(labels),
            'benign_count': len(labels) - sum(labels),
            'timestamp': datetime.now().isoformat()
        },
        'semantic_entropy': {},
        'baselines': {}
    }
    
    # Evaluate Semantic Entropy for each tau (adapt to H2 structure)
    for tau in tau_grid:
        tau_key = f'tau_{tau}'
        scores = []
        for item in scored_data:
            # Handle both old and new score structures
            if 'semantic_entropy' in item and tau_key in item['semantic_entropy']:
                score = item['semantic_entropy'][tau_key]
            elif 'scores' in item:
                score = item['scores'].get(f'se_tau_{tau}', 0.0)
            else:
                score = 0.0
            scores.append(score if score is not None else 0.0)
        
        if len(scores) == len(labels):
            auroc = calculate_auroc(labels, scores)
            fnr, threshold, actual_fpr, tpr_used = calculate_fnr_at_fpr(labels, scores, target_fpr)
            
            results['semantic_entropy'][f'tau_{tau}'] = {
                'auroc': auroc,
                'fnr_at_5fpr': fnr,
                'actual_fpr': actual_fpr,
                'tpr_used': tpr_used,
                'threshold': threshold,
                'mean_harmful': np.mean([s for i, s in enumerate(scores) if labels[i] == 1]),
                'mean_benign': np.mean([s for i, s in enumerate(scores) if labels[i] == 0])
            }
            
            logger.info(f"SE (tau={tau}): AUROC={auroc:.4f}, FNR@5%FPR={fnr:.4f} (actual FPR: {actual_fpr:.4f})")
    
    # Evaluate baseline methods (handle both structures)
    baseline_methods = ['avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance']
    
    for method in baseline_methods:
        scores = []
        for item in scored_data:
            # Handle both old and new score structures
            if method in item:
                score = item[method]
            elif 'scores' in item:
                score = item['scores'].get(method, 0.0)
            else:
                score = 0.0
            scores.append(score if score is not None else 0.0)
        
        if len(scores) == len(labels):
            auroc = calculate_auroc(labels, scores)
            fnr, threshold, actual_fpr, tpr_used = calculate_fnr_at_fpr(labels, scores, target_fpr)
            
            results['baselines'][method] = {
                'auroc': auroc,
                'fnr_at_5fpr': fnr,
                'actual_fpr': actual_fpr,
                'tpr_used': tpr_used,
                'threshold': threshold,
                'mean_harmful': np.mean([s for i, s in enumerate(scores) if labels[i] == 1]),
                'mean_benign': np.mean([s for i, s in enumerate(scores) if labels[i] == 0])
            }
            
            logger.info(f"{method}: AUROC={auroc:.4f}, FNR@5%FPR={fnr:.4f} (actual FPR: {actual_fpr:.4f})")
    
    # Check H7 success criteria
    best_baseline_auroc = max(results['baselines'][m]['auroc'] for m in results['baselines'])
    best_se_auroc = max(results['semantic_entropy'][f'tau_{tau}']['auroc'] for tau in tau_grid)
    max_se_fnr = max(results['semantic_entropy'][f'tau_{tau}']['fnr_at_5fpr'] for tau in tau_grid)
    
    h7_success = {
        'se_below_baseline': bool(best_se_auroc < best_baseline_auroc),
        'high_fnr_persists': bool(max_se_fnr > acceptance_criteria['fnr_threshold']),
        'overall_pass': bool(best_se_auroc < best_baseline_auroc and max_se_fnr > acceptance_criteria['fnr_threshold'])
    }
    
    results['h7_evaluation'] = {
        'best_baseline_auroc': best_baseline_auroc,
        'best_se_auroc': best_se_auroc,
        'max_se_fnr': max_se_fnr,
        'success_criteria_met': h7_success,
        'conclusion': 'PASS - SE fails on SOTA model' if h7_success['overall_pass'] else 'FAIL - SE may work on larger models'
    }
    
    # Save results JSON
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    report = generate_h7_report(results, h7_config)
    with open(report_file, 'w') as f:
        f.write(report)
    
    logging.info("="*80)
    logging.info("H7 EVALUATION RESULTS")
    logging.info("="*80)
    logging.info(f"🎯 Best SE AUROC: {best_se_auroc:.4f}")
    logging.info(f"🎯 Best Baseline AUROC: {best_baseline_auroc:.4f}")
    logging.info(f"🎯 Max SE FNR@5%FPR: {max_se_fnr:.4f}")
    logging.info(f"")
    logging.info(f"✅ SE below baseline: {'YES' if h7_success['se_below_baseline'] else 'NO'}")
    logging.info(f"✅ High FNR persists (>{acceptance_criteria['fnr_threshold']}): {'YES' if h7_success['high_fnr_persists'] else 'NO'}")
    logging.info(f"")
    logging.info(f"📊 H7 CONCLUSION: {results['h7_evaluation']['conclusion']}")
    logging.info("="*80)
    
    # Volume commit for persistence
    volume.commit()
    
    return results_file, report_file

def generate_h7_report(results, config):
    """Generate markdown report for H7 results"""
    report = []
    model_name = results['metadata']['model']
    report.append(f"# H7 Evaluation Report: SOTA Model Check ({model_name})")
    report.append("")
    report.append(f"**Generated:** {results['metadata']['timestamp']}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append(f"**Model:** {results['metadata']['model']}")
    report.append(f"**Dataset:** {results['metadata']['dataset_size']} samples "
                  f"({results['metadata']['harmful_count']} harmful, "
                  f"{results['metadata']['benign_count']} benign)")
    report.append("")
    report.append(f"**Conclusion:** {results['h7_evaluation']['conclusion']}")
    report.append("")
    
    # Success Criteria
    report.append("## Success Criteria")
    report.append("")
    report.append("H7 tests whether Semantic Entropy failures persist with larger, more capable models:")
    report.append("")
    report.append("1. **SE must underperform baselines:** SE AUROC < Best Baseline AUROC")
    report.append(f"   - Result: SE AUROC = {results['h7_evaluation']['best_se_auroc']:.4f}, "
                  f"Best Baseline = {results['h7_evaluation']['best_baseline_auroc']:.4f}")
    report.append(f"   - **{'✅ PASS' if results['h7_evaluation']['success_criteria_met']['se_below_baseline'] else '❌ FAIL'}**")
    report.append("")
    report.append(f"2. **High FNR must persist:** SE FNR@5%FPR > {config['acceptance_criteria']['fnr_threshold']}")
    report.append(f"   - Result: Max SE FNR = {results['h7_evaluation']['max_se_fnr']:.4f}")
    report.append(f"   - **{'✅ PASS' if results['h7_evaluation']['success_criteria_met']['high_fnr_persists'] else '❌ FAIL'}**")
    report.append("")
    
    # Detailed Results
    report.append("## Detailed Results")
    report.append("")
    
    # SE Results Table
    report.append("### Semantic Entropy Performance")
    report.append("")
    report.append("| Tau | AUROC | FNR@5%FPR | Mean (Harmful) | Mean (Benign) |")
    report.append("|-----|-------|-----------|----------------|---------------|")
    for tau in sorted([float(k.split('_')[1]) for k in results['semantic_entropy'].keys()]):
        tau_key = f'tau_{tau}'
        r = results['semantic_entropy'][tau_key]
        report.append(f"| {tau} | {r['auroc']:.4f} | {r['fnr_at_5fpr']:.4f} | "
                      f"{r['mean_harmful']:.4f} | {r['mean_benign']:.4f} |")
    report.append("")
    
    # Baseline Results Table
    report.append("### Baseline Methods Performance")
    report.append("")
    report.append("| Method | AUROC | FNR@5%FPR | Mean (Harmful) | Mean (Benign) |")
    report.append("|--------|-------|-----------|----------------|---------------|")
    for method, r in results['baselines'].items():
        report.append(f"| {method} | {r['auroc']:.4f} | {r['fnr_at_5fpr']:.4f} | "
                      f"{r['mean_harmful']:.4f} | {r['mean_benign']:.4f} |")
    report.append("")
    
    # Key Findings
    report.append("## Key Findings")
    report.append("")
    
    if results['h7_evaluation']['success_criteria_met']['overall_pass']:
        report.append("✅ **H7 PASSES:** Semantic Entropy continues to fail even on the SOTA Qwen2.5-72B model.")
        report.append("")
        report.append("This confirms that SE's failure is not due to model size or capability limitations, ")
        report.append("but rather represents a fundamental issue with using output diversity as a ")
        report.append("safety signal for well-aligned models.")
    else:
        report.append("❌ **H7 FAILS:** Semantic Entropy shows improved performance on the SOTA model.")
        report.append("")
        report.append("This suggests that SE's failures may be partially addressable through ")
        report.append("increased model scale and capability.")
    
    report.append("")
    report.append("## Implications")
    report.append("")
    report.append("The results from this SOTA model check have important implications for the ")
    report.append("viability of consistency-based jailbreak detection methods:")
    report.append("")
    
    if results['h7_evaluation']['success_criteria_met']['overall_pass']:
        report.append("1. **Consistency Confound persists:** Even state-of-the-art models produce ")
        report.append("   consistent, templated refusals that defeat SE-based detection.")
        report.append("")
        report.append("2. **Scale is not a solution:** Simply using larger models does not resolve ")
        report.append("   the fundamental limitation of diversity-based detection.")
        report.append("")
        report.append("3. **Alternative approaches needed:** The field needs detection methods that ")
        report.append("   do not rely on output diversity as a primary signal.")
    else:
        report.append("1. **Scale may help:** Larger models might provide better signals for ")
        report.append("   consistency-based detection methods.")
        report.append("")
        report.append("2. **Further investigation needed:** Additional experiments with other ")
        report.append("   SOTA models could clarify the relationship between scale and SE performance.")
    
    return "\n".join(report)

@app.local_entrypoint()
def main(model: str = "qwen-2.5-72b-instruct", test: bool = False):
    """Entry point for H7 evaluation"""
    
    print("=" * 100)
    print("H7 EVALUATION ON MODAL")
    print("=" * 100)
    print(f"Model: {model}")
    print(f"Mode: {'TEST (10 samples)' if test else 'FULL (120 samples)'}")
    print("This will:")
    print("1. Load H7 scoring results for the specified SOTA model")
    print("2. Evaluate SE performance across τ grid")
    print("3. Compare SE vs baseline metrics")
    print("4. Test H7 hypothesis: SE fails even on SOTA models")
    print("5. Generate comprehensive evaluation report")
    print("=" * 100)
    
    try:
        result = evaluate_h7_results.remote(model_short=model, test_mode=test)
        
        print("\n" + "=" * 100)
        print("✅ H7 EVALUATION COMPLETE!")
        print("=" * 100)
        print(f"Model: {result['model_short']}")
        print(f"Samples evaluated: {result['samples_evaluated']}")
        print(f"H7 passes: {result['h7_passes']}")
        print(f"Conclusion: {result['h7_conclusion']}")
        print(f"Best SE AUROC: {result['best_se_auroc']:.4f}")
        print(f"Best Baseline AUROC: {result['best_baseline_auroc']:.4f}")
        print(f"Max SE FNR@5%FPR: {result['max_se_fnr']:.4f}")
        print(f"Results: {result['results_path']}")
        print(f"Report: {result['report_path']}")
        print("=" * 100)
        
        return result
        
    except Exception as e:
        print(f"\n❌ H7 EVALUATION FAILED: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    main()