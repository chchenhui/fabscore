
#!/usr/bin/env python3
"""
H2 Evaluation - Evaluate H2 twins scoring results on Modal.
Updated for new methodology: evaluate τ grid results vs baselines, no frozen parameters.
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
    "numpy", 
    "scikit-learn",
    "pandas",
    "pyyaml",
    "sentence-transformers",
    "torch",
    "bert-score",
    "python-Levenshtein"
]).add_local_python_source("src").add_local_dir("configs", "/configs").add_local_dir("data", "/data")

# Use same persistent storage volume
volume = modal.Volume.from_name("alignment-research-storage", create_if_missing=True)

app = modal.App("h2-evaluation")

@app.function(
    image=image,
    gpu="A100-40GB",  # GPU for consistency with scoring scripts
    timeout=7200,  # 2 hours (consistent with H2 scoring)
    volumes={"/research_storage": volume}
)
def evaluate_h2_results(model_short: str):
    """Evaluate H2 scoring results and generate comprehensive analysis."""
    from src.core.evaluation import calculate_auroc, calculate_fnr_at_fpr
    import json
    import logging
    from pathlib import Path
    import numpy as np
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 100)
    logger.info(f"H2 EVALUATION - {model_short}")
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
    scores_path = Path(f'/research_storage/outputs/h2/scoring/{model_short}_h2_scores.jsonl')
    output_dir = Path('/research_storage/outputs/h2/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{model_short}_h2_results.json"
    report_path = Path('/research_storage/reports') / f"h2_{model_short}_evaluation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Scores input: {scores_path}")
    logger.info(f"📁 Results output: {results_path}")
    logger.info(f"📁 Report output: {report_path}")
    
    # Check if scores file exists
    if not scores_path.exists():
        raise FileNotFoundError(f"H2 scores file not found at {scores_path}. "
                               "Run run_h2_scoring.py first.")
    
    # Load scores
    scores_data = []
    try:
        with open(scores_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    scores_data.append(data)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON error on line {line_num}: {e}")
                    
        logger.info(f"✅ Loaded {len(scores_data)} scored samples")
    except Exception as e:
        logger.error(f"❌ Failed to load scores: {e}")
        raise
    
    # Analyze dataset composition
    harmful_samples = [s for s in scores_data if s['label'] == 1]
    benign_samples = [s for s in scores_data if s['label'] == 0]
    
    logger.info(f"📊 Dataset composition:")
    logger.info(f"   Harmful samples: {len(harmful_samples)}")
    logger.info(f"   Benign samples: {len(benign_samples)}")
    logger.info(f"   Total samples: {len(scores_data)}")
    
    if len(harmful_samples) == 0 or len(benign_samples) == 0:
        raise ValueError("Need both harmful and benign samples for evaluation")
    
    # Setup evaluation parameters
    target_fpr = config.get('evaluation', {}).get('target_fpr', 0.05)
    tau_grid = [0.1, 0.2, 0.3, 0.4]  # Standard grid from H1
    
    logger.info(f"⚙️ Evaluation parameters:")
    logger.info(f"   Target FPR: {target_fpr}")
    logger.info(f"   τ grid: {tau_grid}")
    
    # Evaluation results storage
    evaluation_results = {
        'model': model_short,
        'dataset_composition': {
            'harmful_samples': len(harmful_samples),
            'benign_samples': len(benign_samples),
            'total_samples': len(scores_data)
        },
        'target_fpr': target_fpr,
        'semantic_entropy_results': {},
        'baseline_results': {},
        'comparison_analysis': {}
    }
    
    logger.info("🚀 Starting evaluation process...")
    
    # Extract labels and scores for evaluation
    labels = np.array([s['label'] for s in scores_data])
    
    # Evaluate Semantic Entropy for all τ values
    logger.info("\n🧠 Evaluating Semantic Entropy across τ grid...")
    se_results = {}
    
    for tau in tau_grid:
        logger.info(f"\n   Evaluating SE with τ={tau}...")
        
        # Extract SE scores for this τ
        se_key = f'tau_{tau}'
        if se_key not in scores_data[0]['semantic_entropy']:
            logger.warning(f"   ⚠️ τ={tau} not found in scores, skipping")
            continue
            
        se_scores = np.array([s['semantic_entropy'][se_key] for s in scores_data])
        
        # Calculate metrics
        try:
            auroc = calculate_auroc(labels, se_scores)
            fnr, threshold, actual_fpr, tpr_used = calculate_fnr_at_fpr(labels, se_scores, target_fpr=target_fpr)
            
            se_results[se_key] = {
                'tau': tau,
                'auroc': float(auroc),
                'fnr_at_target_fpr': float(fnr),
                'actual_fpr': float(actual_fpr),
                'tpr_used': float(tpr_used),
                'threshold': float(threshold),
                'score_stats': {
                    'mean': float(np.mean(se_scores)),
                    'std': float(np.std(se_scores)),
                    'min': float(np.min(se_scores)),
                    'max': float(np.max(se_scores)),
                    'harmful_mean': float(np.mean([s['semantic_entropy'][se_key] for s in harmful_samples])),
                    'benign_mean': float(np.mean([s['semantic_entropy'][se_key] for s in benign_samples])),
                    'separation': float(abs(np.mean([s['semantic_entropy'][se_key] for s in harmful_samples]) - np.mean([s['semantic_entropy'][se_key] for s in benign_samples])))
                }
            }
            
            logger.info(f"      AUROC: {auroc:.4f}")
            logger.info(f"      FNR@{target_fpr}FPR: {fnr:.4f} (actual FPR: {actual_fpr:.4f}, TPR: {tpr_used:.4f})")
            logger.info(f"      Threshold: {threshold:.6f}")
            logger.info(f"      Score separation (H-B): {abs(np.mean([s['semantic_entropy'][se_key] for s in harmful_samples]) - np.mean([s['semantic_entropy'][se_key] for s in benign_samples])):.6f}")
            
        except Exception as e:
            logger.error(f"      ❌ Evaluation failed for τ={tau}: {e}")
            se_results[se_key] = {'error': str(e)}
    
    evaluation_results['semantic_entropy_results'] = se_results
    
    # Evaluate baseline metrics
    logger.info("\n📏 Evaluating baseline metrics...")
    
    baseline_metrics = [
        ('avg_pairwise_bertscore', 'Average Pairwise BERTScore'),
        ('embedding_variance', 'Embedding Variance'),
        ('levenshtein_variance', 'Levenshtein Variance')
    ]
    
    baseline_results = {}
    
    for metric_key, metric_name in baseline_metrics:
        logger.info(f"\n   Evaluating {metric_name}...")
        
        try:
            metric_scores = np.array([s[metric_key] for s in scores_data])
            
            auroc = calculate_auroc(labels, metric_scores)
            fnr, threshold, actual_fpr, tpr_used = calculate_fnr_at_fpr(labels, metric_scores, target_fpr=target_fpr)
            
            baseline_results[metric_key] = {
                'name': metric_name,
                'auroc': float(auroc),
                'fnr_at_target_fpr': float(fnr),
                'actual_fpr': float(actual_fpr),
                'tpr_used': float(tpr_used),
                'threshold': float(threshold),
                'score_stats': {
                    'mean': float(np.mean(metric_scores)),
                    'std': float(np.std(metric_scores)),
                    'min': float(np.min(metric_scores)),
                    'max': float(np.max(metric_scores)),
                    'harmful_mean': float(np.mean([s[metric_key] for s in harmful_samples])),
                    'benign_mean': float(np.mean([s[metric_key] for s in benign_samples])),
                    'separation': float(abs(np.mean([s[metric_key] for s in harmful_samples]) - np.mean([s[metric_key] for s in benign_samples])))
                }
            }
            
            logger.info(f"      AUROC: {auroc:.4f}")
            logger.info(f"      FNR@{target_fpr}FPR: {fnr:.4f} (actual FPR: {actual_fpr:.4f}, TPR: {tpr_used:.4f})")
            logger.info(f"      Threshold: {threshold:.6f}")
            logger.info(f"      Score separation (H-B): {abs(np.mean([s[metric_key] for s in harmful_samples]) - np.mean([s[metric_key] for s in benign_samples])):.6f}")
            
        except Exception as e:
            logger.error(f"      ❌ Evaluation failed for {metric_name}: {e}")
            baseline_results[metric_key] = {'error': str(e)}
    
    evaluation_results['baseline_results'] = baseline_results
    
    # Comparative analysis
    logger.info("\n🔍 Performing comparative analysis...")
    
    # Find best SE performance
    valid_se_results = {k: v for k, v in se_results.items() if 'error' not in v}
    if valid_se_results:
        best_se_tau = min(valid_se_results.keys(), key=lambda x: valid_se_results[x]['fnr_at_target_fpr'])
        best_se_result = valid_se_results[best_se_tau]
        logger.info(f"   Best SE performance: {best_se_tau} (FNR: {best_se_result['fnr_at_target_fpr']:.4f})")
    else:
        best_se_result = None
        logger.warning("   ⚠️ No valid SE results for comparison")
    
    # Find best baseline performance
    valid_baseline_results = {k: v for k, v in baseline_results.items() if 'error' not in v}
    if valid_baseline_results:
        best_baseline_key = min(valid_baseline_results.keys(), key=lambda x: valid_baseline_results[x]['fnr_at_target_fpr'])
        best_baseline_result = valid_baseline_results[best_baseline_key]
        logger.info(f"   Best baseline performance: {best_baseline_result['name']} (FNR: {best_baseline_result['fnr_at_target_fpr']:.4f})")
    else:
        best_baseline_result = None
        logger.warning("   ⚠️ No valid baseline results for comparison")
    
    # H2 hypothesis test: SE should underperform baselines
    comparison_analysis = {
        'best_se': best_se_result,
        'best_baseline': best_baseline_result,
        'h2_hypothesis_supported': False,
        'performance_gap': None
    }
    
    if best_se_result and best_baseline_result:
        se_fnr = best_se_result['fnr_at_target_fpr']
        baseline_fnr = best_baseline_result['fnr_at_target_fpr']
        performance_gap = se_fnr - baseline_fnr
        
        # H2 success: SE FNR > baseline FNR (SE performs worse)
        hypothesis_supported = se_fnr > baseline_fnr
        
        comparison_analysis.update({
            'h2_hypothesis_supported': hypothesis_supported,
            'performance_gap': float(performance_gap),
            'se_fnr': float(se_fnr),
            'baseline_fnr': float(baseline_fnr),
            'interpretation': 'SE underperforms baseline' if hypothesis_supported else 'SE outperforms baseline'
        })
        
        logger.info(f"   H2 hypothesis test:")
        logger.info(f"      SE FNR: {se_fnr:.4f}")
        logger.info(f"      Baseline FNR: {baseline_fnr:.4f}")
        logger.info(f"      Performance gap: {performance_gap:.4f}")
        logger.info(f"      H2 supported: {hypothesis_supported} ({comparison_analysis['interpretation']})")
    
    evaluation_results['comparison_analysis'] = comparison_analysis
    
    # Generate detailed performance table like H1
    logger.info("\n" + "="*80)
    logger.info("DETAILED PERFORMANCE COMPARISON")
    logger.info("="*80)
    logger.info("\n📊 H2 RESULTS TABLE:")
    logger.info("Method                    | AUROC  | FNR@5%FPR | FPR_used | TPR_used | Params")
    logger.info("-" * 80)
    
    # Semantic Entropy rows (all tau values)
    for tau_key, result in valid_se_results.items():
        if 'error' not in result:
            marker = " ⭐" if tau_key == best_se_tau else ""
            logger.info(f"{'SE τ=' + str(result['tau']) + marker:<25} | {result['auroc']:.4f} | {result['fnr_at_target_fpr']:.4f}    | {result['actual_fpr']:.4f}   | {result['tpr_used']:.4f}   | τ={result['tau']}")
    
    # Baseline rows
    for method_key, result in valid_baseline_results.items():
        if 'error' not in result:
            marker = " ⭐" if method_key == best_baseline_key else ""
            method_display = result['name'] + marker
            logger.info(f"{method_display:<25} | {result['auroc']:.4f} | {result['fnr_at_target_fpr']:.4f}    | {result['actual_fpr']:.4f}   | {result['tpr_used']:.4f}   | thresh={result['threshold']:.4f}")
    
    logger.info("-" * 80)
    
    # Performance ranking
    logger.info(f"\n🏆 PERFORMANCE RANKING (by AUROC):")
    all_methods = []
    for tau_key, result in valid_se_results.items():
        all_methods.append((f"SE τ={result['tau']}", result['auroc']))
    for method_key, result in valid_baseline_results.items():
        all_methods.append((result['name'], result['auroc']))
    
    all_methods.sort(key=lambda x: x[1], reverse=True)
    
    for i, (method, auroc) in enumerate(all_methods):
        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣"][i] if i < 7 else f"{i+1}️⃣"
        logger.info(f"  {rank_emoji} {method}: {auroc:.4f}")
    
    logger.info("="*80)
    
    # Final results summary
    logger.info("\n" + "=" * 100)
    logger.info("H2 EVALUATION COMPLETE")
    logger.info("=" * 100)
    
    logger.info(f"📊 EVALUATION SUMMARY:")
    logger.info(f"   Model: {model_short}")
    logger.info(f"   Samples evaluated: {len(scores_data)}")
    logger.info(f"   SE configurations tested: {len(valid_se_results)}")
    logger.info(f"   Baseline methods tested: {len(valid_baseline_results)}")
    
    if comparison_analysis['h2_hypothesis_supported']:
        logger.info(f"   🎯 H2 HYPOTHESIS SUPPORTED: SE underperforms best baseline")
        logger.info(f"      Performance gap: {comparison_analysis['performance_gap']:.4f} FNR difference")
    else:
        logger.info(f"   ❌ H2 HYPOTHESIS NOT SUPPORTED: SE outperforms best baseline")
        if comparison_analysis['performance_gap'] is not None:
            logger.info(f"      Performance gap: {comparison_analysis['performance_gap']:.4f} FNR difference")
    
    # Save results
    try:
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Results saved to {results_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        raise
    
    # Generate comprehensive evaluation report
    try:
        with open(report_path, 'w') as f:
            f.write(f"# H2 Evaluation Report - {model_short}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Model:** {model_short}\n")
            f.write(f"- **Dataset:** HarmBench Twins ({len(harmful_samples)} harmful + {len(benign_samples)} benign)\n")
            f.write(f"- **Target FPR:** {target_fpr}\n")
            f.write(f"- **Evaluation Date:** {evaluation_results['model']}\n\n")
            
            f.write("## Hypothesis Testing\n\n")
            f.write("**H2 Claim:** SE underperforms Avg-BERTScore on HarmBench-Benign-Matched at low FPR\n\n")
            
            if comparison_analysis['h2_hypothesis_supported']:
                f.write("### ✅ H2 HYPOTHESIS SUPPORTED\n\n")
                f.write(f"SE underperforms the best baseline by {comparison_analysis['performance_gap']:.4f} FNR points.\n\n")
                f.write(f"- **Best SE FNR@5%FPR:** {comparison_analysis['se_fnr']:.4f}\n")
                f.write(f"- **Best Baseline FNR@5%FPR:** {comparison_analysis['baseline_fnr']:.4f} ({best_baseline_result['name']})\n")
            else:
                f.write("### ❌ H2 HYPOTHESIS NOT SUPPORTED\n\n")
                f.write(f"SE outperforms the best baseline by {abs(comparison_analysis['performance_gap']):.4f} FNR points.\n\n")
                f.write(f"- **Best SE FNR@5%FPR:** {comparison_analysis['se_fnr']:.4f}\n")
                f.write(f"- **Best Baseline FNR@5%FPR:** {comparison_analysis['baseline_fnr']:.4f} ({best_baseline_result['name']})\n")
            
            f.write("\n## Detailed Results\n\n")
            f.write("### Semantic Entropy Results\n\n")
            f.write("| τ | AUROC | FNR@5%FPR | Actual FPR | TPR Used | Threshold |\n")
            f.write("|---|-------|-----------|------------|----------|----------|\n")
            
            for tau_key, result in se_results.items():
                if 'error' not in result:
                    f.write(f"| {result['tau']} | {result['auroc']:.4f} | {result['fnr_at_target_fpr']:.4f} | {result['actual_fpr']:.4f} | {result['tpr_used']:.4f} | {result['threshold']:.6f} |\n")
            
            f.write("\n### Baseline Results\n\n")
            f.write("| Method | AUROC | FNR@5%FPR | Actual FPR | TPR Used | Threshold |\n")
            f.write("|--------|-------|-----------|------------|----------|----------|\n")
            
            for metric_key, result in baseline_results.items():
                if 'error' not in result:
                    f.write(f"| {result['name']} | {result['auroc']:.4f} | {result['fnr_at_target_fpr']:.4f} | {result['actual_fpr']:.4f} | {result['tpr_used']:.4f} | {result['threshold']:.6f} |\n")
            
            f.write("\n## Score Distribution Analysis\n\n")
            
            if valid_se_results:
                best_tau = min(valid_se_results.keys(), key=lambda x: valid_se_results[x]['fnr_at_target_fpr'])
                best_se = valid_se_results[best_tau]
                f.write(f"### Best SE Configuration (τ={best_se['tau']})\n")
                f.write(f"- **Harmful mean score:** {best_se['score_stats']['harmful_mean']:.6f}\n")
                f.write(f"- **Benign mean score:** {best_se['score_stats']['benign_mean']:.6f}\n")
                f.write(f"- **Score separation:** {abs(best_se['score_stats']['harmful_mean'] - best_se['score_stats']['benign_mean']):.6f}\n\n")
            
            if valid_baseline_results:
                best_baseline_key = min(valid_baseline_results.keys(), key=lambda x: valid_baseline_results[x]['fnr_at_target_fpr'])
                best_baseline = valid_baseline_results[best_baseline_key]
                f.write(f"### Best Baseline ({best_baseline['name']})\n")
                f.write(f"- **Harmful mean score:** {best_baseline['score_stats']['harmful_mean']:.6f}\n")
                f.write(f"- **Benign mean score:** {best_baseline['score_stats']['benign_mean']:.6f}\n")
                f.write(f"- **Score separation:** {abs(best_baseline['score_stats']['harmful_mean'] - best_baseline['score_stats']['benign_mean']):.6f}\n\n")
            
            f.write("## Methodology\n\n")
            f.write("- **SE Implementation:** Agglomerative clustering with cosine distance\n")
            f.write("- **Embedding Model:** Alibaba-NLP/gte-large-en-v1.5\n")
            f.write("- **Response Generation:** N=5, temp=0.7, top_p=0.95, max_tokens=1024\n")
            f.write("- **Evaluation:** ROC analysis with threshold selection at target FPR\n")
            
        logger.info(f"✅ Evaluation report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save evaluation report: {e}")
    
    # Commit volume changes
    try:
        volume.commit()
        logger.info("✅ Volume changes committed")
    except Exception as e:
        logger.error(f"❌ Volume commit failed: {e}")
    
    return {
        'success': True,
        'model_short': model_short,
        'samples_evaluated': len(scores_data),
        'h2_hypothesis_supported': comparison_analysis['h2_hypothesis_supported'],
        'performance_gap': comparison_analysis.get('performance_gap'),
        'results_path': str(results_path),
        'report_path': str(report_path)
    }

@app.local_entrypoint()
def main(model_short: str = "qwen-2.5-7b-instruct"):
    """Main entrypoint for H2 evaluation."""
    
    print("=" * 100)
    print("H2 EVALUATION ON MODAL")
    print("=" * 100)
    print(f"Model: {model_short}")
    print("This will:")
    print("1. Load H2 scoring results for the specified model")
    print("2. Evaluate SE performance across τ grid")
    print("3. Compare SE vs baseline metrics")
    print("4. Test H2 hypothesis: SE underperforms baselines")
    print("5. Generate comprehensive evaluation report")
    print("=" * 100)
    
    try:
        result = evaluate_h2_results.remote(model_short)
        
        print("\n" + "=" * 100)
        print("✅ H2 EVALUATION COMPLETE!")
        print("=" * 100)
        print(f"Model: {result['model_short']}")
        print(f"Samples evaluated: {result['samples_evaluated']}")
        print(f"H2 hypothesis supported: {result['h2_hypothesis_supported']}")
        if result['performance_gap'] is not None:
            print(f"Performance gap: {result['performance_gap']:.4f} FNR")
        print(f"Results: {result['results_path']}")
        print(f"Report: {result['report_path']}")
        print("=" * 100)
        
        return result
        
    except Exception as e:
        print(f"\n❌ H2 EVALUATION FAILED: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen-2.5-7b-instruct"
    main(model)
