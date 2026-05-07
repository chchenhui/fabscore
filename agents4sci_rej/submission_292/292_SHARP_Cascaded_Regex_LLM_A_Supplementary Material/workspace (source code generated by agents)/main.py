#!/usr/bin/env python3
"""
Fast Phishing Email Detection Experiment
Focuses on methods that don't require external LLMs
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from improved_data_loader import ImprovedDataLoader
from baseline_methods import BaselineMethods
from evaluation import Evaluator
from visualizer import ResultsVisualizer

# Import new academic baseline methods
from phishintention_adapter import PhishIntentionAdapter
from cnn_bigru_detector import CNNBiGRUDetector
from feature_ensemble_detector import FeatureEnsembleDetector

# Import our novel hybrid detector
from hybrid_llm_regex_detector import HybridLLMRegexDetector

def main():
    """Main experiment pipeline - Fast Version"""

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PHISHING DETECTION - ACADEMIC METHODS COMPARISON")
    logger.info("=" * 80)

    # Step 1: Load and prepare datasets
    logger.info("\n[Step 1/4] Loading and preparing datasets...")
    data_loader = ImprovedDataLoader()
    train_data, val_data, test_data = data_loader.load_and_split_data()

    logger.info(f"Dataset statistics:")
    logger.info(f"  Training samples: {len(train_data)}")
    logger.info(f"  Validation samples: {len(val_data)}")
    logger.info(f"  Test samples: {len(test_data)}")

    # Step 2: Initialize methods
    logger.info("\n[Step 2/4] Initializing detection methods...")

    # Baseline methods
    baseline = BaselineMethods()

    # Academic methods from papers
    logger.info("  - PhishIntention adapter (USENIX 2022)...")
    phishintention = PhishIntentionAdapter()

    logger.info("  - CNN-BiGRU detector (Sensors 2024)...")
    cnn_bigru = CNNBiGRUDetector()

    logger.info("  - Feature Ensemble detector (uOttawa 2023)...")
    feature_ensemble = FeatureEnsembleDetector()

    logger.info("  - Hybrid LLM-Regex detector (Ours)...")
    hybrid_detector = HybridLLMRegexDetector()

    # Step 3: Run evaluations
    logger.info("\n[Step 3/4] Running evaluations...")
    evaluator = Evaluator()

    results = {}

    # Define all methods to evaluate
    all_methods = [
        # Baseline methods
        ("Rule-based Baseline", baseline.rule_based_detector, False),
        ("Regex Pattern Baseline", baseline.regex_pattern_detector, False),

        # Academic methods from papers
        ("PhishIntention (USENIX'22)", phishintention, "train"),
        ("CNN-BiGRU (Sensors'24)", cnn_bigru, "train"),
        ("Feature Ensemble (uOttawa'23)", feature_ensemble, "train"),

        # Our novel hybrid approach
        ("Hybrid LLM-Regex (Ours)", hybrid_detector, "train")
    ]

    logger.info("\nEvaluating all detection methods:")

    for method_name, method_func, needs_training in all_methods:
        logger.info(f"\n  Testing {method_name}...")
        start_time = time.time()

        try:
            # Train if needed
            if needs_training == True and hasattr(method_func, 'fit'):
                method_func.fit(train_data)
            elif needs_training == "train" and hasattr(method_func, 'train'):
                # Use smaller subset for faster training
                train_subset = train_data[:200] if len(train_data) > 200 else train_data
                val_subset = val_data[:50] if len(val_data) > 50 else val_data
                method_func.train(train_subset, val_subset)

            # Evaluate
            metrics = evaluator.evaluate_method(
                method_func,
                test_data,
                method_name
            )

            elapsed_time = time.time() - start_time
            metrics['time_seconds'] = elapsed_time
            results[method_name] = metrics

            logger.info(f"    Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"    Precision: {metrics['precision']:.3f}")
            logger.info(f"    Recall: {metrics['recall']:.3f}")
            logger.info(f"    F1-Score: {metrics['f1_score']:.3f}")
            logger.info(f"    Time: {elapsed_time:.2f}s")

        except Exception as e:
            logger.error(f"    Error evaluating {method_name}: {e}")
            results[method_name] = {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'error': str(e)
            }

    # Step 4: Generate visualizations and report
    logger.info("\n[Step 4/4] Generating results report and visualizations...")
    visualizer = ResultsVisualizer(results_dir)
    visualizer.generate_all_visualizations(results, test_data)

    # Save results to JSON
    results_file = os.path.join(results_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate summary report
    generate_summary_report(results, results_dir)

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("=" * 80)

    # Print final comparison
    print_results_comparison(results)

    return results

def generate_summary_report(results, results_dir):
    """Generate a markdown summary report"""
    report_file = os.path.join(results_dir, "report.md")

    with open(report_file, 'w') as f:
        f.write("# Phishing Detection: Academic Methods Comparison\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This experiment compares recent academic approaches to phishing detection:\n")
        f.write("- **PhishIntention (USENIX 2022)**: Vision-based approach analyzing brand and credential intentions\n")
        f.write("- **CNN-BiGRU (Sensors 2024)**: Deep learning with 1D-CNN and bidirectional GRU\n")
        f.write("- **Feature Ensemble (uOttawa 2023)**: ML ensemble trained on 737,000 URLs\n")
        f.write("- **Hybrid LLM-Regex (Ours)**: Novel cascaded approach combining regex patterns with LLM semantic analysis\n\n")

        best_method = max(results.items(), key=lambda x: x[1]['f1_score'])
        f.write(f"**Best performing method:** {best_method[0]} ")
        f.write(f"(F1-Score: {best_method[1]['f1_score']:.3f})\n\n")

        f.write("## Detailed Results\n\n")
        f.write("| Method | Paper/Source | Accuracy | Precision | Recall | F1-Score | Time (s) |\n")
        f.write("|--------|--------------|----------|-----------|--------|----------|----------|\n")

        # Add paper sources
        sources = {
            "PhishIntention (USENIX'22)": "USENIX Security 2022",
            "CNN-BiGRU (Sensors'24)": "Sensors 2024, 24(7)",
            "Feature Ensemble (uOttawa'23)": "U. Ottawa 2023",
            "Hybrid LLM-Regex (Ours)": "Novel Contribution",
            "Rule-based Baseline": "Traditional",
            "Regex Pattern Baseline": "Traditional"
        }

        for method_name, metrics in sorted(results.items(),
                                          key=lambda x: x[1]['f1_score'],
                                          reverse=True):
            source = sources.get(method_name, "N/A")
            f.write(f"| {method_name} | {source} | ")
            f.write(f"{metrics['accuracy']:.3f} | ")
            f.write(f"{metrics['precision']:.3f} | ")
            f.write(f"{metrics['recall']:.3f} | ")
            f.write(f"{metrics['f1_score']:.3f} | ")
            f.write(f"{metrics.get('time_seconds', 0):.2f} |\n")

        f.write("\n## Key Findings\n\n")

        # Analyze performance differences
        academic_methods = ["PhishIntention (USENIX'22)", "CNN-BiGRU (Sensors'24)",
                          "Feature Ensemble (uOttawa'23)"]
        baseline_methods = ["Rule-based Baseline", "Regex Pattern Baseline"]

        academic_scores = [results[m]['f1_score'] for m in academic_methods if m in results]
        baseline_scores = [results[m]['f1_score'] for m in baseline_methods if m in results]

        if academic_scores and baseline_scores:
            import numpy as np
            academic_avg = np.mean(academic_scores)
            baseline_avg = np.mean(baseline_scores)
            improvement = ((academic_avg - baseline_avg) / baseline_avg) * 100

            f.write(f"- Academic methods show **{improvement:.1f}%** average improvement over baselines\n")
            f.write(f"- Best academic method: {best_method[0]}\n")
            f.write(f"- Most efficient method: {min(results.items(), key=lambda x: x[1].get('time_seconds', float('inf')))[0]}\n")

        f.write("\n## References\n\n")
        f.write("1. Liu et al., \"Inferring Phishing Intention via Webpage Appearance and Dynamics\", USENIX Security 2022\n")
        f.write("2. \"Advancing Phishing Email Detection: A Comparative Study of Deep Learning Models\", Sensors 2024\n")
        f.write("3. \"Phishing Attack Detection using Machine Learning\", University of Ottawa, 2023\n")

def print_results_comparison(results):
    """Print a formatted comparison of results"""
    print("\n" + "=" * 100)
    print("FINAL RESULTS COMPARISON - ACADEMIC METHODS")
    print("=" * 100)

    # Sort by F1 score
    sorted_results = sorted(results.items(),
                          key=lambda x: x[1]['f1_score'],
                          reverse=True)

    print(f"\n{'Method':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Time(s)':<10}")
    print("-" * 100)

    for method_name, metrics in sorted_results:
        print(f"{method_name:<35} ", end="")
        print(f"{metrics['accuracy']:<12.3f} ", end="")
        print(f"{metrics['precision']:<12.3f} ", end="")
        print(f"{metrics['recall']:<12.3f} ", end="")
        print(f"{metrics['f1_score']:<12.3f} ", end="")
        print(f"{metrics.get('time_seconds', 0):<10.2f}")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        sys.exit(1)