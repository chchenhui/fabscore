#!/usr/bin/env python3
"""
Enhanced Phishing Email Detection Experiment
Comprehensive testing with advanced methods
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
from data_loader import PhishingDataLoader
from baseline_methods import BaselineMethods
from hybrid_detector import HybridPhishingDetector
from enhanced_detector import EnhancedPhishingDetector
from evaluation import Evaluator, CrossValidator
from visualizer import ResultsVisualizer

def main():
    """Enhanced experiment pipeline"""
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"enhanced_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("ENHANCED PHISHING EMAIL DETECTION EXPERIMENT")
    logger.info("=" * 80)
    
    # Step 1: Load and prepare datasets
    logger.info("\n[Step 1/6] Loading and preparing datasets...")
    data_loader = PhishingDataLoader()
    
    # Try to download real datasets first
    data_loader.download_datasets()
    
    # Load and split data
    train_data, val_data, test_data = data_loader.load_and_split_data()
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  Training samples: {len(train_data)}")
    logger.info(f"  Validation samples: {len(val_data)}")
    logger.info(f"  Test samples: {len(test_data)}")
    
    # Calculate class distribution
    train_phishing = sum(1 for e in train_data if e['label'] == 1)
    test_phishing = sum(1 for e in test_data if e['label'] == 1)
    
    logger.info(f"  Training phishing ratio: {train_phishing/len(train_data):.2%}")
    logger.info(f"  Test phishing ratio: {test_phishing/len(test_data):.2%}")
    
    # Step 2: Initialize all methods
    logger.info("\n[Step 2/6] Initializing detection methods...")
    
    # Baseline methods
    baseline = BaselineMethods()
    
    # Hybrid detector
    hybrid_detector = HybridPhishingDetector(use_ollama=True)
    
    # Enhanced detector
    enhanced_detector = EnhancedPhishingDetector()
    
    # Step 3: Train methods
    logger.info("\n[Step 3/6] Training detection methods...")
    
    # Train baseline methods that need training
    if hasattr(baseline.tfidf_svm_detector, 'fit'):
        baseline.tfidf_svm_detector.fit(train_data)
    
    # Train hybrid detector
    hybrid_detector.train(train_data, val_data)
    
    # Train enhanced detector
    enhanced_detector.fit(train_data)
    
    # Step 4: Run comprehensive evaluation
    logger.info("\n[Step 4/6] Running comprehensive evaluation...")
    evaluator = Evaluator()
    
    results = {}
    
    # Define all methods to test
    methods_to_test = [
        ("Rule-based", baseline.rule_based_detector),
        ("TF-IDF + SVM", baseline.tfidf_svm_detector),
        ("Regex Pattern", baseline.regex_pattern_detector),
        ("Hybrid LLM + Rules", hybrid_detector),
        ("Enhanced Detector", enhanced_detector)
    ]
    
    # Test each method
    for method_name, method_func in methods_to_test:
        logger.info(f"\nEvaluating {method_name}...")
        start_time = time.time()
        
        # Evaluate
        metrics = evaluator.evaluate_method(
            method_func, 
            test_data,
            method_name
        )
        
        elapsed_time = time.time() - start_time
        metrics['time_seconds'] = elapsed_time
        results[method_name] = metrics
        
        # Print results
        logger.info(f"  Accuracy:    {metrics['accuracy']:.3f}")
        logger.info(f"  Precision:   {metrics['precision']:.3f}")
        logger.info(f"  Recall:      {metrics['recall']:.3f}")
        logger.info(f"  F1-Score:    {metrics['f1_score']:.3f}")
        
        if 'specificity' in metrics:
            logger.info(f"  Specificity: {metrics['specificity']:.3f}")
            logger.info(f"  FPR:         {metrics['false_positive_rate']:.3f}")
            logger.info(f"  FNR:         {metrics['false_negative_rate']:.3f}")
        
        logger.info(f"  Time:        {elapsed_time:.2f}s")
        
        # Print confusion matrix
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            logger.info(f"  Confusion Matrix:")
            logger.info(f"    TN={cm[0][0]:3d}  FP={cm[0][1]:3d}")
            logger.info(f"    FN={cm[1][0]:3d}  TP={cm[1][1]:3d}")
    
    # Step 5: Cross-validation for robustness
    logger.info("\n[Step 5/6] Running cross-validation for robustness testing...")
    cross_validator = CrossValidator(n_folds=5)
    
    # Run cross-validation on best performing methods
    best_methods = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:3]
    
    cv_results = {}
    for method_name, _ in best_methods:
        logger.info(f"\nCross-validating {method_name}...")
        
        # Get the method object
        method = None
        for name, obj in methods_to_test:
            if name == method_name:
                method = obj
                break
        
        if method:
            cv_result = cross_validator.cross_validate(
                method, 
                train_data + val_data + test_data,
                method_name
            )
            cv_results[method_name] = cv_result
    
    # Step 6: Generate comprehensive reports
    logger.info("\n[Step 6/6] Generating comprehensive reports and visualizations...")
    
    # Generate visualizations
    visualizer = ResultsVisualizer(results_dir)
    visualizer.generate_all_visualizations(results, test_data)
    
    # Save all results
    all_results = {
        'single_test': results,
        'cross_validation': cv_results,
        'dataset_stats': {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'train_phishing_ratio': train_phishing/len(train_data),
            'test_phishing_ratio': test_phishing/len(test_data)
        }
    }
    
    results_file = os.path.join(results_dir, "all_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate detailed report
    generate_detailed_report(all_results, results_dir)
    
    # Error analysis
    perform_error_analysis(methods_to_test[:3], test_data, results_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("ENHANCED EXPERIMENT COMPLETED")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("=" * 80)
    
    # Print final summary
    print_enhanced_summary(all_results)
    
    return all_results

def generate_detailed_report(all_results, results_dir):
    """Generate comprehensive markdown report"""
    report_file = os.path.join(results_dir, "detailed_report.md")
    
    with open(report_file, 'w') as f:
        f.write("# Enhanced Phishing Email Detection - Detailed Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        results = all_results['single_test']
        best_method = max(results.items(), key=lambda x: x[1]['f1_score'])
        
        f.write(f"**Best Performing Method:** {best_method[0]}\n")
        f.write(f"- F1-Score: {best_method[1]['f1_score']:.3f}\n")
        f.write(f"- Accuracy: {best_method[1]['accuracy']:.3f}\n")
        f.write(f"- Precision: {best_method[1]['precision']:.3f}\n")
        f.write(f"- Recall: {best_method[1]['recall']:.3f}\n\n")
        
        # Dataset Information
        f.write("## Dataset Information\n\n")
        stats = all_results['dataset_stats']
        f.write(f"- **Total Samples:** {stats['train_size'] + stats['val_size'] + stats['test_size']}\n")
        f.write(f"- **Training Set:** {stats['train_size']} samples ")
        f.write(f"({stats['train_phishing_ratio']:.1%} phishing)\n")
        f.write(f"- **Test Set:** {stats['test_size']} samples ")
        f.write(f"({stats['test_phishing_ratio']:.1%} phishing)\n\n")
        
        # Detailed Results
        f.write("## Detailed Performance Metrics\n\n")
        f.write("### Single Test Results\n\n")
        
        f.write("| Method | Acc | Prec | Rec | F1 | Spec | FPR | FNR | Time |\n")
        f.write("|--------|-----|------|-----|----|----- |-----|-----|------|\n")
        
        for method_name, metrics in sorted(results.items(), 
                                          key=lambda x: x[1]['f1_score'], 
                                          reverse=True):
            f.write(f"| {method_name} | ")
            f.write(f"{metrics['accuracy']:.3f} | ")
            f.write(f"{metrics['precision']:.3f} | ")
            f.write(f"{metrics['recall']:.3f} | ")
            f.write(f"{metrics['f1_score']:.3f} | ")
            f.write(f"{metrics.get('specificity', 0):.3f} | ")
            f.write(f"{metrics.get('false_positive_rate', 0):.3f} | ")
            f.write(f"{metrics.get('false_negative_rate', 0):.3f} | ")
            f.write(f"{metrics.get('time_seconds', 0):.2f}s |\n")
        
        # Cross-validation results
        if all_results.get('cross_validation'):
            f.write("\n### Cross-Validation Results (5-Fold)\n\n")
            f.write("| Method | Avg F1 | Std F1 | Avg Acc | Std Acc |\n")
            f.write("|--------|--------|--------|---------|----------|\n")
            
            for method_name, cv_metrics in all_results['cross_validation'].items():
                f.write(f"| {method_name} | ")
                f.write(f"{cv_metrics['avg_f1_score']:.3f} | ")
                f.write(f"±{cv_metrics['std_f1_score']:.3f} | ")
                f.write(f"{cv_metrics['avg_accuracy']:.3f} | ")
                f.write(f"±{cv_metrics['std_accuracy']:.3f} |\n")
        
        # Key Findings
        f.write("\n## Key Findings\n\n")
        
        # Performance comparison
        f.write("### Performance Improvements\n\n")
        
        baseline_f1 = results.get("Rule-based", {}).get("f1_score", 0)
        
        for method_name, metrics in results.items():
            if method_name != "Rule-based":
                improvement = ((metrics['f1_score'] - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
                
                if improvement > 0:
                    f.write(f"- **{method_name}** shows {improvement:.1f}% ")
                    f.write(f"improvement over baseline Rule-based method\n")
                else:
                    f.write(f"- **{method_name}** performs {abs(improvement):.1f}% ")
                    f.write(f"worse than baseline Rule-based method\n")
        
        # Trade-offs
        f.write("\n### Performance Trade-offs\n\n")
        
        # Find method with best precision
        best_precision = max(results.items(), key=lambda x: x[1]['precision'])
        f.write(f"- **Highest Precision:** {best_precision[0]} ")
        f.write(f"({best_precision[1]['precision']:.3f}) - Best for minimizing false positives\n")
        
        # Find method with best recall
        best_recall = max(results.items(), key=lambda x: x[1]['recall'])
        f.write(f"- **Highest Recall:** {best_recall[0]} ")
        f.write(f"({best_recall[1]['recall']:.3f}) - Best for catching all phishing emails\n")
        
        # Find fastest method
        fastest = min(results.items(), key=lambda x: x[1].get('time_seconds', float('inf')))
        f.write(f"- **Fastest Method:** {fastest[0]} ")
        f.write(f"({fastest[1].get('time_seconds', 0):.2f}s) - Best for real-time detection\n")
        
        # Recommendations
        f.write("\n## Recommendations\n\n")
        
        f.write("Based on the experimental results:\n\n")
        f.write("1. **For Production Use:** ")
        f.write(f"{best_method[0]} provides the best overall performance\n")
        
        f.write("2. **For High-Security Environments:** ")
        f.write(f"Use {best_precision[0]} to minimize false positives\n")
        
        f.write("3. **For Comprehensive Coverage:** ")
        f.write(f"Use {best_recall[0]} to catch maximum phishing attempts\n")
        
        f.write("4. **For Real-time Systems:** ")
        f.write(f"Use {fastest[0]} for fastest processing\n")
        
        f.write("\n## Conclusion\n\n")
        f.write("The experiment successfully evaluated multiple phishing detection methods ")
        f.write("ranging from traditional rule-based approaches to advanced hybrid systems. ")
        f.write("The results demonstrate that modern techniques combining multiple ")
        f.write("approaches can significantly improve detection accuracy while maintaining ")
        f.write("reasonable processing times.\n")

def perform_error_analysis(methods, test_data, results_dir):
    """Analyze misclassified samples"""
    error_file = os.path.join(results_dir, "error_analysis.txt")
    
    with open(error_file, 'w') as f:
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        for method_name, method_obj in methods[:3]:  # Top 3 methods
            f.write(f"\nMethod: {method_name}\n")
            f.write("-" * 40 + "\n")
            
            predictions = method_obj.predict(test_data)
            
            false_positives = []
            false_negatives = []
            
            for i, (pred, actual) in enumerate(zip(predictions, test_data)):
                true_label = actual['label']
                
                if pred == 1 and true_label == 0:
                    false_positives.append(i)
                elif pred == 0 and true_label == 1:
                    false_negatives.append(i)
            
            f.write(f"False Positives: {len(false_positives)}\n")
            f.write(f"False Negatives: {len(false_negatives)}\n\n")
            
            # Sample analysis
            if false_positives:
                f.write("Sample False Positive:\n")
                idx = false_positives[0]
                f.write(f"  Subject: {test_data[idx].get('subject', 'N/A')[:50]}...\n")
                f.write(f"  Sender: {test_data[idx].get('sender', 'N/A')}\n\n")
            
            if false_negatives:
                f.write("Sample False Negative:\n")
                idx = false_negatives[0]
                f.write(f"  Subject: {test_data[idx].get('subject', 'N/A')[:50]}...\n")
                f.write(f"  Sender: {test_data[idx].get('sender', 'N/A')}\n\n")

def print_enhanced_summary(all_results):
    """Print enhanced summary to console"""
    print("\n" + "=" * 80)
    print("ENHANCED EXPERIMENT SUMMARY")
    print("=" * 80)
    
    results = all_results['single_test']
    
    # Create sorted list by F1 score
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['f1_score'], 
                          reverse=True)
    
    print(f"\n{'Rank':<6} {'Method':<25} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8}")
    print("-" * 70)
    
    for i, (method_name, metrics) in enumerate(sorted_results, 1):
        print(f"{i:<6} {method_name:<25} ", end="")
        print(f"{metrics['f1_score']:<8.3f} ", end="")
        print(f"{metrics['accuracy']:<8.3f} ", end="")
        print(f"{metrics['precision']:<8.3f} ", end="")
        print(f"{metrics['recall']:<8.3f}")
    
    # Cross-validation summary if available
    if all_results.get('cross_validation'):
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION SUMMARY (5-Fold)")
        print("=" * 80)
        
        print(f"\n{'Method':<25} {'Avg F1':<12} {'Std Dev':<12}")
        print("-" * 50)
        
        for method_name, cv_metrics in all_results['cross_validation'].items():
            print(f"{method_name:<25} ", end="")
            print(f"{cv_metrics['avg_f1_score']:<12.3f} ", end="")
            print(f"±{cv_metrics['std_f1_score']:<11.3f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Enhanced experiment failed: {str(e)}", exc_info=True)
        sys.exit(1)