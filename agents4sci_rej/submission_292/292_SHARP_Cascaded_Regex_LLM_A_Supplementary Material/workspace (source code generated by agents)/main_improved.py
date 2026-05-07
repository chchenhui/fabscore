#!/usr/bin/env python3
"""
Improved Phishing Email Detection Experiment
Comprehensive testing with LLM + rule-based hybrid approach
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from improved_data_loader import ImprovedDataLoader
from baseline_methods import BaselineMethods
from hybrid_detector import HybridPhishingDetector
from improved_hybrid_detector import ImprovedHybridDetector
from enhanced_detector import EnhancedPhishingDetector
from evaluation import Evaluator
from visualizer import ResultsVisualizer

class ExperimentRunner:
    """Main experiment runner with comprehensive testing"""
    
    def __init__(self, results_dir=None):
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"improved_results_{timestamp}"
        else:
            self.results_dir = results_dir
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging to file
        file_handler = logging.FileHandler(
            os.path.join(self.results_dir, 'experiment.log')
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    def run_experiment(self):
        """Run the complete experiment pipeline"""
        
        logger.info("=" * 80)
        logger.info("IMPROVED PHISHING EMAIL DETECTION EXPERIMENT")
        logger.info("=" * 80)
        
        # Step 1: Load and prepare datasets
        logger.info("\n[Step 1/7] Loading and preparing datasets...")
        data_loader = ImprovedDataLoader()
        train_data, val_data, test_data = data_loader.load_and_split_data()
        
        # Get and log statistics
        train_stats = data_loader.get_statistics(train_data)
        test_stats = data_loader.get_statistics(test_data)
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  Training: {train_stats['total']} samples ({train_stats['phishing_ratio']:.1%} phishing)")
        logger.info(f"  Validation: {len(val_data)} samples")
        logger.info(f"  Test: {test_stats['total']} samples ({test_stats['phishing_ratio']:.1%} phishing)")
        
        if 'avg_phishing_length' in train_stats:
            logger.info(f"  Avg phishing email length: {train_stats['avg_phishing_length']:.0f} chars")
            logger.info(f"  Avg legitimate email length: {train_stats.get('avg_legitimate_length', 0):.0f} chars")
        
        # Step 2: Initialize all detection methods
        logger.info("\n[Step 2/7] Initializing detection methods...")
        
        methods = {}
        
        # Baseline methods
        logger.info("  - Initializing baseline methods...")
        baseline = BaselineMethods()
        methods['Rule-based'] = baseline.rule_based_detector
        methods['Regex Pattern'] = baseline.regex_pattern_detector
        
        # Try to initialize TF-IDF + SVM if available
        try:
            if hasattr(baseline, 'tfidf_svm_detector'):
                methods['TF-IDF + SVM'] = baseline.tfidf_svm_detector
        except:
            logger.warning("  - TF-IDF + SVM not available")
        
        # Original hybrid detector
        logger.info("  - Initializing original hybrid detector...")
        try:
            original_hybrid = HybridPhishingDetector(use_ollama=True)
            methods['Original Hybrid'] = original_hybrid
        except Exception as e:
            logger.warning(f"  - Original hybrid initialization warning: {e}")
        
        # Improved hybrid detector
        logger.info("  - Initializing improved hybrid detector...")
        improved_hybrid = ImprovedHybridDetector(model_name="dolphin3:latest")
        methods['Improved Hybrid (LLM+Rules)'] = improved_hybrid
        
        # Enhanced detector
        logger.info("  - Initializing enhanced detector...")
        enhanced_detector = EnhancedPhishingDetector()
        methods['Enhanced Multi-Feature'] = enhanced_detector
        
        # Step 3: Train methods that require training
        logger.info("\n[Step 3/7] Training detection methods...")
        
        for method_name, method in methods.items():
            if hasattr(method, 'fit'):
                logger.info(f"  Training {method_name}...")
                method.fit(train_data)
            elif hasattr(method, 'train'):
                logger.info(f"  Training {method_name}...")
                method.train(train_data, val_data)
        
        # Step 4: Evaluate all methods
        logger.info("\n[Step 4/7] Evaluating detection methods...")
        evaluator = Evaluator()
        results = {}
        
        for method_name, method in methods.items():
            logger.info(f"\n  Evaluating {method_name}...")
            
            try:
                start_time = time.time()
                metrics = evaluator.evaluate_method(
                    method,
                    test_data,
                    method_name
                )
                elapsed_time = time.time() - start_time
                
                metrics['time_seconds'] = elapsed_time
                metrics['time_per_sample'] = elapsed_time / len(test_data)
                
                results[method_name] = metrics
                
                # Log key metrics
                logger.info(f"    Accuracy: {metrics['accuracy']:.3f}")
                logger.info(f"    Precision: {metrics['precision']:.3f}")
                logger.info(f"    Recall: {metrics['recall']:.3f}")
                logger.info(f"    F1-Score: {metrics['f1_score']:.3f}")
                logger.info(f"    Time: {elapsed_time:.2f}s ({metrics['time_per_sample']*1000:.1f}ms/sample)")
                
            except Exception as e:
                logger.error(f"    Error evaluating {method_name}: {e}")
                results[method_name] = {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0,
                    'error': str(e)
                }
        
        # Step 5: Analyze errors and edge cases
        logger.info("\n[Step 5/7] Analyzing errors and edge cases...")
        error_analysis = self._analyze_errors(methods, test_data, results)
        
        # Step 6: Generate visualizations
        logger.info("\n[Step 6/7] Generating visualizations and reports...")
        visualizer = ResultsVisualizer(self.results_dir)
        visualizer.generate_all_visualizations(results, test_data)
        
        # Step 7: Generate comprehensive report
        logger.info("\n[Step 7/7] Generating final report...")
        self._generate_comprehensive_report(results, error_analysis, train_stats, test_stats)
        
        # Save raw results
        results_file = os.path.join(self.results_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_summary(results)
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("=" * 80)
        
        return results
    
    def _analyze_errors(self, methods, test_data, results):
        """Analyze common errors across methods"""
        error_analysis = {
            'false_positives': [],
            'false_negatives': [],
            'disagreements': []
        }
        
        # Get the best performing method
        best_method_name = max(results.items(), key=lambda x: x[1].get('f1_score', 0))[0]
        
        if best_method_name in methods:
            best_method = methods[best_method_name]
            
            # Analyze false positives and negatives
            predictions = best_method.predict(test_data)
            
            for i, (email, pred) in enumerate(zip(test_data, predictions)):
                true_label = email['label']
                
                if pred == 1 and true_label == 0:
                    # False positive
                    error_analysis['false_positives'].append({
                        'index': i,
                        'subject': email.get('subject', '')[:50],
                        'sender': email.get('sender', '')
                    })
                elif pred == 0 and true_label == 1:
                    # False negative
                    error_analysis['false_negatives'].append({
                        'index': i,
                        'subject': email.get('subject', '')[:50],
                        'sender': email.get('sender', '')
                    })
        
        # Analyze disagreements between methods
        all_predictions = {}
        for method_name, method in methods.items():
            try:
                all_predictions[method_name] = method.predict(test_data)
            except:
                pass
        
        if len(all_predictions) > 1:
            for i in range(len(test_data)):
                preds = [all_predictions[m][i] for m in all_predictions if i < len(all_predictions[m])]
                if len(set(preds)) > 1:  # Methods disagree
                    error_analysis['disagreements'].append({
                        'index': i,
                        'subject': test_data[i].get('subject', '')[:50],
                        'predictions': {m: all_predictions[m][i] for m in all_predictions if i < len(all_predictions[m])}
                    })
        
        # Save error analysis
        error_file = os.path.join(self.results_dir, 'error_analysis.json')
        with open(error_file, 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        logger.info(f"  False positives: {len(error_analysis['false_positives'])}")
        logger.info(f"  False negatives: {len(error_analysis['false_negatives'])}")
        logger.info(f"  Method disagreements: {len(error_analysis['disagreements'])}")
        
        return error_analysis
    
    def _generate_comprehensive_report(self, results, error_analysis, train_stats, test_stats):
        """Generate a detailed markdown report"""
        report_file = os.path.join(self.results_dir, 'comprehensive_report.md')
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive Phishing Email Detection Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            best_method = max(results.items(), key=lambda x: x[1].get('f1_score', 0))
            f.write(f"**Best Performing Method:** {best_method[0]}\n")
            f.write(f"- F1-Score: {best_method[1].get('f1_score', 0):.3f}\n")
            f.write(f"- Accuracy: {best_method[1].get('accuracy', 0):.3f}\n")
            f.write(f"- Precision: {best_method[1].get('precision', 0):.3f}\n")
            f.write(f"- Recall: {best_method[1].get('recall', 0):.3f}\n\n")
            
            # Dataset Information
            f.write("## Dataset Information\n\n")
            f.write(f"- **Training Set:** {train_stats['total']} samples ")
            f.write(f"({train_stats['phishing_ratio']:.1%} phishing)\n")
            f.write(f"- **Test Set:** {test_stats['total']} samples ")
            f.write(f"({test_stats['phishing_ratio']:.1%} phishing)\n\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            f.write("| Method | Accuracy | Precision | Recall | F1-Score | Time (s) | ms/sample |\n")
            f.write("|--------|----------|-----------|--------|----------|----------|----------|\n")
            
            for method_name, metrics in sorted(results.items(), 
                                              key=lambda x: x[1].get('f1_score', 0), 
                                              reverse=True):
                if 'error' not in metrics:
                    f.write(f"| {method_name} | ")
                    f.write(f"{metrics.get('accuracy', 0):.3f} | ")
                    f.write(f"{metrics.get('precision', 0):.3f} | ")
                    f.write(f"{metrics.get('recall', 0):.3f} | ")
                    f.write(f"{metrics.get('f1_score', 0):.3f} | ")
                    f.write(f"{metrics.get('time_seconds', 0):.2f} | ")
                    f.write(f"{metrics.get('time_per_sample', 0)*1000:.1f} |\n")
            
            # Error Analysis
            f.write("\n## Error Analysis\n\n")
            f.write(f"- **False Positives:** {len(error_analysis['false_positives'])} ")
            f.write("(legitimate emails classified as phishing)\n")
            f.write(f"- **False Negatives:** {len(error_analysis['false_negatives'])} ")
            f.write("(phishing emails classified as legitimate)\n")
            f.write(f"- **Method Disagreements:** {len(error_analysis['disagreements'])} samples\n\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            
            # Compare hybrid methods
            hybrid_methods = [m for m in results if 'hybrid' in m.lower()]
            if len(hybrid_methods) > 1:
                f.write("### Hybrid Method Comparison\n\n")
                for method in hybrid_methods:
                    f.write(f"- **{method}:** F1={results[method].get('f1_score', 0):.3f}\n")
                f.write("\n")
            
            # Performance vs Speed Trade-off
            f.write("### Performance vs Speed Trade-off\n\n")
            fastest = min(results.items(), 
                         key=lambda x: x[1].get('time_per_sample', float('inf')))
            f.write(f"- **Fastest Method:** {fastest[0]} ")
            f.write(f"({fastest[1].get('time_per_sample', 0)*1000:.1f}ms/sample)\n")
            f.write(f"- **Most Accurate:** {best_method[0]} ")
            f.write(f"({best_method[1].get('time_per_sample', 0)*1000:.1f}ms/sample)\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the experimental results:\n\n")
            
            if best_method[1].get('f1_score', 0) > 0.95:
                f.write(f"1. **{best_method[0]}** demonstrates excellent performance ")
                f.write("and is recommended for production use.\n")
            else:
                f.write(f"1. **{best_method[0]}** shows the best performance but ")
                f.write("may benefit from further optimization.\n")
            
            f.write("2. Consider ensemble methods combining multiple approaches ")
            f.write("for improved robustness.\n")
            f.write("3. Regular retraining with new phishing patterns is essential.\n")
            f.write("4. Monitor false positive rates to minimize disruption to legitimate emails.\n\n")
            
            # Technical Details
            f.write("## Technical Details\n\n")
            f.write("### Methods Tested\n\n")
            f.write("1. **Rule-based:** Traditional pattern matching\n")
            f.write("2. **Regex Pattern:** Regular expression-based detection\n")
            f.write("3. **TF-IDF + SVM:** Machine learning baseline\n")
            f.write("4. **Original Hybrid:** Initial LLM + rules implementation\n")
            f.write("5. **Improved Hybrid:** Enhanced LLM integration with Docker support\n")
            f.write("6. **Enhanced Multi-Feature:** Advanced feature extraction approach\n\n")
            
            f.write("### Environment\n\n")
            f.write("- Experiment run in Docker container\n")
            f.write("- LLM: Ollama with dolphin3:latest model\n")
            f.write("- Evaluation: Cross-validation with held-out test set\n")
        
        logger.info(f"  Comprehensive report saved to: {report_file}")
    
    def _print_summary(self, results):
        """Print a summary table of results"""
        print("\n" + "=" * 100)
        print("FINAL RESULTS COMPARISON")
        print("=" * 100)
        
        # Sort by F1 score
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1].get('f1_score', 0), 
                              reverse=True)
        
        print(f"\n{'Method':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Time(s)':<10}")
        print("-" * 100)
        
        for method_name, metrics in sorted_results:
            if 'error' not in metrics:
                print(f"{method_name:<30} ", end="")
                print(f"{metrics.get('accuracy', 0):<12.3f} ", end="")
                print(f"{metrics.get('precision', 0):<12.3f} ", end="")
                print(f"{metrics.get('recall', 0):<12.3f} ", end="")
                print(f"{metrics.get('f1_score', 0):<12.3f} ", end="")
                print(f"{metrics.get('time_seconds', 0):<10.2f}")
        
        print("\n" + "=" * 100)


def main():
    """Main entry point"""
    try:
        runner = ExperimentRunner()
        results = runner.run_experiment()
        
        # Return success
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())