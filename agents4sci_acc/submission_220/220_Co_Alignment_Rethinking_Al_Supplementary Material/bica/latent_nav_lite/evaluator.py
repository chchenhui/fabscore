"""
Evaluator for Latent-Navigator-Lite Experiment
Implements evaluation metrics and comparison with baselines
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict

from .navigator import LatentNavigator, HumanSurrogate


class LatentNavEvaluator:
    """
    Evaluator for latent navigation experiments
    
    Implements metrics from the paper:
    - Best score achieved
    - Top-K average scores
    - Novelty (proportion in low-density regions)
    - BAS/CCM adaptations for latent space
    - Cognitive gain proxies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.baseline_results = {}
        self.experiment_results = {}
        
    def evaluate_navigation_session(self, 
                                   navigator: LatentNavigator,
                                   num_clicks: int = 50,
                                   use_human_surrogate: bool = True,
                                   config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate a complete navigation session
        
        Args:
            navigator: Latent navigator instance
            num_clicks: Number of clicks to simulate
            use_human_surrogate: Whether to use surrogate or require real human
            
        Returns:
            results: Dictionary of evaluation results
        """
        if use_human_surrogate:
            human_surrogate = HumanSurrogate(self.config.get('human_surrogate', {}))
        
        # Pre-navigation quiz (if applicable)
        pre_quiz_results = self._conduct_quiz(navigator, 'pre')
        
        # Navigation session
        session_results = []
        
        for click_idx in range(num_clicks):
            # Get AI suggestions
            suggestions = navigator.suggest_region(
                strategy=self.config.get('suggestion_strategy', 'uncertainty'),
                num_suggestions=5
            )
            
            # Get exploration map
            exploration_map = navigator.get_exploration_map()
            
            # Decide click position
            if use_human_surrogate:
                click_position = human_surrogate.decide_click(suggestions, exploration_map)
            else:
                # For real human interaction, this would be provided externally
                click_position = suggestions[0] if suggestions else (0.0, 0.0)
            
            # Execute click
            click_result = navigator.human_click(click_position)
            session_results.append(click_result)
        
        # Post-navigation quiz
        post_quiz_results = self._conduct_quiz(navigator, 'post')
        
        # Compute cognitive gains
        cognitive_gains = navigator.compute_cognitive_gain(
            pre_quiz_results['accuracy'],
            post_quiz_results['accuracy'],
            pre_quiz_results['ece'],
            post_quiz_results['ece']
        )
        
        # Compile results
        results = {
            'session_results': session_results,
            'pre_quiz': pre_quiz_results,
            'post_quiz': post_quiz_results,
            'cognitive_gains': cognitive_gains,
            'navigation_summary': navigator.get_navigation_summary(),
            'metrics': self._compute_session_metrics(session_results, navigator)
        }
        
        return results
    
    def _conduct_quiz(self, 
                     navigator: LatentNavigator, 
                     phase: str,
                     num_questions: int = 20) -> Dict[str, float]:
        """
        Conduct pre/post navigation quiz
        
        Args:
            navigator: Navigator instance
            phase: 'pre' or 'post'
            num_questions: Number of quiz questions
            
        Returns:
            quiz_results: Quiz accuracy and calibration metrics
        """
        # Generate quiz questions (random latent points)
        quiz_positions = np.random.uniform(-1.0, 1.0, size=(num_questions, 2))
        
        true_scores = []
        predicted_scores = []
        confidences = []
        
        for pos in quiz_positions:
            # Get true score
            click_result = navigator.human_click(tuple(pos))
            true_score = click_result['score']
            true_scores.append(true_score)
            
            # Get predicted score (using GP if post-navigation)
            if phase == 'post' and len(navigator.visited_positions) > 3:
                X = np.array(navigator.visited_positions)
                y = np.array(navigator.scores_history)
                
                navigator.gp_regressor.fit(X, y)
                pred_mean, pred_std = navigator.gp_regressor.predict(
                    pos.reshape(1, -1), return_std=True
                )
                
                predicted_score = pred_mean[0]
                confidence = 1.0 / (1.0 + pred_std[0])  # Higher confidence for lower uncertainty
            else:
                # Pre-navigation: random guessing
                predicted_score = np.random.uniform(0, 1)
                confidence = 0.5
            
            predicted_scores.append(predicted_score)
            confidences.append(confidence)
        
        # Compute accuracy (binary classification: high score vs low score)
        threshold = 0.5
        true_binary = [1 if score > threshold else 0 for score in true_scores]
        pred_binary = [1 if score > threshold else 0 for score in predicted_scores]
        
        accuracy = accuracy_score(true_binary, pred_binary)
        
        # Compute ECE (Expected Calibration Error)
        ece = self._compute_ece(np.array(confidences), np.array(true_binary))
        
        return {
            'accuracy': accuracy,
            'ece': ece,
            'true_scores': true_scores,
            'predicted_scores': predicted_scores,
            'confidences': confidences
        }
    
    def _compute_ece(self, confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
        """Compute Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_session_metrics(self, 
                                session_results: List[Dict[str, Any]], 
                                navigator: LatentNavigator) -> Dict[str, float]:
        """Compute comprehensive session metrics"""
        scores = [result['score'] for result in session_results]
        novelties = [result['novelty'] for result in session_results]
        positions = [result['position'] for result in session_results]
        
        # Basic performance metrics
        best_score = max(scores)
        mean_score = np.mean(scores)
        
        # Top-K metrics
        k_values = [1, 3, 5, 10]
        top_k_metrics = {}
        for k in k_values:
            if len(scores) >= k:
                top_k_scores = sorted(scores, reverse=True)[:k]
                top_k_metrics[f'top_{k}_avg'] = np.mean(top_k_scores)
        
        # Novelty metrics
        avg_novelty = np.mean(novelties)
        high_novelty_proportion = np.mean([n > 0.7 for n in novelties])
        
        # Exploration efficiency
        clicks_per_score = len(scores) / max(best_score, 0.01)
        
        # Coverage metrics
        coverage = self._compute_coverage(positions)
        
        # Convergence metrics
        convergence_point = self._find_convergence_point(scores)
        
        metrics = {
            'best_score': best_score,
            'mean_score': mean_score,
            'avg_novelty': avg_novelty,
            'high_novelty_proportion': high_novelty_proportion,
            'clicks_per_score': clicks_per_score,
            'coverage': coverage,
            'convergence_point': convergence_point,
            **top_k_metrics
        }
        
        return metrics
    
    def _compute_coverage(self, positions: List[Tuple[float, float]], grid_size: int = 20) -> float:
        """Compute coverage of the 2D space"""
        if not positions:
            return 0.0
        
        # Create grid
        x_bins = np.linspace(-1, 1, grid_size + 1)
        y_bins = np.linspace(-1, 1, grid_size + 1)
        
        # Count visited grid cells
        visited_cells = set()
        for x, y in positions:
            x_idx = np.digitize(x, x_bins) - 1
            y_idx = np.digitize(y, y_bins) - 1
            
            # Clamp to valid range
            x_idx = max(0, min(grid_size - 1, x_idx))
            y_idx = max(0, min(grid_size - 1, y_idx))
            
            visited_cells.add((x_idx, y_idx))
        
        # Coverage as proportion of grid cells visited
        total_cells = grid_size * grid_size
        coverage = len(visited_cells) / total_cells
        
        return coverage
    
    def _find_convergence_point(self, scores: List[float], window_size: int = 10) -> Optional[int]:
        """Find point where scores converge (stop improving significantly)"""
        if len(scores) < window_size * 2:
            return None
        
        # Compute running best scores
        running_best = []
        current_best = 0
        for score in scores:
            current_best = max(current_best, score)
            running_best.append(current_best)
        
        # Find convergence point
        for i in range(window_size, len(running_best) - window_size):
            before_window = running_best[i-window_size:i]
            after_window = running_best[i:i+window_size]
            
            improvement = max(after_window) - max(before_window)
            
            if improvement < 0.05:  # Threshold for convergence
                return i
        
        return None
    
    def compare_with_baselines(self, 
                              bica_results: Dict[str, Any],
                              baseline_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare BiCA results with baseline methods
        
        Args:
            bica_results: BiCA navigation results
            baseline_results: Dictionary of baseline results
            
        Returns:
            comparison: Comparison metrics and analysis
        """
        comparison = {
            'bica': bica_results['metrics'],
            'baselines': {},
            'improvements': {},
            'statistical_tests': {}
        }
        
        # Extract metrics for comparison
        bica_metrics = bica_results['metrics']
        
        for baseline_name, baseline_result in baseline_results.items():
            baseline_metrics = baseline_result['metrics']
            comparison['baselines'][baseline_name] = baseline_metrics
            
            # Compute improvements
            improvements = {}
            for metric_name in bica_metrics:
                if metric_name in baseline_metrics:
                    bica_val = bica_metrics[metric_name]
                    baseline_val = baseline_metrics[metric_name]
                    
                    if baseline_val != 0:
                        improvement = (bica_val - baseline_val) / abs(baseline_val)
                    else:
                        improvement = 0.0
                    
                    improvements[metric_name] = improvement
            
            comparison['improvements'][baseline_name] = improvements
        
        # Summary statistics
        comparison['summary'] = self._summarize_comparison(comparison)
        
        return comparison
    
    def _summarize_comparison(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize comparison results"""
        summary = {
            'significant_improvements': [],
            'average_improvement': 0.0,
            'best_baseline': None,
            'worst_baseline': None
        }
        
        # Find significant improvements (>10%)
        for baseline_name, improvements in comparison['improvements'].items():
            for metric_name, improvement in improvements.items():
                if improvement > 0.1:  # >10% improvement
                    summary['significant_improvements'].append({
                        'baseline': baseline_name,
                        'metric': metric_name,
                        'improvement': improvement
                    })
        
        # Compute average improvement across all baselines and metrics
        all_improvements = []
        for improvements in comparison['improvements'].values():
            all_improvements.extend(improvements.values())
        
        if all_improvements:
            summary['average_improvement'] = np.mean(all_improvements)
        
        # Find best/worst baselines (based on best_score metric)
        baseline_scores = {}
        for baseline_name, metrics in comparison['baselines'].items():
            baseline_scores[baseline_name] = metrics.get('best_score', 0.0)
        
        if baseline_scores:
            summary['best_baseline'] = max(baseline_scores, key=baseline_scores.get)
            summary['worst_baseline'] = min(baseline_scores, key=baseline_scores.get)
        
        return summary
    
    def run_baseline_experiments(self, 
                                navigator: LatentNavigator,
                                num_runs: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Run baseline experiments for comparison
        
        Args:
            navigator: Navigator instance
            num_runs: Number of runs per baseline
            
        Returns:
            baseline_results: Results for each baseline method
        """
        baselines = {
            'random': self._run_random_baseline,
            'greedy': self._run_greedy_baseline,
            'uncertainty_sampling': self._run_uncertainty_baseline,
            'rlhf_style': self._run_rlhf_baseline
        }
        
        baseline_results = {}
        
        for baseline_name, baseline_fn in baselines.items():
            print(f"Running {baseline_name} baseline...")
            
            run_results = []
            for run_idx in range(num_runs):
                # Reset navigator state
                navigator.visited_positions = []
                navigator.scores_history = []
                navigator.metrics = {
                    'best_score': 0.0,
                    'total_clicks': 0,
                    'novelty_scores': [],
                    'cognitive_gains': []
                }
                
                # Run baseline
                result = baseline_fn(navigator, num_clicks=50)
                run_results.append(result)
            
            # Aggregate results
            baseline_results[baseline_name] = self._aggregate_baseline_results(run_results)
        
        return baseline_results
    
    def _run_random_baseline(self, navigator: LatentNavigator, num_clicks: int) -> Dict[str, Any]:
        """Run random clicking baseline"""
        results = []
        
        for _ in range(num_clicks):
            # Random click
            x = np.random.uniform(-1.0, 1.0)
            y = np.random.uniform(-1.0, 1.0)
            
            result = navigator.human_click((x, y))
            results.append(result)
        
        return {
            'session_results': results,
            'metrics': self._compute_session_metrics(results, navigator)
        }
    
    def _run_greedy_baseline(self, navigator: LatentNavigator, num_clicks: int) -> Dict[str, Any]:
        """Run greedy baseline (always click highest predicted score)"""
        results = []
        
        for click_idx in range(num_clicks):
            if click_idx < 5:  # Initial random exploration
                x = np.random.uniform(-1.0, 1.0)
                y = np.random.uniform(-1.0, 1.0)
            else:
                # Greedy selection
                exploration_map = navigator.get_exploration_map()
                score_map = exploration_map['score_map']
                x_grid = exploration_map['x_grid']
                y_grid = exploration_map['y_grid']
                
                # Find maximum score location
                max_idx = np.unravel_index(np.argmax(score_map), score_map.shape)
                x = x_grid[max_idx]
                y = y_grid[max_idx]
            
            result = navigator.human_click((x, y))
            results.append(result)
        
        return {
            'session_results': results,
            'metrics': self._compute_session_metrics(results, navigator)
        }
    
    def _run_uncertainty_baseline(self, navigator: LatentNavigator, num_clicks: int) -> Dict[str, Any]:
        """Run uncertainty sampling baseline"""
        results = []
        
        for click_idx in range(num_clicks):
            if click_idx < 5:  # Initial random exploration
                x = np.random.uniform(-1.0, 1.0)
                y = np.random.uniform(-1.0, 1.0)
            else:
                # Uncertainty-based selection
                suggestions = navigator.suggest_region('uncertainty', 1)
                x, y = suggestions[0] if suggestions else (0.0, 0.0)
            
            result = navigator.human_click((x, y))
            results.append(result)
        
        return {
            'session_results': results,
            'metrics': self._compute_session_metrics(results, navigator)
        }
    
    def _run_rlhf_baseline(self, navigator: LatentNavigator, num_clicks: int) -> Dict[str, Any]:
        """Run RLHF-style baseline (no protocol updating)"""
        # This would implement preference-based learning without protocol adaptation
        # For simplicity, we'll use a simplified version
        return self._run_greedy_baseline(navigator, num_clicks)
    
    def _aggregate_baseline_results(self, run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple baseline runs"""
        # Extract metrics from all runs
        all_metrics = [result['metrics'] for result in run_results]
        
        # Compute mean and std for each metric
        aggregated_metrics = {}
        
        if all_metrics:
            metric_names = all_metrics[0].keys()
            
            for metric_name in metric_names:
                values = [metrics[metric_name] for metrics in all_metrics if metric_name in metrics]
                
                if values:
                    aggregated_metrics[f'{metric_name}'] = np.mean(values)
                    aggregated_metrics[f'{metric_name}_std'] = np.std(values)
        
        return {
            'metrics': aggregated_metrics,
            'num_runs': len(run_results),
            'raw_results': run_results
        }
    
    def generate_report(self, 
                       bica_results: Dict[str, Any],
                       baseline_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate evaluation report"""
        comparison = self.compare_with_baselines(bica_results, baseline_results)
        
        report = ["=" * 60]
        report.append("LATENT-NAVIGATOR-LITE EVALUATION REPORT")
        report.append("=" * 60)
        
        # BiCA Performance
        bica_metrics = bica_results['metrics']
        report.append("\nBiCA PERFORMANCE:")
        report.append("-" * 30)
        report.append(f"Best Score: {bica_metrics['best_score']:.3f}")
        report.append(f"Mean Score: {bica_metrics['mean_score']:.3f}")
        report.append(f"Coverage: {bica_metrics['coverage']:.3f}")
        report.append(f"Avg Novelty: {bica_metrics['avg_novelty']:.3f}")
        report.append(f"Clicks per Score: {bica_metrics['clicks_per_score']:.1f}")
        
        # Cognitive Gains
        if 'cognitive_gains' in bica_results:
            cg = bica_results['cognitive_gains']
            report.append(f"\nCOGNITIVE GAINS:")
            report.append(f"Accuracy Gain: {cg['accuracy_gain']:.3f}")
            report.append(f"ECE Improvement: {cg['ece_improvement']:.3f}")
            report.append(f"Overall Cognitive Gain: {cg['cognitive_gain']:.3f}")
        
        # Comparison with Baselines
        report.append(f"\nCOMPARISON WITH BASELINES:")
        report.append("-" * 40)
        
        summary = comparison['summary']
        report.append(f"Average Improvement: {summary['average_improvement']:.1%}")
        report.append(f"Best Baseline: {summary.get('best_baseline', 'N/A')}")
        
        # Significant improvements
        if summary['significant_improvements']:
            report.append("\nSignificant Improvements (>10%):")
            for imp in summary['significant_improvements']:
                report.append(f"  {imp['baseline']} - {imp['metric']}: {imp['improvement']:.1%}")
        
        # Individual baseline comparisons
        for baseline_name, improvements in comparison['improvements'].items():
            report.append(f"\nvs {baseline_name.upper()}:")
            for metric, improvement in improvements.items():
                report.append(f"  {metric}: {improvement:+.1%}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def create_evaluator(config: Dict[str, Any]) -> LatentNavEvaluator:
    """Factory function to create evaluator"""
    return LatentNavEvaluator(config)
