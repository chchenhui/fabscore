"""
Analysis Tools for BiCA Experiments
Statistical analysis and experiment comparison utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """
    Statistical analysis tools for BiCA experiments
    
    Implements statistical tests mentioned in the paper:
    - Paired t-tests
    - Wilcoxon signed-rank tests
    - Bootstrap confidence intervals
    - FDR correction
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compare_methods(self, 
                       bica_results: List[float],
                       baseline_results: List[float],
                       method_name: str = "Baseline") -> Dict[str, Any]:
        """
        Compare BiCA with baseline method using multiple statistical tests
        
        Args:
            bica_results: List of BiCA performance scores
            baseline_results: List of baseline performance scores
            method_name: Name of baseline method
            
        Returns:
            comparison_results: Dictionary with statistical test results
        """
        # Basic statistics
        bica_mean = np.mean(bica_results)
        bica_std = np.std(bica_results)
        baseline_mean = np.mean(baseline_results)
        baseline_std = np.std(baseline_results)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(bica_results) - 1) * bica_std**2 + 
                             (len(baseline_results) - 1) * baseline_std**2) / 
                            (len(bica_results) + len(baseline_results) - 2))
        cohens_d = (bica_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0.0
        
        # Paired t-test (if same number of samples)
        paired_test_results = None
        if len(bica_results) == len(baseline_results):
            t_stat, p_value_paired = stats.ttest_rel(bica_results, baseline_results)
            paired_test_results = {
                'statistic': t_stat,
                'p_value': p_value_paired,
                'significant': p_value_paired < self.alpha
            }
        
        # Independent t-test
        t_stat_ind, p_value_ind = stats.ttest_ind(bica_results, baseline_results)
        independent_test_results = {
            'statistic': t_stat_ind,
            'p_value': p_value_ind,
            'significant': p_value_ind < self.alpha
        }
        
        # Wilcoxon signed-rank test (if paired)
        wilcoxon_results = None
        if len(bica_results) == len(baseline_results):
            w_stat, p_value_wilcoxon = stats.wilcoxon(bica_results, baseline_results)
            wilcoxon_results = {
                'statistic': w_stat,
                'p_value': p_value_wilcoxon,
                'significant': p_value_wilcoxon < self.alpha
            }
        
        # Mann-Whitney U test (independent samples)
        u_stat, p_value_mannwhitney = stats.mannwhitneyu(
            bica_results, baseline_results, alternative='two-sided'
        )
        mannwhitney_results = {
            'statistic': u_stat,
            'p_value': p_value_mannwhitney,
            'significant': p_value_mannwhitney < self.alpha
        }
        
        # Bootstrap confidence intervals
        bica_ci = self._bootstrap_ci(bica_results)
        baseline_ci = self._bootstrap_ci(baseline_results)
        
        # Improvement metrics
        absolute_improvement = bica_mean - baseline_mean
        relative_improvement = (absolute_improvement / abs(baseline_mean)) if baseline_mean != 0 else 0.0
        
        results = {
            'method_name': method_name,
            'bica_stats': {
                'mean': bica_mean,
                'std': bica_std,
                'ci': bica_ci,
                'n': len(bica_results)
            },
            'baseline_stats': {
                'mean': baseline_mean,
                'std': baseline_std,
                'ci': baseline_ci,
                'n': len(baseline_results)
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_cohens_d(cohens_d)
            },
            'improvements': {
                'absolute': absolute_improvement,
                'relative': relative_improvement,
                'percentage': relative_improvement * 100
            },
            'statistical_tests': {
                'paired_t_test': paired_test_results,
                'independent_t_test': independent_test_results,
                'wilcoxon': wilcoxon_results,
                'mann_whitney_u': mannwhitney_results
            }
        }
        
        return results
    
    def _bootstrap_ci(self, 
                     data: List[float], 
                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Compute bootstrap confidence interval"""
        def mean_statistic(x):
            return np.mean(x)
        
        data_array = np.array(data)
        res = bootstrap((data_array,), mean_statistic, n_resamples=n_bootstrap,
                       confidence_level=self.confidence_level, random_state=42)
        
        return (res.confidence_interval.low, res.confidence_interval.high)
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def multiple_comparisons_correction(self, 
                                      p_values: List[float],
                                      method: str = 'fdr_bh') -> Tuple[List[bool], List[float]]:
        """
        Apply multiple comparisons correction
        
        Args:
            p_values: List of p-values
            method: Correction method ('fdr_bh', 'bonferroni', etc.)
            
        Returns:
            rejected: Boolean array indicating rejected hypotheses
            corrected_p_values: Corrected p-values
        """
        rejected, corrected_p_values, _, _ = multipletests(
            p_values, alpha=self.alpha, method=method
        )
        
        return rejected.tolist(), corrected_p_values.tolist()
    
    def power_analysis(self, 
                      effect_size: float,
                      sample_size: int,
                      alpha: float = 0.05) -> float:
        """
        Compute statistical power for given effect size and sample size
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size per group
            alpha: Significance level
            
        Returns:
            power: Statistical power
        """
        from statsmodels.stats.power import ttest_power
        
        power = ttest_power(effect_size, sample_size, alpha, alternative='two-sided')
        return power


class ExperimentAnalyzer:
    """
    High-level experiment analysis and comparison
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.statistical_analyzer = StatisticalAnalyzer(
            confidence_level=config.get('confidence_level', 0.95)
        )
    
    def analyze_experiment_results(self, 
                                  results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """
        Comprehensive analysis of experiment results
        
        Args:
            results: Dictionary with method names as keys and metrics as values
            
        Returns:
            analysis: Comprehensive analysis results
        """
        # Extract BiCA results
        bica_results = results.get('bica', {})
        baseline_results = {k: v for k, v in results.items() if k != 'bica'}
        
        analysis = {
            'summary_statistics': {},
            'pairwise_comparisons': {},
            'overall_ranking': {},
            'significance_summary': {}
        }
        
        # Summary statistics for all methods
        for method_name, method_results in results.items():
            analysis['summary_statistics'][method_name] = self._compute_summary_stats(method_results)
        
        # Pairwise comparisons with BiCA
        all_p_values = []
        comparison_names = []
        
        for baseline_name, baseline_data in baseline_results.items():
            for metric_name in bica_results.keys():
                if metric_name in baseline_data:
                    comparison = self.statistical_analyzer.compare_methods(
                        bica_results[metric_name],
                        baseline_data[metric_name],
                        baseline_name
                    )
                    
                    comparison_key = f"{baseline_name}_{metric_name}"
                    analysis['pairwise_comparisons'][comparison_key] = comparison
                    
                    # Collect p-values for multiple comparisons correction
                    if comparison['statistical_tests']['independent_t_test']:
                        all_p_values.append(comparison['statistical_tests']['independent_t_test']['p_value'])
                        comparison_names.append(comparison_key)
        
        # Multiple comparisons correction
        if all_p_values:
            rejected, corrected_p_values = self.statistical_analyzer.multiple_comparisons_correction(
                all_p_values, method='fdr_bh'
            )
            
            analysis['significance_summary'] = {
                'original_p_values': all_p_values,
                'corrected_p_values': corrected_p_values,
                'rejected_hypotheses': rejected,
                'comparison_names': comparison_names,
                'significant_comparisons': [name for name, rejected in zip(comparison_names, rejected) if rejected]
            }
        
        # Overall ranking
        analysis['overall_ranking'] = self._rank_methods(results)
        
        # Performance summary
        analysis['performance_summary'] = self._summarize_performance(results)
        
        return analysis
    
    def _compute_summary_stats(self, method_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compute summary statistics for a method"""
        summary = {}
        
        for metric_name, values in method_results.items():
            if values:  # Check if list is not empty
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'n': len(values)
                }
        
        return summary
    
    def _rank_methods(self, results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, int]]:
        """Rank methods by performance on each metric"""
        rankings = {}
        
        # Get all metrics
        all_metrics = set()
        for method_results in results.values():
            all_metrics.update(method_results.keys())
        
        for metric in all_metrics:
            metric_means = {}
            
            for method_name, method_results in results.items():
                if metric in method_results and method_results[metric]:
                    metric_means[method_name] = np.mean(method_results[metric])
            
            # Rank methods (higher is better for most metrics)
            if metric in ['collision_rate', 'miscalibration', 'clicks_per_score']:
                # Lower is better for these metrics
                sorted_methods = sorted(metric_means.items(), key=lambda x: x[1])
            else:
                # Higher is better
                sorted_methods = sorted(metric_means.items(), key=lambda x: x[1], reverse=True)
            
            rankings[metric] = {method: rank + 1 for rank, (method, _) in enumerate(sorted_methods)}
        
        return rankings
    
    def _summarize_performance(self, results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Summarize overall performance"""
        summary = {
            'best_method_per_metric': {},
            'average_ranks': {},
            'win_rates': {}
        }
        
        rankings = self._rank_methods(results)
        
        # Best method per metric
        for metric, method_ranks in rankings.items():
            best_method = min(method_ranks.items(), key=lambda x: x[1])[0]
            summary['best_method_per_metric'][metric] = best_method
        
        # Average ranks
        for method_name in results.keys():
            ranks = [method_ranks.get(method_name, len(results)) 
                    for method_ranks in rankings.values()]
            summary['average_ranks'][method_name] = np.mean(ranks)
        
        # Win rates (proportion of metrics where method is best)
        for method_name in results.keys():
            wins = sum(1 for best_method in summary['best_method_per_metric'].values() 
                      if best_method == method_name)
            summary['win_rates'][method_name] = wins / len(rankings) if rankings else 0.0
        
        return summary
    
    def generate_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable analysis report"""
        report = ["=" * 80]
        report.append("BICA EXPERIMENT ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Performance Summary
        report.append("\nPERFORMANCE SUMMARY:")
        report.append("-" * 40)
        
        perf_summary = analysis['performance_summary']
        
        # Average ranks
        report.append("Average Rankings (1 = best):")
        for method, avg_rank in sorted(perf_summary['average_ranks'].items(), 
                                      key=lambda x: x[1]):
            report.append(f"  {method}: {avg_rank:.2f}")
        
        # Win rates
        report.append("\nWin Rates (% of metrics where method is best):")
        for method, win_rate in sorted(perf_summary['win_rates'].items(), 
                                      key=lambda x: x[1], reverse=True):
            report.append(f"  {method}: {win_rate:.1%}")
        
        # Statistical Significance
        if 'significance_summary' in analysis:
            sig_summary = analysis['significance_summary']
            report.append(f"\nSTATISTICAL SIGNIFICANCE:")
            report.append("-" * 40)
            report.append(f"Total comparisons: {len(sig_summary['original_p_values'])}")
            report.append(f"Significant after FDR correction: {sum(sig_summary['rejected_hypotheses'])}")
            
            if sig_summary['significant_comparisons']:
                report.append("\nSignificant improvements for BiCA:")
                for comp_name in sig_summary['significant_comparisons']:
                    baseline, metric = comp_name.rsplit('_', 1)
                    comparison = analysis['pairwise_comparisons'][comp_name]
                    improvement = comparison['improvements']['percentage']
                    report.append(f"  vs {baseline} on {metric}: {improvement:+.1f}%")
        
        # Effect Sizes
        report.append(f"\nEFFECT SIZES (Cohen's d):")
        report.append("-" * 40)
        
        for comp_name, comparison in analysis['pairwise_comparisons'].items():
            baseline, metric = comp_name.rsplit('_', 1)
            cohens_d = comparison['effect_size']['cohens_d']
            interpretation = comparison['effect_size']['interpretation']
            
            report.append(f"{baseline} - {metric}: {cohens_d:.3f} ({interpretation})")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def plot_comparison_summary(self, 
                               analysis: Dict[str, Any],
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot comparison summary visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        perf_summary = analysis['performance_summary']
        
        # Plot 1: Average rankings
        methods = list(perf_summary['average_ranks'].keys())
        avg_ranks = [perf_summary['average_ranks'][method] for method in methods]
        
        axes[0, 0].barh(methods, avg_ranks, color='skyblue')
        axes[0, 0].set_xlabel('Average Rank (lower is better)')
        axes[0, 0].set_title('Method Rankings Across Metrics')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Win rates
        win_rates = [perf_summary['win_rates'][method] * 100 for method in methods]
        
        axes[0, 1].bar(methods, win_rates, color='lightgreen')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].set_title('Win Rates (% Metrics Where Best)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Effect sizes heatmap
        if analysis['pairwise_comparisons']:
            effect_sizes_data = []
            for comp_name, comparison in analysis['pairwise_comparisons'].items():
                baseline, metric = comp_name.rsplit('_', 1)
                effect_sizes_data.append({
                    'Baseline': baseline,
                    'Metric': metric,
                    'Effect Size': comparison['effect_size']['cohens_d']
                })
            
            if effect_sizes_data:
                df = pd.DataFrame(effect_sizes_data)
                pivot_df = df.pivot(index='Baseline', columns='Metric', values='Effect Size')
                
                sns.heatmap(pivot_df, annot=True, cmap='RdYlBu_r', center=0, 
                           ax=axes[1, 0], cbar_kws={'label': "Cohen's d"})
                axes[1, 0].set_title('Effect Sizes (BiCA vs Baselines)')
        
        # Plot 4: Significance indicators
        if 'significance_summary' in analysis:
            sig_data = []
            sig_summary = analysis['significance_summary']
            
            for i, (comp_name, rejected) in enumerate(zip(sig_summary['comparison_names'], 
                                                         sig_summary['rejected_hypotheses'])):
                baseline, metric = comp_name.rsplit('_', 1)
                p_val = sig_summary['corrected_p_values'][i]
                
                sig_data.append({
                    'Comparison': f"{baseline}\n{metric}",
                    'P-value': p_val,
                    'Significant': rejected
                })
            
            if sig_data:
                df_sig = pd.DataFrame(sig_data)
                colors = ['red' if sig else 'gray' for sig in df_sig['Significant']]
                
                axes[1, 1].bar(range(len(df_sig)), -np.log10(df_sig['P-value']), color=colors)
                axes[1, 1].axhline(y=-np.log10(0.05), color='black', linestyle='--', 
                                  label='α = 0.05')
                axes[1, 1].set_ylabel('-log10(p-value)')
                axes[1, 1].set_title('Statistical Significance (Red = Significant)')
                axes[1, 1].set_xticks(range(len(df_sig)))
                axes[1, 1].set_xticklabels(df_sig['Comparison'], rotation=45, ha='right')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_statistical_analyzer(config: Dict[str, Any]) -> StatisticalAnalyzer:
    """Factory function to create statistical analyzer"""
    return StatisticalAnalyzer(
        confidence_level=config.get('confidence_level', 0.95)
    )


def create_experiment_analyzer(config: Dict[str, Any]) -> ExperimentAnalyzer:
    """Factory function to create experiment analyzer"""
    return ExperimentAnalyzer(config)
