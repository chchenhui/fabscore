"""
Statistical Analysis and Visualization Tools
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path


class StatisticalAnalyzer:
    """
    Statistical analysis for hierarchical meta-learning results.
    """
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def compare_methods_statistical(self, 
                                  results_dict: Dict,
                                  baseline_method: str = 'RandomForest',
                                  alpha: float = 0.05) -> Dict:
        """
        Perform statistical comparison between methods.
        
        Args:
            results_dict: Dictionary containing results for different methods
            baseline_method: Baseline method for comparison
            alpha: Significance level
            
        Returns:
            statistical_results: Statistical test results
        """
        self.logger.info("Performing statistical comparisons...")
        
        if baseline_method not in results_dict:
            raise ValueError(f"Baseline method {baseline_method} not found in results")
        
        baseline_scores = results_dict[baseline_method]
        statistical_results = {}
        
        for method_name, method_scores in results_dict.items():
            if method_name == baseline_method:
                continue
            
            # Paired t-test
            t_stat, t_pvalue = stats.ttest_rel(method_scores, baseline_scores)
            
            # Wilcoxon signed-rank test (non-parametric)
            wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(method_scores, baseline_scores)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(method_scores) + np.var(baseline_scores)) / 2)
            cohens_d = (np.mean(method_scores) - np.mean(baseline_scores)) / pooled_std
            
            # Mann-Whitney U test (independent samples)
            u_stat, u_pvalue = stats.mannwhitneyu(method_scores, baseline_scores, alternative='two-sided')
            
            statistical_results[method_name] = {
                'paired_t_test': {
                    't_statistic': t_stat,
                    'p_value': t_pvalue,
                    'significant': t_pvalue < alpha
                },
                'wilcoxon_test': {
                    'statistic': wilcoxon_stat,
                    'p_value': wilcoxon_pvalue,
                    'significant': wilcoxon_pvalue < alpha
                },
                'mann_whitney_u': {
                    'u_statistic': u_stat,
                    'p_value': u_pvalue,
                    'significant': u_pvalue < alpha
                },
                'effect_size': {
                    'cohens_d': cohens_d,
                    'interpretation': self._interpret_cohens_d(cohens_d)
                },
                'descriptive_stats': {
                    'mean_difference': np.mean(method_scores) - np.mean(baseline_scores),
                    'median_difference': np.median(method_scores) - np.median(baseline_scores),
                    'method_mean': np.mean(method_scores),
                    'method_std': np.std(method_scores),
                    'baseline_mean': np.mean(baseline_scores),
                    'baseline_std': np.std(baseline_scores)
                }
            }
        
        # Multiple comparison correction
        p_values = [result['paired_t_test']['p_value'] for result in statistical_results.values()]
        corrected_p_values = multipletests(p_values, method='bonferroni')[1]
        
        for i, method_name in enumerate(statistical_results.keys()):
            statistical_results[method_name]['corrected_p_value'] = corrected_p_values[i]
            statistical_results[method_name]['significant_corrected'] = corrected_p_values[i] < alpha
        
        return statistical_results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def analyze_few_shot_learning_curve(self, few_shot_results: Dict) -> Dict:
        """Analyze learning curves across few-shot scenarios."""
        
        shot_sizes = []
        accuracies = []
        std_errors = []
        
        for scenario, results in few_shot_results.items():
            if 'shot' in scenario:
                shot_size = int(scenario.split('_')[0])
                accuracy = results['molecular_accuracy']['mean']
                std_error = results['molecular_accuracy']['std'] / np.sqrt(100)  # Assuming 100 episodes
                
                shot_sizes.append(shot_size)
                accuracies.append(accuracy)
                std_errors.append(std_error)
        
        # Sort by shot size
        sorted_indices = np.argsort(shot_sizes)
        shot_sizes = np.array(shot_sizes)[sorted_indices]
        accuracies = np.array(accuracies)[sorted_indices]
        std_errors = np.array(std_errors)[sorted_indices]
        
        # Fit learning curve
        # Use power law: accuracy = a * (shot_size)^b + c
        def power_law(x, a, b, c):
            return a * np.power(x, b) + c
        
        try:
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(power_law, shot_sizes, accuracies, maxfev=1000)
            
            # Generate smooth curve for visualization
            x_smooth = np.linspace(shot_sizes.min(), shot_sizes.max(), 100)
            y_smooth = power_law(x_smooth, *popt)
            
            learning_curve_analysis = {
                'shot_sizes': shot_sizes.tolist(),
                'accuracies': accuracies.tolist(),
                'std_errors': std_errors.tolist(),
                'fitted_curve': {
                    'x': x_smooth.tolist(),
                    'y': y_smooth.tolist(),
                    'parameters': {
                        'a': popt[0],
                        'b': popt[1],
                        'c': popt[2]
                    }
                },
                'r_squared': self._calculate_r_squared(accuracies, power_law(shot_sizes, *popt))
            }
        except:
            learning_curve_analysis = {
                'shot_sizes': shot_sizes.tolist(),
                'accuracies': accuracies.tolist(),
                'std_errors': std_errors.tolist(),
                'fitted_curve': None
            }
        
        return learning_curve_analysis
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared value."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


class VisualizationGenerator:
    """
    Generate comprehensive visualizations for hierarchical meta-learning results.
    """
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_performance_comparison_plot(self, 
                                         baseline_results: Dict,
                                         meta_learning_results: Dict,
                                         save_name: str = 'performance_comparison') -> str:
        """Create comprehensive performance comparison plot."""
        
        # Prepare data
        methods = []
        accuracies = []
        method_types = []
        
        # Baseline methods
        for method, results in baseline_results.items():
            if 'val_accuracy' in results:
                methods.append(method)
                accuracies.append(results['val_accuracy'])
                method_types.append('Baseline')
        
        # Meta-learning results (5-shot scenario)
        if '5_shot_15_query' in meta_learning_results:
            methods.append('Hierarchical MAML')
            accuracies.append(meta_learning_results['5_shot_15_query']['molecular_accuracy']['mean'])
            method_types.append('Meta-Learning')
        
        # Create DataFrame
        df = pd.DataFrame({
            'Method': methods,
            'Accuracy': accuracies,
            'Type': method_types
        })
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Bar plot
        sns.barplot(data=df, x='Method', y='Accuracy', hue='Type', ax=ax)
        
        # Customize plot
        ax.set_title('Performance Comparison: Baselines vs Hierarchical Meta-Learning', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_xlabel('Method', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add horizontal line for best baseline
        best_baseline = max([acc for acc, typ in zip(accuracies, method_types) if typ == 'Baseline'])
        ax.axhline(y=best_baseline, color='red', linestyle='--', alpha=0.7, 
                  label=f'Best Baseline: {best_baseline:.3f}')
        
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        save_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_few_shot_learning_curve(self, 
                                     learning_curve_data: Dict,
                                     save_name: str = 'few_shot_learning_curve') -> str:
        """Create few-shot learning curve visualization."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        shot_sizes = learning_curve_data['shot_sizes']
        accuracies = learning_curve_data['accuracies']
        std_errors = learning_curve_data['std_errors']
        
        # Plot data points with error bars
        ax.errorbar(shot_sizes, accuracies, yerr=std_errors, 
                   marker='o', markersize=8, capsize=5, capthick=2,
                   linewidth=2, label='Observed Performance')
        
        # Plot fitted curve if available
        if learning_curve_data.get('fitted_curve'):
            fitted_x = learning_curve_data['fitted_curve']['x']
            fitted_y = learning_curve_data['fitted_curve']['y']
            ax.plot(fitted_x, fitted_y, '--', linewidth=2, alpha=0.8, 
                   label='Fitted Power Law')
            
            # Add R² to legend
            r_squared = learning_curve_data.get('r_squared', 0)
            ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Number of Support Samples (K-shot)', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_title('Few-Shot Learning Performance', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set x-axis ticks
        ax.set_xticks(shot_sizes)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_transferability_heatmap(self, 
                                     transfer_matrix: np.ndarray,
                                     cancer_types: List[str],
                                     save_name: str = 'transferability_heatmap') -> str:
        """Create transferability heatmap."""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.eye(len(cancer_types), dtype=bool)  # Mask diagonal
        sns.heatmap(transfer_matrix, 
                   xticklabels=cancer_types,
                   yticklabels=cancer_types,
                   mask=mask,
                   annot=True, fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0.5,
                   square=True,
                   ax=ax,
                   cbar_kws={'label': 'Transfer Score'})
        
        # Customize plot
        ax.set_title('Cross-Cancer Transfer Learning Performance', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Target Cancer Type', fontsize=14)
        ax.set_ylabel('Source Cancer Type', fontsize=14)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_pathway_importance_plot(self, 
                                     importance_data: Dict,
                                     top_n: int = 20,
                                     save_name: str = 'pathway_importance') -> str:
        """Create pathway importance visualization."""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Integrated Gradients
        if 'integrated_gradients' in importance_data:
            ig_scores = np.array(importance_data['integrated_gradients']['scores'])
            top_indices = np.argsort(ig_scores)[-top_n:][::-1]
            
            axes[0].barh(range(top_n), ig_scores[top_indices])
            axes[0].set_yticks(range(top_n))
            axes[0].set_yticklabels([f'Pathway_{i}' for i in top_indices])
            axes[0].set_xlabel('Importance Score')
            axes[0].set_title('Integrated Gradients')
            axes[0].invert_yaxis()
        
        # Permutation Importance
        if 'permutation' in importance_data:
            perm_scores = np.array(importance_data['permutation']['scores'])
            top_indices = np.argsort(perm_scores)[-top_n:][::-1]
            
            axes[1].barh(range(top_n), perm_scores[top_indices])
            axes[1].set_yticks(range(top_n))
            axes[1].set_yticklabels([f'Pathway_{i}' for i in top_indices])
            axes[1].set_xlabel('Importance Score')
            axes[1].set_title('Permutation Importance')
            axes[1].invert_yaxis()
        
        plt.suptitle('Top Pathway Importance Scores', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        save_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_hierarchical_performance_plot(self, 
                                           hierarchical_results: Dict,
                                           save_name: str = 'hierarchical_performance') -> str:
        """Create hierarchical performance visualization."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Level performance
        if 'level_performance' in hierarchical_results:
            levels = list(hierarchical_results['level_performance'].keys())
            accuracies = [hierarchical_results['level_performance'][level]['accuracy'] 
                         for level in levels]
            
            bars = ax1.bar(levels, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Performance by Hierarchy Level')
            ax1.set_ylim(0, 1)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Hierarchical consistency
        if 'hierarchical_consistency' in hierarchical_results:
            consistency = hierarchical_results['hierarchical_consistency']
            
            consistency_metrics = {
                'Perfect Hierarchy': consistency['perfect_hierarchy_rate'],
                'Organ-Histology': consistency['organ_histology_consistency'],
                'Organ-Molecular': consistency['organ_molecular_consistency'],
                'Histology-Molecular': consistency['histology_molecular_consistency']
            }
            
            metrics = list(consistency_metrics.keys())
            values = list(consistency_metrics.values())
            
            bars = ax2.bar(metrics, values, color=['#FF9F43', '#10AC84', '#5F27CD', '#C44569'])
            ax2.set_ylabel('Consistency Rate')
            ax2.set_title('Hierarchical Consistency')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_comprehensive_dashboard(self, 
                                     all_results: Dict,
                                     save_name: str = 'comprehensive_dashboard') -> str:
        """Create comprehensive results dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Comparison', 'Few-Shot Learning Curve',
                          'Pathway Importance', 'Hierarchical Performance'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Add plots to dashboard
        # This would include interactive Plotly plots for the dashboard
        
        # Save as HTML
        save_path = self.figures_dir / f'{save_name}.html'
        fig.write_html(str(save_path))
        
        return str(save_path)