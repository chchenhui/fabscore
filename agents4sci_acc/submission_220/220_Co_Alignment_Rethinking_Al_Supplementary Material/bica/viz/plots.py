"""
Visualization and Plotting Tools for BiCA
Implements plots mentioned in the paper and additional analysis tools
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BiCAVisualizer:
    """
    Main visualization class for BiCA experiments
    
    Implements all plots mentioned in the paper:
    - Entropy/avg tokens/steps vs epoch
    - BAS radar charts
    - OOD performance bars
    - CCM trajectory plots
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.figure_size = config.get('figure_size', (10, 6))
        self.dpi = config.get('dpi', 300)
        self.save_format = config.get('save_format', 'png')
        
    def plot_training_curves(self, 
                           training_data: Dict[str, List[float]],
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training curves: entropy/avg tokens/steps vs epoch
        
        Args:
            training_data: Dictionary with training metrics over epochs
            save_path: Optional path to save figure
            
        Returns:
            fig: Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        epochs = training_data.get('epochs', range(len(training_data.get('episode_reward_mean', []))))
        
        # Plot 1: Episode rewards
        if 'episode_reward_mean' in training_data:
            axes[0].plot(epochs, training_data['episode_reward_mean'], label='Mean Reward')
            if 'episode_reward_std' in training_data:
                mean_rewards = np.array(training_data['episode_reward_mean'])
                std_rewards = np.array(training_data['episode_reward_std'])
                axes[0].fill_between(epochs, mean_rewards - std_rewards, 
                                   mean_rewards + std_rewards, alpha=0.3)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Episode Reward')
            axes[0].set_title('Training Rewards')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Success rate
        if 'success_rate' in training_data:
            axes[1].plot(epochs, training_data['success_rate'], color='green', label='Success Rate')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Success Rate')
            axes[1].set_title('Success Rate Over Time')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
        
        # Plot 3: Average steps
        if 'episode_length_mean' in training_data:
            axes[2].plot(epochs, training_data['episode_length_mean'], color='orange', label='Avg Steps')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Average Steps')
            axes[2].set_title('Episode Length Over Time')
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: KL divergences
        if 'avg_ai_kl' in training_data and 'avg_human_kl' in training_data:
            axes[3].plot(epochs, training_data['avg_ai_kl'], label='AI KL', color='blue')
            axes[3].plot(epochs, training_data['avg_human_kl'], label='Human KL', color='red')
            
            # Add budget lines
            if 'lambda_a' in training_data and 'lambda_h' in training_data:
                ai_budget = self.config.get('kl_budget_a', 0.05)
                human_budget = self.config.get('kl_budget_h', 0.03)
                axes[3].axhline(y=ai_budget, color='blue', linestyle='--', alpha=0.5, label='AI Budget')
                axes[3].axhline(y=human_budget, color='red', linestyle='--', alpha=0.5, label='Human Budget')
            
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('KL Divergence')
            axes[3].set_title('KL Budgets Over Time')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_bas_radar(self, 
                      bas_scores: Dict[str, float],
                      baseline_scores: Optional[Dict[str, Dict[str, float]]] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot BAS radar chart
        
        Args:
            bas_scores: Dictionary with BAS component scores
            baseline_scores: Optional baseline scores for comparison
            save_path: Optional path to save figure
            
        Returns:
            fig: Matplotlib figure
        """
        # BAS components
        components = ['mp_score', 'bs_score', 'rc_score', 'ss_score', 'ce_score']
        component_names = ['Mutual\nPredictability', 'Bidirectional\nSteerability', 
                          'Representational\nCompatibility', 'Shift-Robust\nSafety', 
                          'Cognitive\nOffloading']
        
        # Extract scores
        bica_values = [bas_scores.get(comp, 0.0) for comp in components]
        
        # Setup radar chart
        angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        bica_values += bica_values[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot BiCA scores
        ax.plot(angles, bica_values, 'o-', linewidth=2, label='BiCA', color='blue')
        ax.fill(angles, bica_values, alpha=0.25, color='blue')
        
        # Plot baselines if provided
        if baseline_scores:
            colors = ['red', 'green', 'orange', 'purple']
            for i, (baseline_name, baseline_data) in enumerate(baseline_scores.items()):
                baseline_values = [baseline_data.get(comp, 0.0) for comp in components]
                baseline_values += baseline_values[:1]
                
                color = colors[i % len(colors)]
                ax.plot(angles, baseline_values, 'o-', linewidth=2, 
                       label=baseline_name, color=color, alpha=0.7)
                ax.fill(angles, baseline_values, alpha=0.15, color=color)
        
        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(component_names)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Title
        total_bas = bas_scores.get('bas_score', sum(bica_values[:-1]) / len(components))
        plt.title(f'BAS Components (Total: {total_bas:.3f})', size=16, pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_ood_performance(self, 
                           ood_results: Dict[str, Dict[str, float]],
                           baseline_results: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot OOD performance bars
        
        Args:
            ood_results: OOD evaluation results
            baseline_results: Optional baseline results for comparison
            save_path: Optional path to save figure
            
        Returns:
            fig: Matplotlib figure
        """
        # Extract OOD variants and metrics
        ood_variants = [k for k in ood_results.keys() if k != 'aggregate']
        metrics = ['success_rate', 'collision_rate', 'avg_steps']
        metric_names = ['Success Rate', 'Collision Rate', 'Avg Steps']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            # BiCA results
            bica_values = [ood_results[variant].get(metric, 0.0) for variant in ood_variants]
            
            x_pos = np.arange(len(ood_variants))
            width = 0.35
            
            axes[i].bar(x_pos - width/2, bica_values, width, label='BiCA', color='blue', alpha=0.7)
            
            # Baseline results if provided
            if baseline_results:
                baseline_name = list(baseline_results.keys())[0]  # Use first baseline
                baseline_data = baseline_results[baseline_name]
                baseline_values = [baseline_data.get(variant, {}).get(metric, 0.0) 
                                 for variant in ood_variants]
                
                axes[i].bar(x_pos + width/2, baseline_values, width, 
                           label=baseline_name, color='red', alpha=0.7)
            
            # Customize subplot
            axes[i].set_xlabel('OOD Variant')
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f'{metric_name} Across OOD Variants')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels([v.replace('_', ' ').title() for v in ood_variants], 
                                   rotation=45, ha='right')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Special handling for collision rate (lower is better)
            if metric == 'collision_rate':
                axes[i].set_ylim(0, max(max(bica_values), 0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_ccm_trajectory(self, 
                          ccm_data: Dict[str, List[float]],
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot CCM trajectory over time
        
        Args:
            ccm_data: CCM scores and components over time
            save_path: Optional path to save figure
            
        Returns:
            fig: Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        epochs = ccm_data.get('epochs', range(len(ccm_data.get('ccm_score', []))))
        
        # Plot 1: CCM score over time
        if 'ccm_score' in ccm_data:
            ax1.plot(epochs, ccm_data['ccm_score'], linewidth=2, color='purple', label='CCM Score')
            
            # Add target range if specified
            target_range = self.config.get('ccm_target_range', [0.3, 0.5])
            ax1.axhspan(target_range[0], target_range[1], alpha=0.2, color='green', 
                       label=f'Target Range [{target_range[0]:.1f}, {target_range[1]:.1f}]')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('CCM Score')
            ax1.set_title('Cognitive Complementarity Metric Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
        
        # Plot 2: CCM components (Diversity and Synergy)
        if 'diversity_score' in ccm_data and 'synergy_score' in ccm_data:
            ax2.plot(epochs, ccm_data['diversity_score'], linewidth=2, color='blue', 
                    label='Diversity', alpha=0.8)
            ax2.plot(epochs, ccm_data['synergy_score'], linewidth=2, color='red', 
                    label='Synergy', alpha=0.8)
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Component Score')
            ax2.set_title('CCM Components: Diversity vs Synergy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_protocol_analysis(self, 
                             protocol_data: Dict[str, Any],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot protocol learning analysis
        
        Args:
            protocol_data: Protocol learning data and statistics
            save_path: Optional path to save figure
            
        Returns:
            fig: Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        epochs = protocol_data.get('epochs', [])
        
        # Plot 1: Gumbel temperature decay
        if 'gumbel_tau' in protocol_data:
            axes[0].plot(epochs, protocol_data['gumbel_tau'], color='orange')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Gumbel Temperature')
            axes[0].set_title('Gumbel Temperature Decay')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_yscale('log')
        
        # Plot 2: Protocol diversity
        if 'protocol_diversity' in protocol_data:
            axes[1].plot(epochs, protocol_data['protocol_diversity'], color='green')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Protocol Diversity')
            axes[1].set_title('Protocol Message Diversity')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
        
        # Plot 3: IB loss components
        if 'ib_loss' in protocol_data and 'kl_from_prior' in protocol_data:
            axes[2].plot(epochs, protocol_data['ib_loss'], label='Total IB Loss', color='blue')
            axes[2].plot(epochs, protocol_data['kl_from_prior'], label='KL from Prior', color='red', alpha=0.7)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Loss')
            axes[2].set_title('Information Bottleneck Components')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Protocol usage histogram
        if 'protocol_usage' in protocol_data:
            usage_counts = protocol_data['protocol_usage']
            protocol_ids = range(len(usage_counts))
            
            axes[3].bar(protocol_ids, usage_counts, color='purple', alpha=0.7)
            axes[3].set_xlabel('Protocol ID')
            axes[3].set_ylabel('Usage Count')
            axes[3].set_title('Protocol Message Usage Distribution')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


class MetricsPlotter:
    """
    Specialized plotter for detailed metrics analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def plot_loss_decomposition(self, 
                               loss_data: Dict[str, List[float]],
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot detailed loss decomposition"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        epochs = loss_data.get('epochs', range(len(list(loss_data.values())[0])))
        
        loss_components = [
            ('ai_total_loss', 'AI Total Loss', 'blue'),
            ('human_loss', 'Human Surrogate Loss', 'red'),
            ('protocol_loss', 'Protocol Loss', 'green'),
            ('rep_repgap_loss', 'RepGap Loss', 'orange'),
            ('instructor_loss', 'Instructor Loss', 'purple'),
            ('intervention_cost', 'Intervention Cost', 'brown')
        ]
        
        for i, (loss_key, title, color) in enumerate(loss_components):
            if loss_key in loss_data and i < len(axes):
                axes[i].plot(epochs, loss_data[loss_key], color=color, linewidth=2)
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Loss')
                axes[i].set_title(title)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(self, 
                               metrics_data: pd.DataFrame,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot correlation matrix of metrics"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Compute correlation matrix
        corr_matrix = metrics_data.corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title('Metrics Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class TrainingVisualizer:
    """
    Real-time training visualization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = {}
        
    def update_metrics(self, new_metrics: Dict[str, float], epoch: int):
        """Update metrics history"""
        if 'epochs' not in self.metrics_history:
            self.metrics_history['epochs'] = []
        
        self.metrics_history['epochs'].append(epoch)
        
        for key, value in new_metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def create_live_dashboard(self):
        """Create live training dashboard using plotly"""
        # This would create an interactive dashboard
        # Implementation depends on specific requirements
        pass


def create_visualizer(config: Dict[str, Any]) -> BiCAVisualizer:
    """Factory function to create visualizer"""
    return BiCAVisualizer(config)


def create_metrics_plotter(config: Dict[str, Any]) -> MetricsPlotter:
    """Factory function to create metrics plotter"""
    return MetricsPlotter(config)


def create_training_visualizer(config: Dict[str, Any]) -> TrainingVisualizer:
    """Factory function to create training visualizer"""
    return TrainingVisualizer(config)
