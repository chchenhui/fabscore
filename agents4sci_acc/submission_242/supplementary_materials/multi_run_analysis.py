#!/usr/bin/env python3
"""
Enhanced Multi-Run Marketplace Analysis

Statistical comparison across configurations with comprehensive visualizations.
Uses integrated analysis framework to avoid code duplication and provides 
publication-ready analysis.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import logging
from dataclasses import dataclass

# Import from integrated analysis framework
from src.analysis.core import (
    MultiRunMetricsExtractor, TimeSeriesExtractor, StatisticalAnalysis,
    ConfigurationParser, DataLoader, SimulationMetrics
)
from src.analysis.visualization import setup_figure, setup_subplots, save_plot
from src.analysis.visualization.market_trend_plots import smooth_line

import warnings
warnings.filterwarnings('ignore')


# Default color palette for configurations
DEFAULT_COLORS = [
    '#2E86AB',  # Blue
    '#87CEEB',  # Light Blue  
    '#A23B72',  # Magenta
    '#F18F01',  # Orange
    '#C73E1D',  # Red
    '#4CAF50',  # Green
    '#9C27B0',  # Purple
    '#FF6B6B',  # Coral
    '#4ECDC4',  # Teal
    '#45B7D1',  # Sky Blue
]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AggregatedResults:
    """Container for aggregated results across multiple runs"""
    config_name: str
    n_runs: int
    metrics_mean: SimulationMetrics
    metrics_std: SimulationMetrics
    metrics_ci_lower: SimulationMetrics
    metrics_ci_upper: SimulationMetrics
    raw_runs: List[Dict]
    time_series_data: List[pd.DataFrame]  # Time series from each run


class EnhancedMultiRunAnalyzer:
    """Enhanced analyzer for multiple marketplace simulation runs with comprehensive visualizations"""
    
    def __init__(self, results_dir: str = "results/simuleval"):
        self.results_dir = Path(results_dir)
        self.configurations = defaultdict(list)
        self.aggregated_results = {}
        self.varying_params = set()  # Track which parameters actually vary
        self.synthetic_names = {}  # Map config names to synthetic labels
        self.config_colors = {}  # Dynamic color mapping
    
    def _generate_synthetic_names(self, config_names: List[str]) -> Dict[str, str]:
        """Generate synthetic names for configurations based on agent types and reflection status"""
        synthetic_names = {}
        
        for config_name in config_names:
            # Extract agent types from config name
            parts = config_name.split('_')
            if len(parts) >= 2:
                freelancer_type = parts[0].upper()
                client_type = parts[1].upper()
                
                # Check for reflection status in config name
                if 'refltrue' in config_name.lower() or 'reflection_true' in config_name.lower():
                    synthetic_names[config_name] = f"{freelancer_type}-{client_type} (w/ Reflections)"
                elif 'reflfalse' in config_name.lower() or 'reflection_false' in config_name.lower():
                    synthetic_names[config_name] = f"{freelancer_type}-{client_type} (w/o Reflections)"
                else:
                    # For LLM-LLM configurations, we need to check the actual data to determine reflection status
                    if freelancer_type == 'LLM' and client_type == 'LLM':
                        # We'll update this in discover_configurations after loading the data
                        synthetic_names[config_name] = "LLM-F + LLM-C"
                    else:
                        freelancer_label = "LLM-F" if freelancer_type == 'LLM' else "Rand-F"
                        client_label = "LLM-C" if client_type == 'LLM' else "Rand-C"
                        synthetic_names[config_name] = f"{freelancer_label} + {client_label}"
            else:
                # Fallback if format is unexpected
                synthetic_names[config_name] = config_name.replace('_', '-').upper()
            
        return synthetic_names
    
    def _generate_synthetic_names_from_data(self, config_groups: Dict[str, List[str]]) -> Dict[str, str]:
        """Generate synthetic names based on actual configuration data"""
        synthetic_names = {}
        
        for config_key in config_groups.keys():
            synthetic_name = self._create_readable_name_from_key(config_key)
            synthetic_names[config_key] = synthetic_name
        
        return synthetic_names
    
    def _create_readable_name_from_key(self, config_key: str) -> str:
        """Create a readable name from configuration key based on agent types"""
        # Handle reflection variants first
        if 'reflections_true' in config_key:
            base_config = config_key.replace('_reflections_true', '')
            base_name = self._get_base_agent_name(base_config)
            return f"{base_name} (w/ Refl.)"
        elif 'reflections_false' in config_key:
            base_config = config_key.replace('_reflections_false', '')
            base_name = self._get_base_agent_name(base_config)
            return f"{base_name} (w/o Refl.)"
        else:
            return self._get_base_agent_name(config_key)
    
    def _get_base_agent_name(self, config_key: str) -> str:
        """Get base agent name from configuration key"""
        parts = config_key.split('_')
        if len(parts) >= 2:
            freelancer_type = self._format_agent_type(parts[0], 'F')
            client_type = self._format_agent_type(parts[1], 'C')
            return f"{freelancer_type} + {client_type}"
        else:
            # Fallback for unexpected format
            return config_key.replace('_', '-').upper()
    
    def _format_agent_type(self, agent_type: str, role: str) -> str:
        """Format agent type with role indicator"""
        if agent_type.upper() == 'LLM':
            return f"LLM-{role}"
        elif agent_type.upper() == 'RANDOM':
            return f"Rand-{role}"
        elif agent_type.upper() == 'GREEDY':
            return f"Greedy-{role}"
        else:
            # Handle any other agent types dynamically
            return f"{agent_type.capitalize()}-{role}"
    
    def _assign_colors_to_configs(self, config_keys: List[str]) -> Dict[str, str]:
        """Assign colors to configurations dynamically"""
        colors = {}
        sorted_configs = self._sort_configurations(config_keys)
        
        for i, config_key in enumerate(sorted_configs):
            color_index = i % len(DEFAULT_COLORS)
            colors[config_key] = DEFAULT_COLORS[color_index]
        
        return colors
    
    def _sort_configurations(self, config_names: List[str]) -> List[str]:
        """Sort configurations by agent type priority and reflection status"""
        def sort_key(config_name):
            # Parse the configuration to get agent types and reflection status
            has_reflections = 'reflections_true' in config_name
            no_reflections = 'reflections_false' in config_name
            
            # Remove reflection suffix for type parsing
            base_config = config_name.replace('_reflections_true', '').replace('_reflections_false', '')
            parts = base_config.split('_')
            
            if len(parts) >= 2:
                freelancer_type = parts[0].upper()
                client_type = parts[1].upper()
                
                # Priority order: LLM-LLM, LLM-Other, Other-LLM, Other-Other
                if freelancer_type == 'LLM' and client_type == 'LLM':
                    if has_reflections:
                        return (0, config_name)  # LLM-LLM with reflections first
                    elif no_reflections:
                        return (1, config_name)  # LLM-LLM without reflections second
                    else:
                        return (2, config_name)  # Other LLM-LLM configurations
                elif freelancer_type == 'LLM':
                    return (3, config_name)  # LLM freelancer configurations
                elif client_type == 'LLM':
                    return (4, config_name)  # LLM client configurations
                else:
                    return (5, config_name)  # Non-LLM configurations
            else:
                return (6, config_name)  # Fallback for unexpected format
                
        return sorted(config_names, key=sort_key)
    
    def _align_time_series_efficiently(self, time_series_data: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Efficiently align time series data with minimal memory usage"""
        if not time_series_data:
            return []
        
        # Find max rounds more efficiently
        max_rounds = max(len(ts) for ts in time_series_data)
        aligned_data = []
        
        for ts in time_series_data:
            if len(ts) == max_rounds:
                # No alignment needed
                aligned_data.append(ts.copy())
            elif len(ts) > 0:
                # Pad efficiently using reindex
                full_index = range(1, max_rounds + 1)
                ts_reindexed = ts.set_index('round').reindex(full_index, method='ffill')
                ts_reindexed['round'] = full_index
                ts_reindexed = ts_reindexed.reset_index(drop=True)
                aligned_data.append(ts_reindexed)
        
        return aligned_data
        
    def discover_configurations(self) -> Dict[str, List[str]]:
        """Discover and group simulation files by configuration, separating by reflection status"""
        logger.info("🔍 Discovering simulation configurations...")
        
        config_groups = defaultdict(list)
        simulation_files = DataLoader.discover_simulation_files(self.results_dir)
        
        for file_path in simulation_files:
            # Load simulation data to check reflection status for LLM configurations
            try:
                simulation_data = DataLoader.load_simulation_data(file_path)
                config = simulation_data.get('simulation_config', {})
                
                freelancer_type = config.get('freelancer_agent_type', 'unknown')
                client_type = config.get('client_agent_type', 'unknown')
                enable_reflections = config.get('enable_reflections', False)
                
                # Create configuration key that includes reflection status for LLM agents
                if freelancer_type == 'llm' and client_type == 'llm':
                    if enable_reflections:
                        config_key = "llm_llm_reflections_true"
                    else:
                        config_key = "llm_llm_reflections_false"
                else:
                    # For non-LLM configurations, use standard key
                    config_key = f"{freelancer_type}_{client_type}"
                
                config_groups[config_key].append(file_path)
                
                # Clear simulation_data from memory
                del simulation_data
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path} for configuration detection: {e}")
                # Fallback to filename-based detection
                config_key = ConfigurationParser.extract_config_key(file_path)
                config_groups[config_key].append(file_path)
        
        # Filter to configurations with multiple runs
        filtered_configs = {k: v for k, v in config_groups.items() if len(v) >= 2}
        
        # Update synthetic names and assign colors based on discovered configurations
        self.synthetic_names = self._generate_synthetic_names_from_data(filtered_configs)
        self.config_colors = self._assign_colors_to_configs(list(filtered_configs.keys()))
        
        logger.info("📊 Discovered configurations:")
        for config, files in filtered_configs.items():
            synthetic_name = self.synthetic_names.get(config, config)
            logger.info(f"  • {synthetic_name}: {len(files)} runs")
        
        return filtered_configs
    
    def aggregate_configuration_results(self, config_name: str, run_files: List[str]) -> AggregatedResults:
        """Aggregate results across multiple runs of the same configuration"""
        readable_name = ConfigurationParser.get_readable_config_name(config_name, self.varying_params)
        logger.info(f"📈 Aggregating results for {readable_name} ({len(run_files)} files)...")
        
        # Extract metrics and time series from all runs
        run_metrics = []
        time_series_data = []
        
        for i, file_path in enumerate(run_files):
            try:
                logger.info(f"  Processing file {i+1}/{len(run_files)}: {Path(file_path).name}")
                simulation_data = DataLoader.load_simulation_data(file_path)
                
                # Extract basic metrics
                metrics = MultiRunMetricsExtractor.extract_basic_metrics(simulation_data)
                run_metrics.append(metrics.__dict__)
                
                # Extract time series data
                time_series = TimeSeriesExtractor.extract_round_metrics(simulation_data)
                time_series_data.append(time_series)
                
                # Clear simulation_data from memory immediately
                del simulation_data
                
            except Exception as e:
                logger.exception(f"Failed to process {file_path}: {e}")
                continue
        
        if not run_metrics:
            logger.error(f"❌ No valid metrics found for {config_name}")
            return None
        
        # Calculate statistical aggregates for each metric
        metrics_df = pd.DataFrame(run_metrics)
        
        means = {}
        stds = {}
        ci_lowers = {}
        ci_uppers = {}
        
        for col in metrics_df.columns:
            values = metrics_df[col].values
            mean_val, ci_lower, ci_upper = StatisticalAnalysis.calculate_confidence_interval(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0
            
            means[col] = mean_val
            stds[col] = std_val
            ci_lowers[col] = ci_lower
            ci_uppers[col] = ci_upper
        
        return AggregatedResults(
            config_name=config_name,
            n_runs=len(run_metrics),
            metrics_mean=SimulationMetrics(**means),
            metrics_std=SimulationMetrics(**stds),
            metrics_ci_lower=SimulationMetrics(**ci_lowers),
            metrics_ci_upper=SimulationMetrics(**ci_uppers),
            raw_runs=run_metrics,
            time_series_data=time_series_data
        )
    
    def run_complete_analysis(self) -> Dict[str, AggregatedResults]:
        """Run complete multi-configuration analysis"""
        logger.info("🚀 Starting enhanced multi-run analysis...")
        
        # Discover configurations
        configurations = self.discover_configurations()
        
        if not configurations:
            logger.error("❌ No configurations with multiple runs found!")
            return {}
        
        # Synthetic names are already set in discover_configurations()
        # No need to regenerate them here
        
        # Aggregate results for each configuration
        results = {}
        for config_name, run_files in configurations.items():
            aggregated = self.aggregate_configuration_results(config_name, run_files)
            if aggregated:
                results[config_name] = aggregated
        
        logger.info(f"✅ Analysis complete for {len(results)} configurations")
        return results
    
    def create_comprehensive_visualizations(
        self, 
        results: Dict[str, AggregatedResults], 
        output_dir: str = "analysis_results"
    ):
        """Create comprehensive comparison visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"📊 Creating comprehensive visualizations in {output_path}")
        total_steps = 5
        
        # 1. Performance comparison with error bars
        logger.info(f"[1/{total_steps}] Creating performance comparison plot...")
        self._create_performance_comparison(results, output_path / "performance_comparison.png")
        
        # 2. Detailed metrics grid
        logger.info(f"[2/{total_steps}] Creating detailed metrics grid...")
        self._create_detailed_metrics_grid(results, output_path / "detailed_metrics_grid.png")
        
        # 3. **NEW**: Time series trend plots for each configuration
        logger.info(f"[3/{total_steps}] Creating individual trend plots...")
        self._create_trend_plots(results, output_path / "trend_analysis")
        
        # 4. **NEW**: Comparative time series overlay with variance bands
        logger.info(f"[4/{total_steps}] Creating comparative trend overlays...")
        self._create_comparative_trends(results, output_path / "comparative_trends.png")
        
        # 5. **NEW**: Market evolution heatmap
        logger.info(f"[5/{total_steps}] Creating market evolution heatmap...")
        self._create_market_evolution_heatmap(results, output_path / "market_evolution_heatmap.png")
        
        logger.info("✅ All visualizations created successfully")
    
    def _create_performance_comparison(self, results: Dict[str, AggregatedResults], output_path: Path):
        """Create main performance comparison plot with error bars"""
        fig, axes = setup_subplots(2, 2, (15, 8))
        axes = axes.flatten()  # Ensure axes is 1D array for easy indexing
        
        config_names = self._sort_configurations(list(results.keys()))
        synthetic_names = [self.synthetic_names.get(name, f"Config {i+1}") for i, name in enumerate(config_names)]
        x_pos = np.arange(len(config_names))
        
        colors = [self.config_colors.get(name, '#1f77b4') for name in config_names]
        
        # Fill Rate
        fill_rates = [results[name].metrics_mean.fill_rate for name in config_names]
        fill_rate_errors = [
            (results[name].metrics_mean.fill_rate - results[name].metrics_ci_lower.fill_rate,
             results[name].metrics_ci_upper.fill_rate - results[name].metrics_mean.fill_rate) 
            for name in config_names
        ]
        
        axes[0].bar(x_pos, fill_rates, yerr=np.array(fill_rate_errors).T, 
                   capsize=5, alpha=0.8, color=colors)
        axes[0].set_title('Fill Rate', fontweight='bold')
        axes[0].set_ylabel('Fill Rate')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(synthetic_names, rotation=15, ha='right')
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Bid Efficiency
        efficiencies = [results[name].metrics_mean.bid_efficiency for name in config_names]
        efficiency_errors = [
            (results[name].metrics_mean.bid_efficiency - results[name].metrics_ci_lower.bid_efficiency,
             results[name].metrics_ci_upper.bid_efficiency - results[name].metrics_mean.bid_efficiency)
            for name in config_names
        ]
        
        axes[1].bar(x_pos, efficiencies, yerr=np.array(efficiency_errors).T, 
                   capsize=5, alpha=0.8, color=colors)
        axes[1].set_title('Bid Efficiency', fontweight='bold')
        axes[1].set_ylabel('Bid Efficiency')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(synthetic_names, rotation=15, ha='right')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Participation Rate
        participation = [results[name].metrics_mean.participation_rate for name in config_names]
        participation_errors = [
            (results[name].metrics_mean.participation_rate - results[name].metrics_ci_lower.participation_rate,
             results[name].metrics_ci_upper.participation_rate - results[name].metrics_mean.participation_rate)
            for name in config_names
        ]
        
        axes[2].bar(x_pos, participation, yerr=np.array(participation_errors).T, 
                   capsize=5, alpha=0.8, color=colors)
        axes[2].set_title('Participation Rate', fontweight='bold')
        axes[2].set_ylabel('Participation Rate')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(synthetic_names, rotation=15, ha='right')
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Market Health
        health = [results[name].metrics_mean.market_health_score for name in config_names]
        health_errors = [
            (results[name].metrics_mean.market_health_score - results[name].metrics_ci_lower.market_health_score,
             results[name].metrics_ci_upper.market_health_score - results[name].metrics_mean.market_health_score)
            for name in config_names
        ]
        
        axes[3].bar(x_pos, health, yerr=np.array(health_errors).T, 
                   capsize=5, alpha=0.8, color=colors)
        axes[3].set_title('Market Health Score', fontweight='bold')
        axes[3].set_ylabel('Health Score')
        axes[3].set_xticks(x_pos)
        axes[3].set_xticklabels(synthetic_names, rotation=15, ha='right')
        
        save_plot(
            fig, output_path, 
            'Marketplace Performance Comparison\n(Error bars show 95% confidence intervals)'
        )
    
    def _create_detailed_metrics_grid(self, results: Dict[str, AggregatedResults], output_path: Path):
        """Create detailed metrics grid with error bars"""
        fig, axes = setup_subplots(2, 4, (20, 10))
        axes = axes.flatten()  # Ensure axes is 1D array for easy indexing
        
        config_names = self._sort_configurations(list(results.keys()))
        synthetic_names = [self.synthetic_names.get(name, f"Config {i+1}") for i, name in enumerate(config_names)]
        x_pos = np.arange(len(config_names))
        
        metrics = [
            ('Fill Rate', 'fill_rate', lambda y, _: '{:.0%}'.format(y)),
            ('Bid Efficiency', 'bid_efficiency', lambda y, _: '{:.0%}'.format(y)),
            ('Bids per Job', 'avg_bids_per_job', lambda y, _: '{:.1f}'.format(y)),
            ('Participation Rate', 'participation_rate', lambda y, _: '{:.0%}'.format(y)),
            ('Rejection Rate', 'rejection_rate', lambda y, _: '{:.0%}'.format(y)),
            ('Freelancer Hiring Rate', 'freelancer_hiring_rate', lambda y, _: '{:.0%}'.format(y)),
            ('Gini Coefficient', 'gini_coefficient', lambda y, _: '{:.2f}'.format(y)),
            ('Market Health', 'market_health_score', lambda y, _: '{:.1f}'.format(y))
        ]
        
        for idx, (title, metric_key, formatter) in enumerate(metrics):
            ax = axes[idx]
            
            values = [getattr(results[name].metrics_mean, metric_key) for name in config_names]
            errors = [getattr(results[name].metrics_std, metric_key) for name in config_names]
            
            colors = [self.config_colors.get(name, '#1f77b4') for name in config_names]
            
            bars = ax.bar(x_pos, values, yerr=errors, capsize=4, alpha=0.8, color=colors)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(synthetic_names, rotation=45, ha='right', fontsize=10)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter))
            
            # Add value labels on bars
            for bar, value, error in zip(bars, values, errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + error,
                       formatter(value, None), ha='center', va='bottom', fontsize=8)
        
        save_plot(
            fig, output_path, 
            'Detailed Metrics Comparison\n(Error bars show ±1 standard deviation)'
        )
    
    def _create_significance_heatmap(self, results: Dict[str, AggregatedResults], output_path: Path):
        """Create statistical significance heatmap"""
        config_names = self._sort_configurations(list(results.keys()))
        synthetic_names = [self.synthetic_names.get(name, f"Config {i+1}") for i, name in enumerate(config_names)]
        n_configs = len(config_names)
        
        # Create p-value matrix for key metrics
        metrics_to_test = ['fill_rate', 'bid_efficiency', 'participation_rate']
        
        fig, axes = setup_subplots(1, len(metrics_to_test), 
                                               (5 * len(metrics_to_test), 5))
        
        for metric_idx, metric_key in enumerate(metrics_to_test):
            p_matrix = np.ones((n_configs, n_configs))  # Initialize with 1s (no significance)
            
            for i, config1 in enumerate(config_names):
                for j, config2 in enumerate(config_names):
                    if i != j:
                        values1 = [getattr(SimulationMetrics(**run), metric_key) for run in results[config1].raw_runs]
                        values2 = [getattr(SimulationMetrics(**run), metric_key) for run in results[config2].raw_runs]
                        
                        if len(values1) >= 2 and len(values2) >= 2:
                            _, p_value = StatisticalAnalysis.perform_t_test(values1, values2)
                            p_matrix[i, j] = p_value
            
            # Create heatmap
            ax = axes[metric_idx] if len(metrics_to_test) > 1 else axes
            sns.heatmap(p_matrix, 
                       xticklabels=synthetic_names, 
                       yticklabels=synthetic_names,
                       annot=True, 
                       fmt='.3f',
                       cmap='RdYlBu_r',
                       center=0.05,
                       ax=ax,
                       cbar_kws={'label': 'p-value'})
            
            ax.set_title(f'{metric_key.replace("_", " ").title()}\nStatistical Significance')
        
        save_plot(fig, output_path, 'Statistical Significance Tests (p-values)')
    
    def _create_variance_analysis(self, results: Dict[str, AggregatedResults], output_path: Path):
        """Create variance analysis plot"""
        fig, axes = setup_subplots(1, 2, (15, 6))
        # axes is already flattened for 1xN layouts
        
        config_names = self._sort_configurations(list(results.keys()))
        synthetic_names = [self.synthetic_names.get(name, f"Config {i+1}") for i, name in enumerate(config_names)]
        
        # Coefficient of variation for key metrics
        metrics = ['fill_rate', 'bid_efficiency', 'participation_rate', 'rejection_rate']
        metric_labels = ['Fill Rate', 'Bid Efficiency', 'Participation Rate', 'Rejection Rate']
        
        cv_data = []
        for config_name in config_names:
            cv_row = []
            for metric in metrics:
                mean_val = getattr(results[config_name].metrics_mean, metric)
                std_val = getattr(results[config_name].metrics_std, metric)
                cv = std_val / mean_val if mean_val > 0 else 0
                cv_row.append(cv)
            cv_data.append(cv_row)
        
        # Coefficient of variation heatmap
        cv_df = pd.DataFrame(cv_data, index=synthetic_names, columns=metric_labels)
        sns.heatmap(cv_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0])
        axes[0].set_title('Coefficient of Variation\n(Lower = More Consistent)')
        
        # Run count and confidence interval width
        n_runs = [results[name].n_runs for name in config_names]
        colors = [self.config_colors.get(name, '#1f77b4') for name in config_names]
        
        axes[1].bar(synthetic_names, n_runs, alpha=0.8, color=colors)
        axes[1].set_title('Number of Runs per Configuration')
        axes[1].set_ylabel('Number of Runs')
        axes[1].tick_params(axis='x', rotation=45)
        
        save_plot(fig, output_path, 'Variance Analysis')
    
    def _create_trend_plots(self, results: Dict[str, AggregatedResults], output_dir: Path):
        """Create time series trend plots for each configuration"""
        output_dir.mkdir(exist_ok=True)
        
        sorted_config_names = self._sort_configurations(list(results.keys()))
        logger.info(f"📊 Creating individual trend plots for {len(sorted_config_names)} configurations...")
        
        for i, config_name in enumerate(sorted_config_names):
            result = results[config_name]
            synthetic_name = self.synthetic_names.get(config_name, f"Config {config_name}")
            logger.info(f"  Creating trends for {synthetic_name} ({i+1}/{len(sorted_config_names)})...")
            
            # Average time series across runs
            if not result.time_series_data:
                logger.warning(f"  No time series data for {synthetic_name}, skipping...")
                continue
            
            # Use more efficient time series alignment
            aligned_data = self._align_time_series_efficiently(result.time_series_data)
            if not aligned_data:
                logger.warning(f"  Failed to align time series for {synthetic_name}, skipping...")
                continue
            
            # Calculate mean and std across runs for each round
            combined_df = pd.concat(aligned_data, keys=range(len(aligned_data)))
            mean_df = combined_df.groupby(level=1).mean()
            std_df = combined_df.groupby(level=1).std().fillna(0)
            
            # Create trend plots
            fig, axes = setup_subplots(2, 3, (18, 12))
            axes = axes.flatten()  # Ensure axes is 1D array for easy indexing
            
            rounds = mean_df['round'].values
            
            # Fill rate trend
            self._plot_trend_with_confidence(
                axes[0], rounds, 
                mean_df['jobs_filled_cumulative'] / mean_df['jobs_posted'].cumsum(),
                std_df['jobs_filled_cumulative'] / mean_df['jobs_posted'].cumsum(),
                'Fill Rate Over Time', 'Fill Rate', format_pct=True
            )
            
            # Participation rate trend
            self._plot_trend_with_confidence(
                axes[1], rounds, 
                mean_df['participation_rate'], std_df['participation_rate'],
                'Participation Rate Over Time', 'Participation Rate', format_pct=True
            )
            
            # Bid rejection rate trend  
            self._plot_trend_with_confidence(
                axes[2], rounds,
                mean_df['bid_rejection_rate'], std_df['bid_rejection_rate'],
                'Bid Rejection Rate Over Time', 'Rejection Rate', format_pct=True
            )
            
            # Market health trend
            self._plot_trend_with_confidence(
                axes[3], rounds,
                mean_df['health_score'], std_df['health_score'],
                'Market Health Over Time', 'Health Score'
            )
            
            # Bids per job trend
            self._plot_trend_with_confidence(
                axes[4], rounds,
                mean_df['avg_bids_per_job'], std_df['avg_bids_per_job'],
                'Competition Level Over Time', 'Avg Bids per Job'
            )
            
            # Supply-demand ratio trend
            self._plot_trend_with_confidence(
                axes[5], rounds,
                mean_df['supply_demand_ratio'], std_df['supply_demand_ratio'],
                'Supply-Demand Balance Over Time', 'Supply/Demand Ratio'
            )
            
            save_plot(
                fig, output_dir / f"trends_{config_name}.png", 
                f'Market Trends: {synthetic_name}\n(n={result.n_runs} runs, shaded areas show ±1 std)'
            )
    
    def _plot_trend_with_confidence(self, ax, x, y_mean, y_std, title, ylabel, format_pct=False):
        """Plot trend line with confidence band"""
        # Smooth the lines
        x_smooth, y_smooth = smooth_line(x, y_mean)
        
        # Plot mean line
        ax.plot(x_smooth, y_smooth, linewidth=2, alpha=0.8)
        
        # Plot confidence band
        y_upper = y_mean + y_std
        y_lower = y_mean - y_std
        
        # Clamp percentage values to valid range [0, 1]
        if format_pct:
            y_upper = np.clip(y_upper, 0, 1)
            y_lower = np.clip(y_lower, 0, 1)
        
        ax.fill_between(x, y_upper, y_lower, alpha=0.3)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Round')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        if format_pct:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    def _create_comparative_trends(self, results: Dict[str, AggregatedResults], output_path: Path):
        """Create overlay of trends across all configurations with variance bands"""
        logger.info("📊 Creating comparative trend overlays...")
        
        fig, axes = setup_subplots(2, 2, (16, 12))
        axes = axes.flatten()  # Ensure axes is 1D array for easy indexing
        
        metrics_to_plot = [
            ('fill_rate', 'Fill Rate Over Time', True),
            ('participation_rate', 'Participation Rate Over Time', True),
            ('bid_rejection_rate', 'Bid Rejection Rate Over Time', True),
            ('health_score', 'Market Health Over Time', False)
        ]
        
        for metric_idx, (metric_key, title, format_pct) in enumerate(metrics_to_plot):
            ax = axes[metric_idx]
            logger.info(f"  Processing metric: {title}")
            
            sorted_config_names = self._sort_configurations(list(results.keys()))
            for config_name in sorted_config_names:
                result = results[config_name]
                if not result.time_series_data:
                    continue
                
                synthetic_name = self.synthetic_names.get(config_name, f"Config {config_name}")
                color = self.config_colors.get(config_name, '#1f77b4')
                
                # Use efficient time series alignment
                aligned_data = self._align_time_series_efficiently(result.time_series_data)
                if not aligned_data:
                    continue
                
                # Calculate mean and std across runs
                combined_df = pd.concat(aligned_data, keys=range(len(aligned_data)))
                mean_df = combined_df.groupby(level=1).mean()
                std_df = combined_df.groupby(level=1).std().fillna(0)
                
                rounds = mean_df['round'].values
                
                if metric_key == 'fill_rate':
                    y_mean = mean_df['jobs_filled_cumulative'] / mean_df['jobs_posted'].cumsum()
                    y_std = std_df['jobs_filled_cumulative'] / mean_df['jobs_posted'].cumsum()
                else:
                    y_mean = mean_df[metric_key].values
                    y_std = std_df[metric_key].values
                
                # Smooth the mean line
                x_smooth, y_smooth = smooth_line(rounds, y_mean)
                
                # Plot mean line
                ax.plot(x_smooth, y_smooth, label=synthetic_name, linewidth=2, 
                       alpha=0.8, color=color)
                
                # Add confidence band (±1 std)
                y_upper = y_mean + y_std
                y_lower = y_mean - y_std
                
                # Clamp percentage values to valid range [0, 1]
                if format_pct:
                    y_upper = np.clip(y_upper, 0, 1)
                    y_lower = np.clip(y_lower, 0, 1)
                
                ax.fill_between(rounds, y_upper, y_lower, alpha=0.2, color=color)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Round')
            ax.set_ylabel(metric_key.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if format_pct:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        save_plot(fig, output_path, 'Comparative Market Trends Across Configurations\n(Shaded areas show ±1 standard deviation)')
    
    def _create_market_evolution_heatmap(self, results: Dict[str, AggregatedResults], output_path: Path):
        """Create heatmap showing market evolution across configurations"""
        logger.info("📊 Processing market evolution data...")
        
        config_names = self._sort_configurations(list(results.keys()))
        synthetic_names = [self.synthetic_names.get(name, f"Config {i+1}") for i, name in enumerate(config_names)]
        
        # Create evolution matrix (early vs late rounds)
        evolution_data = []
        
        for i, config_name in enumerate(config_names):
            logger.info(f"  Processing evolution for {synthetic_names[i]} ({i+1}/{len(config_names)})...")
            result = results[config_name]
            if not result.time_series_data:
                evolution_data.append([0, 0, 0, 0])  # Placeholder
                continue
            
            # Calculate early vs late metrics
            combined_df = pd.concat(result.time_series_data, keys=range(len(result.time_series_data)))
            mean_df = combined_df.groupby(level=1).mean()
            
            n_rounds = len(mean_df)
            
            # Handle edge cases where DataFrame is too small
            if n_rounds < 6:  # Need at least 6 rounds to split meaningfully
                evolution_data.append([0, 0, 0, 0])  # No meaningful change data
                continue
                
            early_rounds = mean_df.iloc[:n_rounds//3]  # First third
            late_rounds = mean_df.iloc[-n_rounds//3:]  # Last third
            
            # Calculate evolution metrics with bounds checking
            try:
                if len(early_rounds) > 0 and len(late_rounds) > 0:
                    # Calculate fill rate change safely
                    early_fill_rate = (
                        early_rounds['jobs_filled_cumulative'].iloc[-1] / max(early_rounds['jobs_posted'].sum(), 1)
                        if len(early_rounds) > 0 else 0
                    )
                    late_fill_rate = (
                        late_rounds['jobs_filled_cumulative'].iloc[-1] / max(late_rounds['jobs_posted'].sum(), 1)
                        if len(late_rounds) > 0 else 0
                    )
                    fill_rate_change = late_fill_rate - early_fill_rate
                    
                    participation_change = (
                        late_rounds['participation_rate'].mean() - 
                        early_rounds['participation_rate'].mean()
                    )
                    
                    health_change = (
                        late_rounds['health_score'].mean() - 
                        early_rounds['health_score'].mean()
                    )
                    
                    competition_change = (
                        late_rounds['avg_bids_per_job'].mean() - 
                        early_rounds['avg_bids_per_job'].mean()
                    )
                else:
                    fill_rate_change = participation_change = health_change = competition_change = 0
                    
            except (IndexError, KeyError, ZeroDivisionError):
                # Handle any calculation errors gracefully
                fill_rate_change = participation_change = health_change = competition_change = 0
            
            evolution_data.append([fill_rate_change, participation_change, 
                                 health_change, competition_change])
        
        # Create heatmap
        fig, ax = setup_figure((12, 8))
        
        evolution_df = pd.DataFrame(
            evolution_data,
            index=synthetic_names,
            columns=['Fill Rate Δ', 'Participation Δ', 'Health Δ', 'Competition Δ']
        )
        
        sns.heatmap(evolution_df, annot=True, fmt='.3f', cmap='RdBu_r', 
                   center=0, ax=ax, cbar_kws={'label': 'Change (Late - Early)'})
        
        save_plot(fig, output_path, 'Market Evolution Heatmap\n(Late rounds - Early rounds)')
    
    def _create_radar_chart(self, results: Dict[str, AggregatedResults], output_path: Path):
        """Create radar chart comparing configuration performance"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Metrics for radar chart (normalized to 0-1 scale)
        metrics = [
            'fill_rate',
            'bid_efficiency', 
            'participation_rate',
            'freelancer_hiring_rate',
            'market_health_score'
        ]
        
        metric_labels = [
            'Fill Rate',
            'Bid Efficiency',
            'Participation',
            'Hiring Rate', 
            'Market Health'
        ]
        
        # Calculate angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        # Plot each configuration
        config_names = self._sort_configurations(list(results.keys()))
        for config_name in config_names:
            result = results[config_name]
            synthetic_name = self.synthetic_names.get(config_name, f"Config {config_name}")
            color = self.config_colors.get(config_name, '#1f77b4')
            
            # Normalize metrics to 0-1 scale
            values = []
            for metric in metrics:
                raw_value = getattr(result.metrics_mean, metric)
                
                # Normalize based on metric type
                if metric == 'market_health_score':
                    normalized_value = raw_value  # Already 0-1 scale
                else:
                    # For rates, already 0-1, for others normalize by max across all configs
                    if metric in ['fill_rate', 'bid_efficiency', 'participation_rate', 'freelancer_hiring_rate']:
                        normalized_value = raw_value  # Already proportions
                    else:
                        max_val = max(getattr(results[cn].metrics_mean, metric) for cn in config_names)
                        normalized_value = raw_value / max_val if max_val > 0 else 0
                
                values.append(normalized_value)
            
            values += values[:1]  # Close the circle
            
            # Plot the configuration
            ax.plot(angles, values, 'o-', linewidth=2, label=synthetic_name, 
                   color=color, alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Customize radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        save_plot(fig, output_path, 'Configuration Performance Radar Chart')
    
    def generate_comprehensive_report(self, results: Dict[str, AggregatedResults]) -> str:
        """Generate comprehensive text report"""
        report = []
        report.append("=" * 100)
        report.append("ENHANCED MULTI-RUN MARKETPLACE ANALYSIS REPORT")
        report.append("=" * 100)
        
        # Configuration overview
        report.append("\n📊 CONFIGURATION OVERVIEW")
        report.append("-" * 80)
        
        sorted_config_names = self._sort_configurations(list(results.keys()))
        for config_name in sorted_config_names:
            result = results[config_name]
            synthetic_name = self.synthetic_names.get(config_name, config_name)
            report.append(f"{synthetic_name:<50} {result.n_runs} runs")
        
        # Performance comparison with confidence intervals
        report.append("\n🎯 PERFORMANCE COMPARISON (Mean ± 95% CI)")
        report.append("-" * 100)
        
        # Create comparison table
        header = f"{'Metric':<25}"
        for config_name in sorted_config_names:
            synthetic_name = self.synthetic_names.get(config_name, config_name)
            header += f"{synthetic_name[:25]:<27}"
        report.append(header)
        report.append("-" * (25 + 27 * len(results)))
        
        metrics_to_show = [
            ('Fill Rate', 'fill_rate', lambda x: f"{x:.1%}"),
            ('Bid Efficiency', 'bid_efficiency', lambda x: f"{x:.1%}"),
            ('Bids per Job', 'avg_bids_per_job', lambda x: f"{x:.2f}"),
            ('Participation Rate', 'participation_rate', lambda x: f"{x:.1%}"),
            ('Rejection Rate', 'rejection_rate', lambda x: f"{x:.1%}"),
            ('Freelancer Hiring Rate', 'freelancer_hiring_rate', lambda x: f"{x:.1%}"),
            ('Gini Coefficient', 'gini_coefficient', lambda x: f"{x:.3f}"),
            ('Market Health', 'market_health_score', lambda x: f"{x:.2f}")
        ]
        
        for metric_name, metric_key, formatter in metrics_to_show:
            row = f"{metric_name:<25}"
            for config_name in sorted_config_names:
                result = results[config_name]
                mean_val = getattr(result.metrics_mean, metric_key)
                ci_lower = getattr(result.metrics_ci_lower, metric_key)
                ci_upper = getattr(result.metrics_ci_upper, metric_key)
                
                formatted_mean = formatter(mean_val)
                if result.n_runs > 1:
                    ci_range = f"[{formatter(ci_lower)}-{formatter(ci_upper)}]"
                    row += f"{formatted_mean} {ci_range}"[:26] + " "
                else:
                    row += f"{formatted_mean:<26} "
            report.append(row)
        
        # Statistical significance tests
        if len(results) >= 2:
            report.append("\n📈 STATISTICAL SIGNIFICANCE TESTS")
            report.append("-" * 80)
            
            config_names = list(results.keys())
            
            for i, config1 in enumerate(config_names):
                for j, config2 in enumerate(config_names[i+1:], i+1):
                    name1 = self.synthetic_names.get(config1, config1)
                    name2 = self.synthetic_names.get(config2, config2)
                    report.append(f"\n{name1} vs {name2}:")
                    
                    # Perform t-tests for key metrics
                    for metric_name, metric_key, _ in [('Fill Rate', 'fill_rate', None), 
                                                      ('Bid Efficiency', 'bid_efficiency', None),
                                                      ('Participation Rate', 'participation_rate', None)]:
                        
                        values1 = [getattr(SimulationMetrics(**run), metric_key) 
                                 for run in results[config1].raw_runs]
                        values2 = [getattr(SimulationMetrics(**run), metric_key) 
                                 for run in results[config2].raw_runs]
                        
                        if len(values1) >= 2 and len(values2) >= 2:
                            t_stat, p_value = StatisticalAnalysis.perform_t_test(values1, values2)
                            significance = ("***" if p_value < 0.001 else "**" if p_value < 0.01 
                                          else "*" if p_value < 0.05 else "ns")
                            report.append(f"    {metric_name}: t={t_stat:.3f}, p={p_value:.3f} {significance}")
        
        # Reflection Impact Analysis
        reflection_configs = {k: v for k, v in results.items() if 'llm_llm_reflections' in k}
        if len(reflection_configs) >= 2:
            report.append("\n🔄 REFLECTION IMPACT ANALYSIS")
            report.append("-" * 80)
            
            with_reflections = None
            without_reflections = None
            
            for config_name, result in reflection_configs.items():
                if 'reflections_true' in config_name:
                    with_reflections = result
                elif 'reflections_false' in config_name:
                    without_reflections = result
            
            if with_reflections and without_reflections:
                report.append("Reflections enable agents to adapt their strategies based on market feedback.")
                report.append("This additional configuration demonstrates how reflections change actor behavior:\n")
                
                # Compare key metrics
                metrics_comparison = [
                    ("Fill Rate", "fill_rate", "Higher fill rate with reflections indicates better job matching"),
                    ("Bid Efficiency", "bid_efficiency", "Efficiency changes show strategic adaptation"),
                    ("Participation Rate", "participation_rate", "Reflections affect agent engagement levels"),
                    ("Freelancer Hiring Rate", "freelancer_hiring_rate", "Hiring success improves with strategic learning"),
                    ("Rejection Rate", "rejection_rate", "Lower rejection with reflections shows better targeting")
                ]
                
                for metric_name, metric_key, explanation in metrics_comparison:
                    with_val = getattr(with_reflections.metrics_mean, metric_key)
                    without_val = getattr(without_reflections.metrics_mean, metric_key)
                    
                    if metric_key == "rejection_rate":
                        # For rejection rate, lower is better
                        improvement = without_val - with_val
                        direction = "decreased" if improvement > 0 else "increased"
                        report.append(f"• {metric_name}: {direction} by {abs(improvement):.1%} ({without_val:.1%} → {with_val:.1%})")
                    else:
                        # For other metrics, higher is generally better
                        improvement = with_val - without_val
                        direction = "increased" if improvement > 0 else "decreased"
                        report.append(f"• {metric_name}: {direction} by {abs(improvement):.1%} ({without_val:.1%} → {with_val:.1%})")
                    
                    report.append(f"  └─ {explanation}")
                
                # Explain bid efficiency paradox
                bid_eff_with = with_reflections.metrics_mean.bid_efficiency
                bid_eff_without = without_reflections.metrics_mean.bid_efficiency
                
                if bid_eff_without > bid_eff_with:
                    report.append("\n📊 BID EFFICIENCY PARADOX:")
                    report.append(f"Bid efficiency is higher WITHOUT reflections ({bid_eff_without:.1%} vs {bid_eff_with:.1%}).")
                    report.append("This occurs because:")
                    report.append("• Without reflections, agents make simpler, more predictable bids")
                    report.append("• With reflections, agents become more selective and strategic")
                    report.append("• Strategic agents may bid on fewer jobs but with better targeting")
                    report.append("• The efficiency metric (filled jobs / total bids) can appear lower")
                    report.append("  when agents are more selective about bidding")
        
        # Key insights
        report.append("\n💡 KEY INSIGHTS")
        report.append("-" * 50)
        
        # Find best performing configurations
        best_fill_rate = max(results.values(), key=lambda x: x.metrics_mean.fill_rate)
        best_efficiency = max(results.values(), key=lambda x: x.metrics_mean.bid_efficiency)
        most_consistent = min(results.values(), key=lambda x: x.metrics_std.fill_rate)
        
        report.append(f"• Best Fill Rate: {self.synthetic_names.get(best_fill_rate.config_name, best_fill_rate.config_name)} ({best_fill_rate.metrics_mean.fill_rate:.1%})")
        report.append(f"• Best Bid Efficiency: {self.synthetic_names.get(best_efficiency.config_name, best_efficiency.config_name)} ({best_efficiency.metrics_mean.bid_efficiency:.1%})")
        report.append(f"• Most Consistent: {self.synthetic_names.get(most_consistent.config_name, most_consistent.config_name)} (σ={most_consistent.metrics_std.fill_rate:.3f})")
        
        return "\n".join(report)


def main():
    """Main analysis function"""
    print("🚀 Starting Enhanced Multi-Run Marketplace Analysis")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = EnhancedMultiRunAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    if not results:
        print("❌ No results to analyze!")
        return 1
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(results)
    print(report)
    
    # Save report
    with open("enhanced_multi_run_analysis_report.txt", 'w') as f:
        f.write(report)
    print("\n📄 Detailed report saved to: enhanced_multi_run_analysis_report.txt")
    
    # Create comprehensive visualizations
    analyzer.create_comprehensive_visualizations(results)
    print("📊 Comprehensive visualizations saved to: analysis_results/")
    
    # Save aggregated data
    summary_data = {}
    for config_name, result in results.items():
        summary_data[config_name] = {
            'n_runs': result.n_runs,
            'metrics_mean': result.metrics_mean.__dict__,
            'metrics_std': result.metrics_std.__dict__,
            'metrics_ci_lower': result.metrics_ci_lower.__dict__,
            'metrics_ci_upper': result.metrics_ci_upper.__dict__
        }
    
    with open("analysis_results/enhanced_aggregated_results.json", 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print("✅ Multi-run analysis complete!")
    print("\n🎨 Generated visualizations:")
    print("  • Performance comparison with error bars")
    print("  • Detailed metrics grid")
    print("  • Individual configuration trend plots")
    print("  • Comparative trend overlays with variance bands")
    print("  • Market evolution heatmap")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
