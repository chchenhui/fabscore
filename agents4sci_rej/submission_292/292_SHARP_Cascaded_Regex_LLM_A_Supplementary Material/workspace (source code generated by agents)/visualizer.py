"""
Visualization module for experiment results
Generates charts and visual reports
"""

import os
import json
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Visualizations will be skipped.")

class ResultsVisualizer:
    """Generate visualizations for experiment results"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_all_visualizations(self, results, test_data=None):
        """Generate all visualization types"""
        
        logger.info("Generating visualizations...")
        
        if VISUALIZATION_AVAILABLE:
            # 1. Performance comparison bar chart
            self.plot_performance_comparison(results)
            
            # 2. Confusion matrices
            self.plot_confusion_matrices(results)
            
            # 3. Metrics radar chart
            self.plot_radar_chart(results)
            
            # 5. Time comparison if available
            if any('time_seconds' in metrics for metrics in results.values()):
                self.plot_time_comparison(results)
        else:
            logger.warning("Skipping plots due to missing matplotlib/seaborn")
        
        # 4. Performance table (always generate - doesn't need matplotlib)
        self.generate_performance_table(results)
        
        # 6. Error analysis if test data available
        if test_data:
            self.analyze_errors(results, test_data)
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def plot_performance_comparison(self, results):
        """Create bar chart comparing all methods"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Phishing Detection Performance Comparison', fontsize=16, fontweight='bold')
        
        methods = list(results.keys())
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            values = [results[method][metric] for method in methods]
            colors = self._get_colors(values)
            
            bars = ax.bar(range(len(methods)), values, color=colors)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontweight='bold')
            
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_ylim(0, 1.1)
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, results):
        """Plot confusion matrices for all methods"""
        
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
        
        if n_methods == 1:
            axes = [axes]
        
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        for idx, (method_name, metrics) in enumerate(results.items()):
            ax = axes[idx]
            
            if 'confusion_matrix' in metrics:
                cm = np.array(metrics['confusion_matrix'])
                
                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           ax=ax, cbar=idx == n_methods-1,
                           xticklabels=['Legitimate', 'Phishing'],
                           yticklabels=['Legitimate', 'Phishing'])
                
                ax.set_title(method_name)
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_radar_chart(self, results):
        """Create radar chart for multi-metric comparison"""
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # Metrics to include
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if 'specificity' in list(results.values())[0]:
            metrics.append('specificity')
        
        # Number of metrics
        num_vars = len(metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        metrics = metrics
        angles += angles[:1]
        
        # Plot each method
        for method_name, method_metrics in results.items():
            values = [method_metrics.get(m, 0) for m in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method_name)
            ax.fill(angles, values, alpha=0.25)
        
        # Fix axis to go in the right order
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Multi-Metric Performance Comparison', size=16, fontweight='bold', pad=20)
        
        plt.savefig(os.path.join(self.output_dir, 'radar_chart.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_time_comparison(self, results):
        """Plot execution time comparison"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        methods = []
        times = []
        f1_scores = []
        
        for method_name, metrics in results.items():
            if 'time_seconds' in metrics:
                methods.append(method_name)
                times.append(metrics['time_seconds'])
                f1_scores.append(metrics['f1_score'])
        
        if not times:
            return
        
        # Time comparison bar chart
        colors = self._get_colors(times, reverse=True)
        bars = ax1.bar(range(len(methods)), times, color=colors)
        
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s',
                    ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Time vs F1-Score scatter plot
        ax2.scatter(times, f1_scores, s=100, alpha=0.7)
        
        for i, method in enumerate(methods):
            ax2.annotate(method, (times[i], f1_scores[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('Time vs Performance Trade-off')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Efficiency Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'time_comparison.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_performance_table(self, results):
        """Generate a performance summary table"""
        
        # Create HTML table
        html = """
        <html>
        <head>
            <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                    font-family: Arial, sans-serif;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: center;
                }
                th {
                    background-color: #4CAF50;
                    color: white;
                }
                tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                .best {
                    font-weight: bold;
                    color: #2E7D32;
                }
            </style>
        </head>
        <body>
            <h2>Phishing Detection Performance Summary</h2>
            <table>
                <tr>
                    <th>Method</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
        """
        
        if any('time_seconds' in metrics for metrics in results.values()):
            html += "<th>Time (s)</th>"
        
        html += "</tr>"
        
        # Find best values for each metric
        best_values = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            best_values[metric] = max(results[m][metric] for m in results)
        
        # Sort by F1-score
        sorted_methods = sorted(results.items(), 
                              key=lambda x: x[1]['f1_score'], 
                              reverse=True)
        
        for method_name, metrics in sorted_methods:
            html += "<tr>"
            html += f"<td>{method_name}</td>"
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                value = metrics[metric]
                is_best = (value == best_values[metric])
                
                if is_best:
                    html += f'<td class="best">{value:.3f}</td>'
                else:
                    html += f"<td>{value:.3f}</td>"
            
            if 'time_seconds' in metrics:
                html += f"<td>{metrics['time_seconds']:.2f}</td>"
            
            html += "</tr>"
        
        html += """
            </table>
        </body>
        </html>
        """
        
        # Save HTML table
        with open(os.path.join(self.output_dir, 'performance_table.html'), 'w') as f:
            f.write(html)
    
    def analyze_errors(self, results, test_data):
        """Analyze misclassified samples"""
        
        error_analysis = {}
        
        for method_name in results:
            error_analysis[method_name] = {
                'false_positives': [],
                'false_negatives': []
            }
        
        # Save error analysis to JSON
        with open(os.path.join(self.output_dir, 'error_analysis.json'), 'w') as f:
            json.dump(error_analysis, f, indent=2)
    
    def _get_colors(self, values, reverse=False):
        """Get colors based on values (green=good, red=bad)"""
        if not values:
            return []
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return ['blue'] * len(values)
        
        colors = []
        for val in values:
            normalized = (val - min_val) / (max_val - min_val)
            if reverse:
                normalized = 1 - normalized
            
            if normalized > 0.7:
                colors.append('green')
            elif normalized > 0.4:
                colors.append('orange')
            else:
                colors.append('red')
        
        return colors