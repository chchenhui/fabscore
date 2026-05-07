"""Comprehensive interpretability analysis suite."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .attention_analyzer import AttentionAnalyzer, AttentionAnalysisResult
from .activation_analyzer import ActivationAnalyzer, ActivationAnalysisResult
from ..evaluation.base_evaluator import ArithmeticProblem


@dataclass
class ComprehensiveAnalysisResult:
    """Results from comprehensive interpretability analysis."""
    
    problem: str
    attention_analysis: AttentionAnalysisResult
    activation_analysis: ActivationAnalysisResult
    cross_analysis: Dict[str, Any]
    insights: List[str]


class InterpretabilitySuite:
    """Comprehensive interpretability analysis for arithmetic reasoning."""
    
    def __init__(self, model_name_or_path: str, device: str = "auto"):
        self.model_name_or_path = model_name_or_path
        self.device = device
        
        # Initialize analyzers
        self.attention_analyzer = AttentionAnalyzer(model_name_or_path, device)
        self.activation_analyzer = ActivationAnalyzer(model_name_or_path, device)
        
        # Storage for results
        self.analysis_results = []
    
    def analyze_problem(self, problem: str) -> ComprehensiveAnalysisResult:
        """Perform comprehensive analysis of a single arithmetic problem."""
        print(f"Analyzing problem: {problem}")
        
        # Attention analysis
        print("  Running attention analysis...")
        attention_result = self.attention_analyzer.analyze_arithmetic_problem(problem)
        
        # Activation analysis  
        print("  Running activation analysis...")
        activation_result = self.activation_analyzer.analyze_arithmetic_problem(problem)
        
        # Cross-analysis
        print("  Running cross-analysis...")
        cross_analysis = self._perform_cross_analysis(attention_result, activation_result)
        
        # Generate insights
        insights = self._generate_insights(problem, attention_result, activation_result, cross_analysis)
        
        result = ComprehensiveAnalysisResult(
            problem=problem,
            attention_analysis=attention_result,
            activation_analysis=activation_result,
            cross_analysis=cross_analysis,
            insights=insights
        )
        
        self.analysis_results.append(result)
        return result
    
    def analyze_problems(self, problems: List[str]) -> List[ComprehensiveAnalysisResult]:
        """Analyze multiple arithmetic problems."""
        results = []
        
        for i, problem in enumerate(problems):
            print(f"\\nAnalyzing problem {i+1}/{len(problems)}: {problem}")
            result = self.analyze_problem(problem)
            results.append(result)
        
        return results
    
    def _perform_cross_analysis(self, attention_result: AttentionAnalysisResult, 
                               activation_result: ActivationAnalysisResult) -> Dict[str, Any]:
        """Perform cross-analysis between attention and activation patterns."""
        cross_analysis = {}
        
        # Attention-Activation correlation analysis
        cross_analysis['attention_activation_correlation'] = self._analyze_attention_activation_correlation(
            attention_result, activation_result
        )
        
        # Layer-wise analysis
        cross_analysis['layer_wise_patterns'] = self._analyze_layer_wise_patterns(
            attention_result, activation_result
        )
        
        # Position-based analysis
        cross_analysis['position_based_analysis'] = self._analyze_position_based_patterns(
            attention_result, activation_result
        )
        
        return cross_analysis
    
    def _analyze_attention_activation_correlation(self, attention_result: AttentionAnalysisResult,
                                                activation_result: ActivationAnalysisResult) -> Dict[str, Any]:
        """Analyze correlation between attention patterns and activations."""
        correlations = {}
        
        # Get average attention across heads for each layer
        num_layers = attention_result.attention_weights.shape[0]
        
        for layer_idx in range(num_layers):
            layer_name = f"hidden_layer_{layer_idx}"
            
            if layer_name in activation_result.activations:
                # Average attention across heads
                layer_attention = np.mean(attention_result.attention_weights[layer_idx], axis=0)
                
                # Get activation for this layer
                layer_activation = activation_result.activations[layer_name]
                
                # Compute correlation between attention entropy and activation magnitude
                attention_entropy_per_pos = []
                activation_magnitude_per_pos = []
                
                for pos in range(min(layer_attention.shape[0], layer_activation.shape[0])):
                    # Attention entropy for this position
                    attn_dist = layer_attention[pos] + 1e-12
                    attn_dist = attn_dist / np.sum(attn_dist)
                    entropy = -np.sum(attn_dist * np.log(attn_dist))
                    attention_entropy_per_pos.append(entropy)
                    
                    # Activation magnitude for this position
                    act_magnitude = np.linalg.norm(layer_activation[pos])
                    activation_magnitude_per_pos.append(act_magnitude)
                
                if len(attention_entropy_per_pos) > 1:
                    correlation = np.corrcoef(attention_entropy_per_pos, activation_magnitude_per_pos)[0, 1]
                    correlations[layer_idx] = {
                        'attention_entropy': attention_entropy_per_pos,
                        'activation_magnitude': activation_magnitude_per_pos,
                        'correlation': float(correlation) if not np.isnan(correlation) else 0.0
                    }
        
        return correlations
    
    def _analyze_layer_wise_patterns(self, attention_result: AttentionAnalysisResult,
                                   activation_result: ActivationAnalysisResult) -> Dict[str, Any]:
        """Analyze how patterns change across layers."""
        layer_patterns = {}
        
        num_layers = attention_result.attention_weights.shape[0]
        
        for layer_idx in range(num_layers):
            layer_name = f"hidden_layer_{layer_idx}"
            
            patterns = {
                'layer_index': layer_idx,
                'attention_stats': attention_result.layer_attention_stats.get(layer_idx, {}),
                'activation_stats': activation_result.activation_statistics.get(layer_name, {}),
            }
            
            # Add dimensionality info if available
            if layer_name in activation_result.dimensionality_analysis:
                patterns['dimensionality'] = activation_result.dimensionality_analysis[layer_name]
            
            layer_patterns[layer_idx] = patterns
        
        return layer_patterns
    
    def _analyze_position_based_patterns(self, attention_result: AttentionAnalysisResult,
                                       activation_result: ActivationAnalysisResult) -> Dict[str, Any]:
        """Analyze position-based patterns in attention and activations."""
        position_analysis = {}
        
        # Find positions of numbers and operations in tokens
        tokens = attention_result.tokens
        operation_symbols = {"+", "-", "×", "÷", "*", "/", "plus", "minus", "times", "divided"}
        
        operation_positions = []
        number_positions = []
        
        for i, token in enumerate(tokens):
            if any(op in token.lower() for op in operation_symbols):
                operation_positions.append(i)
            elif any(char.isdigit() for char in token):
                number_positions.append(i)
        
        position_analysis['identified_positions'] = {
            'operations': operation_positions,
            'numbers': number_positions,
            'total_tokens': len(tokens)
        }
        
        # Analyze attention patterns for these positions
        if operation_positions and number_positions:
            avg_attention = np.mean(attention_result.attention_weights, axis=(0, 1))
            
            # Operation-to-number attention
            op_to_num_attention = []
            for op_pos in operation_positions:
                for num_pos in number_positions:
                    op_to_num_attention.append(avg_attention[op_pos, num_pos])
            
            position_analysis['attention_patterns'] = {
                'operation_to_number_mean': float(np.mean(op_to_num_attention)) if op_to_num_attention else 0.0,
                'operation_to_number_std': float(np.std(op_to_num_attention)) if op_to_num_attention else 0.0
            }
        
        return position_analysis
    
    def _generate_insights(self, problem: str, attention_result: AttentionAnalysisResult,
                          activation_result: ActivationAnalysisResult, 
                          cross_analysis: Dict[str, Any]) -> List[str]:
        """Generate human-readable insights from the analysis."""
        insights = []
        
        # Attention insights
        avg_entropy = np.mean([stats['entropy'] for stats in attention_result.layer_attention_stats.values()])
        if avg_entropy > 3.0:
            insights.append("High attention entropy suggests the model is considering many tokens simultaneously")
        elif avg_entropy < 1.5:
            insights.append("Low attention entropy indicates focused attention on specific tokens")
        
        # Check for attention to operations vs numbers
        if 'attention_patterns' in cross_analysis.get('position_based_analysis', {}):
            op_to_num_attn = cross_analysis['position_based_analysis']['attention_patterns']['operation_to_number_mean']
            if op_to_num_attn > 0.1:
                insights.append("Model shows strong attention from operation tokens to number tokens")
            elif op_to_num_attn < 0.05:
                insights.append("Model shows weak attention between operations and numbers")
        
        # Activation insights
        layer_names = list(activation_result.activation_statistics.keys())
        if layer_names:
            # Check sparsity patterns
            sparsity_values = [activation_result.activation_statistics[layer]['sparsity'] 
                             for layer in layer_names if 'sparsity' in activation_result.activation_statistics[layer]]
            
            if sparsity_values:
                avg_sparsity = np.mean(sparsity_values)
                if avg_sparsity > 0.5:
                    insights.append("High activation sparsity indicates selective neuron firing")
                elif avg_sparsity < 0.1:
                    insights.append("Low activation sparsity suggests dense, distributed processing")
        
        # Dimensionality insights
        effective_dims = []
        for layer_analysis in activation_result.dimensionality_analysis.values():
            if 'effective_dim_90' in layer_analysis:
                effective_dims.append(layer_analysis['effective_dim_90'])
        
        if effective_dims:
            avg_effective_dim = np.mean(effective_dims)
            total_dims = list(activation_result.activations.values())[0].shape[1] if activation_result.activations else 0
            
            if total_dims > 0:
                dim_ratio = avg_effective_dim / total_dims
                if dim_ratio < 0.1:
                    insights.append("Very low effective dimensionality suggests highly structured representations")
                elif dim_ratio > 0.5:
                    insights.append("High effective dimensionality indicates complex, distributed representations")
        
        # Cross-analysis insights
        correlations = cross_analysis.get('attention_activation_correlation', {})
        if correlations:
            correlation_values = [data['correlation'] for data in correlations.values() 
                                if 'correlation' in data]
            if correlation_values:
                avg_correlation = np.mean(correlation_values)
                if abs(avg_correlation) > 0.5:
                    insights.append(f"{'Strong positive' if avg_correlation > 0 else 'Strong negative'} correlation between attention entropy and activation magnitude")
        
        return insights
    
    def create_comprehensive_report(self, result: ComprehensiveAnalysisResult, 
                                  output_dir: Path) -> str:
        """Create a comprehensive HTML report for a single problem analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        attention_heatmap = self.attention_analyzer.visualize_attention_heatmap(result.attention_analysis)
        layer_comparison = self.attention_analyzer.visualize_layer_comparison(result.attention_analysis)
        
        # Select a representative layer for activation visualization
        layer_names = list(result.activation_analysis.activations.keys())
        if layer_names:
            middle_layer = layer_names[len(layer_names)//2]
            activation_heatmap = self.activation_analyzer.visualize_activation_heatmap(
                result.activation_analysis, middle_layer
            )
            neuron_importance = self.activation_analyzer.visualize_neuron_importance(
                result.activation_analysis, middle_layer
            )
            dimensionality_plot = self.activation_analyzer.visualize_dimensionality_analysis(
                result.activation_analysis
            )
        
        # Create combined report
        from plotly.offline import plot
        import plotly.io as pio
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Arithmetic Interpretability Analysis: {result.problem}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 30px 0; }}
                .insight {{ background-color: #f0f8ff; padding: 10px; border-left: 4px solid #0066cc; margin: 10px 0; }}
                .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Arithmetic Interpretability Analysis</h1>
            <h2>Problem: {result.problem}</h2>
            
            <div class="section">
                <h3>Key Insights</h3>
                {''.join([f'<div class="insight">{insight}</div>' for insight in result.insights])}
            </div>
            
            <div class="section">
                <h3>Attention Analysis</h3>
                <div class="stats">
                    <h4>Layer Statistics</h4>
                    <p>Average Entropy: {np.mean([stats['entropy'] for stats in result.attention_analysis.layer_attention_stats.values()]):.3f}</p>
                    <p>Average Diagonal Attention: {np.mean([stats['diagonal_attention'] for stats in result.attention_analysis.layer_attention_stats.values()]):.3f}</p>
                </div>
                {pio.to_html(attention_heatmap, include_plotlyjs=True, div_id="attention_heatmap")}
                {pio.to_html(layer_comparison, include_plotlyjs=False, div_id="layer_comparison")}
            </div>
            
            <div class="section">
                <h3>Activation Analysis</h3>
                <div class="stats">
                    <h4>Overall Statistics</h4>
                    <p>Total Layers Analyzed: {len(result.activation_analysis.activations)}</p>
                    <p>Average Sparsity: {np.mean([stats.get('sparsity', 0) for stats in result.activation_analysis.activation_statistics.values()]):.3f}</p>
                </div>
                {pio.to_html(activation_heatmap, include_plotlyjs=False, div_id="activation_heatmap") if layer_names else ""}
                {pio.to_html(neuron_importance, include_plotlyjs=False, div_id="neuron_importance") if layer_names else ""}
                {pio.to_html(dimensionality_plot, include_plotlyjs=False, div_id="dimensionality") if layer_names else ""}
            </div>
            
            <div class="section">
                <h3>Cross-Analysis</h3>
                <div class="stats">
                    <h4>Position-Based Analysis</h4>
                    <pre>{json.dumps(result.cross_analysis.get('position_based_analysis', {}), indent=2)}</pre>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save report
        report_file = output_dir / f"interpretability_report_{result.problem.replace(' ', '_').replace('/', 'div')}.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        return str(report_file)
    
    def analyze_problem_set(self, problems: List[str], output_dir: Path) -> Dict[str, Any]:
        """Analyze a set of problems and create comparative analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Analyzing {len(problems)} problems...")
        
        # Analyze all problems
        results = self.analyze_problems(problems)
        
        # Create individual reports
        report_files = []
        for result in results:
            report_file = self.create_comprehensive_report(result, output_dir / "individual_reports")
            report_files.append(report_file)
        
        # Comparative analysis
        comparative_analysis = self._create_comparative_analysis(results)
        
        # Save comparative analysis
        comparison_file = output_dir / "comparative_analysis.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparative_analysis, f, indent=2)
        
        # Create summary report
        summary_report = self._create_summary_report(results, comparative_analysis, output_dir)
        
        return {
            'individual_reports': report_files,
            'comparative_analysis': str(comparison_file),
            'summary_report': summary_report,
            'results': results
        }
    
    def _create_comparative_analysis(self, results: List[ComprehensiveAnalysisResult]) -> Dict[str, Any]:
        """Create comparative analysis across multiple problems."""
        if not results:
            return {}
        
        # Compare attention patterns
        attention_stats = []
        for result in results:
            stats = {
                'problem': result.problem,
                'avg_entropy': np.mean([s['entropy'] for s in result.attention_analysis.layer_attention_stats.values()]),
                'avg_diagonal_attention': np.mean([s['diagonal_attention'] for s in result.attention_analysis.layer_attention_stats.values()]),
            }
            attention_stats.append(stats)
        
        # Compare activation patterns
        activation_stats = []
        for result in results:
            stats = {
                'problem': result.problem,
                'avg_sparsity': np.mean([s.get('sparsity', 0) for s in result.activation_analysis.activation_statistics.values()]),
                'total_layers': len(result.activation_analysis.activations)
            }
            activation_stats.append(stats)
        
        # Identify patterns by operation type
        operation_patterns = {}
        for result in results:
            # Simple operation detection
            problem = result.problem.lower()
            if '+' in problem or 'plus' in problem:
                op_type = 'addition'
            elif '-' in problem or 'minus' in problem:
                op_type = 'subtraction'
            elif '×' in problem or '*' in problem or 'times' in problem:
                op_type = 'multiplication'
            elif '÷' in problem or '/' in problem or 'divided' in problem:
                op_type = 'division'
            else:
                op_type = 'unknown'
            
            if op_type not in operation_patterns:
                operation_patterns[op_type] = []
            
            operation_patterns[op_type].append({
                'problem': result.problem,
                'attention_entropy': np.mean([s['entropy'] for s in result.attention_analysis.layer_attention_stats.values()]),
                'activation_sparsity': np.mean([s.get('sparsity', 0) for s in result.activation_analysis.activation_statistics.values()])
            })
        
        return {
            'attention_comparison': attention_stats,
            'activation_comparison': activation_stats,
            'operation_patterns': operation_patterns,
            'num_problems': len(results)
        }
    
    def _create_summary_report(self, results: List[ComprehensiveAnalysisResult], 
                             comparative_analysis: Dict[str, Any], output_dir: Path) -> str:
        """Create a summary HTML report."""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Arithmetic Interpretability Analysis Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 30px 0; }}
                .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Arithmetic Interpretability Analysis Summary</h1>
            <p>Analysis of {len(results)} arithmetic problems</p>
            
            <div class="section">
                <h2>Overall Statistics</h2>
                <div class="stats">
                    <h3>Attention Patterns</h3>
                    <p>Problems analyzed: {len(comparative_analysis.get('attention_comparison', []))}</p>
                    
                    <h3>Activation Patterns</h3>
                    <p>Problems analyzed: {len(comparative_analysis.get('activation_comparison', []))}</p>
                    
                    <h3>Operation Types</h3>
                    <ul>
                        {''.join([f'<li>{op}: {len(patterns)} problems</li>' 
                                for op, patterns in comparative_analysis.get('operation_patterns', {}).items()])}
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>Individual Problem Results</h2>
                <table>
                    <tr>
                        <th>Problem</th>
                        <th>Key Insights</th>
                        <th>Attention Entropy</th>
                        <th>Activation Sparsity</th>
                    </tr>
                    {''.join([
                        f'''<tr>
                            <td>{result.problem}</td>
                            <td>{'; '.join(result.insights[:2])}</td>
                            <td>{np.mean([s['entropy'] for s in result.attention_analysis.layer_attention_stats.values()]):.3f}</td>
                            <td>{np.mean([s.get('sparsity', 0) for s in result.activation_analysis.activation_statistics.values()]):.3f}</td>
                        </tr>'''
                        for result in results
                    ])}
                </table>
            </div>
        </body>
        </html>
        """
        
        summary_file = output_dir / "summary_report.html"
        with open(summary_file, 'w') as f:
            f.write(html_content)
        
        return str(summary_file)