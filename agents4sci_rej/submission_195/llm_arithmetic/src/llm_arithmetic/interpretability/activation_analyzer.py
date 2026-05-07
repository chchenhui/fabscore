"""Activation analysis for arithmetic reasoning."""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class ActivationAnalysisResult:
    """Results from activation analysis."""
    
    problem: str
    tokens: List[str]
    activations: Dict[str, np.ndarray]  # Layer name -> activation array
    activation_statistics: Dict[str, Dict[str, float]]
    neuron_importance: Dict[str, np.ndarray]
    operation_specific_patterns: Dict[str, Any]
    dimensionality_analysis: Dict[str, Any]


class ActivationAnalyzer:
    """Analyze internal activations during arithmetic reasoning."""
    
    def __init__(self, model_name_or_path: str, device: str = "auto"):
        self.model_name_or_path = model_name_or_path
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            torch_dtype=torch.float32,
            output_hidden_states=True
        ).to(self.device)
        
        self.model.eval()
        self.hooks = []
        self.activations = {}
    
    def _register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks to capture activations."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu().numpy()
                else:
                    self.activations[name] = output.detach().cpu().numpy()
            return hook
        
        if layer_names is None:
            # Register hooks for all transformer layers
            for name, module in self.model.named_modules():
                if 'layer' in name.lower() and ('mlp' in name.lower() or 'feed_forward' in name.lower()):
                    handle = module.register_forward_hook(hook_fn(name))
                    self.hooks.append(handle)
        else:
            for name in layer_names:
                module = dict(self.model.named_modules())[name]
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)
    
    def _clear_hooks(self):
        """Clear all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def extract_activations(self, text: str, layer_names: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Extract activations from model for given text."""
        self._clear_hooks()
        self._register_hooks(layer_names)
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Also capture hidden states
        if 'hidden_states' not in self.activations:
            hidden_states = [h.detach().cpu().numpy() for h in outputs.hidden_states]
            for i, hidden_state in enumerate(hidden_states):
                self.activations[f'hidden_layer_{i}'] = hidden_state[0]  # Remove batch dimension
        
        activations_copy = self.activations.copy()
        self._clear_hooks()
        
        return tokens, activations_copy
    
    def analyze_arithmetic_problem(self, problem: str) -> ActivationAnalysisResult:
        """Analyze activations for a specific arithmetic problem."""
        formatted_input = f"Solve: {problem}"
        
        tokens, activations = self.extract_activations(formatted_input)
        
        # Calculate activation statistics
        activation_statistics = self._calculate_activation_statistics(activations)
        
        # Analyze neuron importance
        neuron_importance = self._analyze_neuron_importance(activations, tokens)
        
        # Analyze operation-specific patterns
        operation_patterns = self._analyze_operation_patterns(problem, tokens, activations)
        
        # Dimensionality analysis
        dimensionality_analysis = self._analyze_dimensionality(activations)
        
        return ActivationAnalysisResult(
            problem=problem,
            tokens=tokens,
            activations=activations,
            activation_statistics=activation_statistics,
            neuron_importance=neuron_importance,
            operation_specific_patterns=operation_patterns,
            dimensionality_analysis=dimensionality_analysis
        )
    
    def _calculate_activation_statistics(self, activations: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for each layer's activations."""
        stats = {}
        
        for layer_name, activation in activations.items():
            # activation shape: [seq_len, hidden_dim]
            stats[layer_name] = {
                'mean_activation': float(np.mean(activation)),
                'std_activation': float(np.std(activation)),
                'max_activation': float(np.max(activation)),
                'min_activation': float(np.min(activation)),
                'sparsity': float(np.mean(activation == 0.0)),
                'dead_neurons': int(np.sum(np.all(activation == 0.0, axis=0))),
                'active_neurons': int(np.sum(np.any(activation != 0.0, axis=0))),
                'l2_norm': float(np.linalg.norm(activation)),
                'frobenius_norm': float(np.linalg.norm(activation, 'fro'))
            }
        
        return stats
    
    def _analyze_neuron_importance(self, activations: Dict[str, np.ndarray], 
                                 tokens: List[str]) -> Dict[str, np.ndarray]:
        """Analyze which neurons are most important for arithmetic reasoning."""
        importance = {}
        
        for layer_name, activation in activations.items():
            # Calculate variance across sequence positions as importance measure
            neuron_variance = np.var(activation, axis=0)
            
            # Calculate mean absolute activation as another importance measure
            neuron_mean_abs = np.mean(np.abs(activation), axis=0)
            
            # Calculate maximum activation as peak importance
            neuron_max = np.max(np.abs(activation), axis=0)
            
            importance[layer_name] = {
                'variance': neuron_variance,
                'mean_absolute': neuron_mean_abs,
                'max_activation': neuron_max,
                'composite_score': neuron_variance * neuron_mean_abs
            }
        
        return importance
    
    def _analyze_operation_patterns(self, problem: str, tokens: List[str], 
                                  activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze activation patterns specific to arithmetic operations."""
        patterns = {}
        
        # Identify operation and number positions
        operation_symbols = {"+", "-", "×", "÷", "*", "/", "plus", "minus", "times", "divided"}
        operation_positions = []
        number_positions = []
        
        for i, token in enumerate(tokens):
            if any(op in token.lower() for op in operation_symbols):
                operation_positions.append(i)
            elif any(char.isdigit() for char in token):
                number_positions.append(i)
        
        # Analyze activation patterns at operation vs number positions
        for layer_name, activation in activations.items():
            layer_patterns = {}
            
            if operation_positions:
                op_activations = activation[operation_positions]
                layer_patterns['operation_activation_stats'] = {
                    'mean': float(np.mean(op_activations)),
                    'std': float(np.std(op_activations)),
                    'l2_norm': float(np.linalg.norm(op_activations))
                }
            
            if number_positions:
                num_activations = activation[number_positions]
                layer_patterns['number_activation_stats'] = {
                    'mean': float(np.mean(num_activations)),
                    'std': float(np.std(num_activations)),
                    'l2_norm': float(np.linalg.norm(num_activations))
                }
            
            # Calculate difference in activation patterns
            if operation_positions and number_positions:
                op_mean = np.mean(activation[operation_positions], axis=0)
                num_mean = np.mean(activation[number_positions], axis=0)
                diff_pattern = op_mean - num_mean
                
                layer_patterns['operation_vs_number_difference'] = {
                    'l2_norm': float(np.linalg.norm(diff_pattern)),
                    'cosine_similarity': float(
                        np.dot(op_mean, num_mean) / (np.linalg.norm(op_mean) * np.linalg.norm(num_mean))
                    ),
                    'most_different_neurons': np.argsort(np.abs(diff_pattern))[-10:].tolist()
                }
            
            patterns[layer_name] = layer_patterns
        
        return patterns
    
    def _analyze_dimensionality(self, activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze the effective dimensionality of activations."""
        dimensionality_analysis = {}
        
        for layer_name, activation in activations.items():
            # PCA analysis
            try:
                pca = PCA()
                pca.fit(activation.T)  # Fit on features (neurons)
                
                # Calculate effective dimensionality (90% variance explained)
                cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
                effective_dim_90 = int(np.argmax(cumsum_ratio >= 0.9) + 1)
                effective_dim_95 = int(np.argmax(cumsum_ratio >= 0.95) + 1)
                
                # Calculate intrinsic dimensionality using participation ratio
                eigenvals = pca.explained_variance_
                participation_ratio = (np.sum(eigenvals) ** 2) / np.sum(eigenvals ** 2)
                
                dimensionality_analysis[layer_name] = {
                    'total_dimensions': activation.shape[1],
                    'effective_dim_90': effective_dim_90,
                    'effective_dim_95': effective_dim_95,
                    'participation_ratio': float(participation_ratio),
                    'explained_variance_ratio': pca.explained_variance_ratio_[:20].tolist(),  # Top 20
                    'top_eigenvalues': eigenvals[:20].tolist()
                }
            except Exception as e:
                dimensionality_analysis[layer_name] = {
                    'error': str(e),
                    'total_dimensions': activation.shape[1]
                }
        
        return dimensionality_analysis
    
    def visualize_activation_heatmap(self, result: ActivationAnalysisResult, 
                                   layer_name: str, max_neurons: int = 100,
                                   save_path: Optional[str] = None) -> go.Figure:
        """Create heatmap of activations for a specific layer."""
        activation = result.activations[layer_name]
        
        # Select most important neurons if too many
        if activation.shape[1] > max_neurons:
            importance_scores = result.neuron_importance[layer_name]['composite_score']
            top_neurons = np.argsort(importance_scores)[-max_neurons:]
            activation = activation[:, top_neurons]
            neuron_labels = [f"Neuron {i}" for i in top_neurons]
        else:
            neuron_labels = [f"Neuron {i}" for i in range(activation.shape[1])]
        
        fig = go.Figure(data=go.Heatmap(
            z=activation.T,  # Transpose for better visualization
            x=result.tokens,
            y=neuron_labels,
            colorscale='RdBu',
            zmid=0,
            showscale=True
        ))
        
        fig.update_layout(
            title=f"Activation Heatmap: {layer_name}<br>Problem: {result.problem}",
            xaxis_title="Tokens",
            yaxis_title="Neurons",
            width=1200,
            height=600
        )
        
        fig.update_xaxes(tickangle=45)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_neuron_importance(self, result: ActivationAnalysisResult, 
                                  layer_name: str, top_k: int = 20,
                                  save_path: Optional[str] = None) -> go.Figure:
        """Visualize most important neurons in a layer."""
        importance = result.neuron_importance[layer_name]['composite_score']
        top_indices = np.argsort(importance)[-top_k:]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f"Neuron {i}" for i in top_indices],
                y=importance[top_indices],
                marker_color='steelblue'
            )
        ])
        
        fig.update_layout(
            title=f"Top {top_k} Most Important Neurons: {layer_name}<br>Problem: {result.problem}",
            xaxis_title="Neurons",
            yaxis_title="Importance Score",
            width=800,
            height=500
        )
        
        fig.update_xaxes(tickangle=45)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_dimensionality_analysis(self, result: ActivationAnalysisResult,
                                        save_path: Optional[str] = None) -> go.Figure:
        """Visualize dimensionality analysis across layers."""
        layer_names = []
        effective_dims_90 = []
        effective_dims_95 = []
        participation_ratios = []
        
        for layer_name, analysis in result.dimensionality_analysis.items():
            if 'error' not in analysis:
                layer_names.append(layer_name)
                effective_dims_90.append(analysis['effective_dim_90'])
                effective_dims_95.append(analysis['effective_dim_95'])
                participation_ratios.append(analysis['participation_ratio'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=layer_names,
            y=effective_dims_90,
            mode='lines+markers',
            name='90% Variance',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=layer_names,
            y=effective_dims_95,
            mode='lines+markers',
            name='95% Variance',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f"Effective Dimensionality Across Layers<br>Problem: {result.problem}",
            xaxis_title="Layers",
            yaxis_title="Effective Dimensions",
            width=800,
            height=500
        )
        
        fig.update_xaxes(tickangle=45)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def compare_problems(self, problems: List[str]) -> Dict[str, Any]:
        """Compare activation patterns across multiple arithmetic problems."""
        results = []
        for problem in problems:
            result = self.analyze_arithmetic_problem(problem)
            results.append(result)
        
        comparison = {
            "problems": problems,
            "results": results,
            "summary_stats": self._compute_comparison_stats(results)
        }
        
        return comparison
    
    def _compute_comparison_stats(self, results: List[ActivationAnalysisResult]) -> Dict[str, Any]:
        """Compute summary statistics across multiple problems."""
        # Compare activation statistics across problems
        layer_comparisons = {}
        
        # Get common layers across all results
        common_layers = set(results[0].activations.keys())
        for result in results[1:]:
            common_layers = common_layers.intersection(set(result.activations.keys()))
        
        for layer_name in common_layers:
            stats_across_problems = {
                'mean_activation': [],
                'sparsity': [],
                'active_neurons': [],
                'effective_dim_90': [],
                'participation_ratio': []
            }
            
            for result in results:
                if layer_name in result.activation_statistics:
                    stats = result.activation_statistics[layer_name]
                    stats_across_problems['mean_activation'].append(stats['mean_activation'])
                    stats_across_problems['sparsity'].append(stats['sparsity'])
                    stats_across_problems['active_neurons'].append(stats['active_neurons'])
                
                if layer_name in result.dimensionality_analysis:
                    dim_analysis = result.dimensionality_analysis[layer_name]
                    if 'effective_dim_90' in dim_analysis:
                        stats_across_problems['effective_dim_90'].append(dim_analysis['effective_dim_90'])
                    if 'participation_ratio' in dim_analysis:
                        stats_across_problems['participation_ratio'].append(dim_analysis['participation_ratio'])
            
            # Compute summary statistics
            layer_summary = {}
            for stat_name, values in stats_across_problems.items():
                if values:
                    layer_summary[stat_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            layer_comparisons[layer_name] = layer_summary
        
        return {
            'layer_comparisons': layer_comparisons,
            'num_problems': len(results),
            'common_layers': list(common_layers)
        }