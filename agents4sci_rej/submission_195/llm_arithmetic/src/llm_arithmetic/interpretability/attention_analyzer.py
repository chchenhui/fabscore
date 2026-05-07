"""Attention pattern analysis for arithmetic reasoning."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class AttentionAnalysisResult:
    """Results from attention pattern analysis."""
    
    problem: str
    tokens: List[str]
    attention_weights: np.ndarray  # [layers, heads, seq_len, seq_len]
    layer_attention_stats: Dict[int, Dict[str, float]]
    head_attention_stats: Dict[Tuple[int, int], Dict[str, float]]
    operation_attention_patterns: Dict[str, np.ndarray]
    position_bias_analysis: Dict[str, Any]


class AttentionAnalyzer:
    """Analyze attention patterns in arithmetic reasoning."""
    
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
            output_attentions=True
        ).to(self.device)
        
        self.model.eval()
    
    def extract_attention_patterns(self, text: str) -> Tuple[List[str], np.ndarray]:
        """Extract attention patterns from model for given text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # Tuple of [batch, heads, seq, seq] tensors
        
        # Convert to numpy and combine all layers
        attention_weights = []
        for layer_attention in attentions:
            # Remove batch dimension and convert to numpy
            layer_attn = layer_attention[0].cpu().numpy()  # [heads, seq, seq]
            attention_weights.append(layer_attn)
        
        attention_weights = np.stack(attention_weights)  # [layers, heads, seq, seq]
        
        return tokens, attention_weights
    
    def analyze_arithmetic_problem(self, problem: str) -> AttentionAnalysisResult:
        """Analyze attention patterns for a specific arithmetic problem."""
        # Create formatted input
        formatted_input = f"Solve: {problem}"
        
        tokens, attention_weights = self.extract_attention_patterns(formatted_input)
        
        # Analyze layer-level statistics
        layer_attention_stats = self._analyze_layer_attention(attention_weights)
        
        # Analyze head-level statistics
        head_attention_stats = self._analyze_head_attention(attention_weights)
        
        # Analyze operation-specific patterns
        operation_attention_patterns = self._analyze_operation_patterns(
            problem, tokens, attention_weights
        )
        
        # Analyze position bias
        position_bias_analysis = self._analyze_position_bias(tokens, attention_weights)
        
        return AttentionAnalysisResult(
            problem=problem,
            tokens=tokens,
            attention_weights=attention_weights,
            layer_attention_stats=layer_attention_stats,
            head_attention_stats=head_attention_stats,
            operation_attention_patterns=operation_attention_patterns,
            position_bias_analysis=position_bias_analysis
        )
    
    def _analyze_layer_attention(self, attention_weights: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Analyze attention statistics at layer level."""
        stats = {}
        
        for layer_idx in range(attention_weights.shape[0]):
            layer_attn = attention_weights[layer_idx]  # [heads, seq, seq]
            
            # Average across heads
            avg_layer_attn = np.mean(layer_attn, axis=0)  # [seq, seq]
            
            stats[layer_idx] = {
                "mean_attention": float(np.mean(avg_layer_attn)),
                "std_attention": float(np.std(avg_layer_attn)),
                "max_attention": float(np.max(avg_layer_attn)),
                "entropy": self._calculate_attention_entropy(avg_layer_attn),
                "diagonal_attention": float(np.mean(np.diag(avg_layer_attn))),
                "off_diagonal_attention": float(
                    np.mean(avg_layer_attn - np.diag(np.diag(avg_layer_attn)))
                )
            }
        
        return stats
    
    def _analyze_head_attention(self, attention_weights: np.ndarray) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Analyze attention statistics at head level."""
        stats = {}
        
        for layer_idx in range(attention_weights.shape[0]):
            for head_idx in range(attention_weights.shape[1]):
                head_attn = attention_weights[layer_idx, head_idx]  # [seq, seq]
                
                stats[(layer_idx, head_idx)] = {
                    "mean_attention": float(np.mean(head_attn)),
                    "std_attention": float(np.std(head_attn)),
                    "max_attention": float(np.max(head_attn)),
                    "entropy": self._calculate_attention_entropy(head_attn),
                    "sparsity": self._calculate_attention_sparsity(head_attn)
                }
        
        return stats
    
    def _analyze_operation_patterns(self, problem: str, tokens: List[str], 
                                  attention_weights: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze attention patterns specific to arithmetic operations."""
        patterns = {}
        
        # Find operation tokens (simplified - could be more sophisticated)
        operation_symbols = {"+", "-", "×", "÷", "*", "/", "plus", "minus", "times", "divided"}
        number_pattern = r'\d+'
        
        operation_positions = []
        number_positions = []
        
        for i, token in enumerate(tokens):
            if any(op in token.lower() for op in operation_symbols):
                operation_positions.append(i)
            elif any(char.isdigit() for char in token):
                number_positions.append(i)
        
        # Average attention across all layers and heads
        avg_attention = np.mean(attention_weights, axis=(0, 1))  # [seq, seq]
        
        # Operation-to-number attention
        if operation_positions and number_positions:
            op_to_num_attention = []
            for op_pos in operation_positions:
                for num_pos in number_positions:
                    op_to_num_attention.append(avg_attention[op_pos, num_pos])
            patterns["operation_to_number"] = np.array(op_to_num_attention)
        
        # Number-to-number attention
        if len(number_positions) >= 2:
            num_to_num_attention = []
            for i, pos1 in enumerate(number_positions):
                for j, pos2 in enumerate(number_positions):
                    if i != j:
                        num_to_num_attention.append(avg_attention[pos1, pos2])
            patterns["number_to_number"] = np.array(num_to_num_attention)
        
        # Self-attention on operations
        if operation_positions:
            op_self_attention = []
            for op_pos in operation_positions:
                op_self_attention.append(avg_attention[op_pos, op_pos])
            patterns["operation_self_attention"] = np.array(op_self_attention)
        
        return patterns
    
    def _analyze_position_bias(self, tokens: List[str], attention_weights: np.ndarray) -> Dict[str, Any]:
        """Analyze position-dependent attention biases."""
        seq_len = len(tokens)
        avg_attention = np.mean(attention_weights, axis=(0, 1))  # [seq, seq]
        
        # Calculate attention to different relative positions
        relative_position_attention = {}
        
        for rel_pos in range(-seq_len + 1, seq_len):
            if rel_pos == 0:
                continue  # Skip self-attention
            
            attentions = []
            for i in range(seq_len):
                j = i + rel_pos
                if 0 <= j < seq_len:
                    attentions.append(avg_attention[i, j])
            
            if attentions:
                relative_position_attention[rel_pos] = {
                    "mean": float(np.mean(attentions)),
                    "std": float(np.std(attentions)),
                    "count": len(attentions)
                }
        
        # Calculate positional attention bias (beginning vs end of sequence)
        beginning_attention = np.mean(avg_attention[:, :seq_len//3])
        middle_attention = np.mean(avg_attention[:, seq_len//3:2*seq_len//3])
        end_attention = np.mean(avg_attention[:, 2*seq_len//3:])
        
        return {
            "relative_position_attention": relative_position_attention,
            "positional_bias": {
                "beginning": float(beginning_attention),
                "middle": float(middle_attention),
                "end": float(end_attention)
            }
        }
    
    def _calculate_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """Calculate entropy of attention distribution."""
        # Calculate entropy for each query position
        entropies = []
        for i in range(attention_matrix.shape[0]):
            attn_dist = attention_matrix[i]
            # Add small epsilon to avoid log(0)
            attn_dist = attn_dist + 1e-12
            attn_dist = attn_dist / np.sum(attn_dist)
            entropy = -np.sum(attn_dist * np.log(attn_dist))
            entropies.append(entropy)
        
        return float(np.mean(entropies))
    
    def _calculate_attention_sparsity(self, attention_matrix: np.ndarray, threshold: float = 0.1) -> float:
        """Calculate sparsity of attention (fraction of weights below threshold)."""
        return float(np.mean(attention_matrix < threshold))
    
    def visualize_attention_heatmap(self, result: AttentionAnalysisResult, 
                                  layer_idx: int = -1, head_idx: int = 0,
                                  save_path: Optional[str] = None) -> go.Figure:
        """Create interactive attention heatmap."""
        if layer_idx == -1:
            layer_idx = result.attention_weights.shape[0] - 1
        
        attention_matrix = result.attention_weights[layer_idx, head_idx]
        
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=result.tokens,
            y=result.tokens,
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title=f"Attention Heatmap: Layer {layer_idx}, Head {head_idx}<br>Problem: {result.problem}",
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            width=800,
            height=800
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickangle=0)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_layer_comparison(self, result: AttentionAnalysisResult,
                                 save_path: Optional[str] = None) -> go.Figure:
        """Compare attention patterns across layers."""
        stats_by_layer = []
        layers = []
        
        for layer_idx, stats in result.layer_attention_stats.items():
            layers.append(layer_idx)
            stats_by_layer.append(stats)
        
        metrics = ['entropy', 'diagonal_attention', 'off_diagonal_attention']
        colors = ['blue', 'red', 'green']
        
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metrics,
            shared_yaxes=True
        )
        
        for i, metric in enumerate(metrics):
            values = [stats[metric] for stats in stats_by_layer]
            fig.add_trace(
                go.Scatter(
                    x=layers,
                    y=values,
                    mode='lines+markers',
                    name=metric,
                    line=dict(color=colors[i])
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title=f"Attention Statistics Across Layers<br>Problem: {result.problem}",
            height=400,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def compare_problems(self, problems: List[str]) -> Dict[str, Any]:
        """Compare attention patterns across multiple arithmetic problems."""
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
    
    def _compute_comparison_stats(self, results: List[AttentionAnalysisResult]) -> Dict[str, Any]:
        """Compute summary statistics across multiple problems."""
        # Compare entropy across problems
        entropies_by_layer = {}
        for layer in range(results[0].attention_weights.shape[0]):
            entropies = []
            for result in results:
                entropies.append(result.layer_attention_stats[layer]['entropy'])
            entropies_by_layer[layer] = {
                'mean': np.mean(entropies),
                'std': np.std(entropies),
                'values': entropies
            }
        
        # Compare operation patterns
        operation_pattern_stats = {}
        for pattern_type in ['operation_to_number', 'number_to_number']:
            pattern_values = []
            for result in results:
                if pattern_type in result.operation_attention_patterns:
                    pattern_values.extend(result.operation_attention_patterns[pattern_type])
            
            if pattern_values:
                operation_pattern_stats[pattern_type] = {
                    'mean': np.mean(pattern_values),
                    'std': np.std(pattern_values),
                    'count': len(pattern_values)
                }
        
        return {
            'entropy_by_layer': entropies_by_layer,
            'operation_patterns': operation_pattern_stats,
            'num_problems': len(results)
        }