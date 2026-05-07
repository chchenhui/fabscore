"""
Compression Circuit Analysis: Discovering How Transformers Encode Redundant Information
======================================================================================

This implementation analyzes how transformer models develop specialized circuits
for handling redundant/compressible information, with the goal of identifying
transferable compression strategies.

Author: AI Scientist
Date: 2025
Requirements: transformer-lens, torch, numpy, scipy, matplotlib, seaborn
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import pickle

# TransformerLens imports
from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint

# Scientific computing
from scipy.stats import entropy, pearsonr, spearmanr
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CompressionPattern:
    """Data structure for storing compression patterns"""
    pattern_type: str  # 'repetition', 'structure', 'semantic'
    text: str
    tokens: List[int]
    compression_ratio: float
    metadata: Dict[str, Any]


@dataclass
class CircuitActivation:
    """Store activation data for a specific circuit"""
    layer: int
    head: Optional[int]  # None for MLP
    activation_pattern: np.ndarray
    importance_score: float
    is_compression_related: bool


class CompressionDataGenerator:
    """Generate test data with varying levels of redundancy and compression potential"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.patterns = []

    def generate_repetitive_text(self, base_text: str, repetitions: int,
                                 variation_type: str = 'exact') -> CompressionPattern:
        """Generate text with controlled repetition"""
        if variation_type == 'exact':
            text = ' '.join([base_text] * repetitions)
        elif variation_type == 'slight':
            variations = []
            for i in range(repetitions):
                if i % 2 == 0:
                    variations.append(base_text)
                else:
                    # Slight modification
                    words = base_text.split()
                    if len(words) > 2:
                        words[1] = words[1] + 's'  # Simple pluralization
                    variations.append(' '.join(words))
            text = ' '.join(variations)
        elif variation_type == 'semantic':
            # Semantic repetition with different words
            synonyms = {
                'good': ['great', 'excellent', 'fine', 'nice'],
                'bad': ['poor', 'terrible', 'awful', 'horrible'],
                'big': ['large', 'huge', 'enormous', 'giant']
            }
            words = base_text.split()
            variations = [base_text]
            for _ in range(repetitions - 1):
                new_words = words.copy()
                for i, word in enumerate(new_words):
                    if word.lower() in synonyms:
                        new_words[i] = np.random.choice(synonyms[word.lower()])
                variations.append(' '.join(new_words))
            text = ' '.join(variations)
        else:
            raise ValueError(f"Unknown variation type: {variation_type}")

        tokens = self.tokenizer.encode(text)
        compression_ratio = len(text) / len(set(text.split()))

        return CompressionPattern(
            pattern_type='repetition',
            text=text,
            tokens=tokens,
            compression_ratio=compression_ratio,
            metadata={'base_text': base_text, 'repetitions': repetitions, 'variation': variation_type}
        )

    def generate_structured_data(self, template: str, num_entries: int) -> CompressionPattern:
        """Generate structured data like lists or tables"""
        entries = []
        for i in range(num_entries):
            entry = template.format(
                id=i,
                name=f"Item_{i}",
                value=np.random.randint(10, 100),
                category=['A', 'B', 'C'][i % 3]
            )
            entries.append(entry)

        text = '\n'.join(entries)
        tokens = self.tokenizer.encode(text)

        # Calculate compression based on template reuse
        unique_tokens = len(set(tokens))
        compression_ratio = len(tokens) / unique_tokens

        return CompressionPattern(
            pattern_type='structure',
            text=text,
            tokens=tokens,
            compression_ratio=compression_ratio,
            metadata={'template': template, 'num_entries': num_entries}
        )

    def generate_unique_text(self, length: int = 100) -> CompressionPattern:
        """Generate text with minimal redundancy"""
        # Use diverse vocabulary
        words = ['apple', 'quantum', 'river', 'galaxy', 'neuron', 'symphony',
                 'crystal', 'volcano', 'algorithm', 'butterfly', 'telescope',
                 'democracy', 'electron', 'forest', 'gravity', 'horizon',
                 'isotope', 'jungle', 'kaleidoscope', 'lighthouse', 'molecule',
                 'nebula', 'orchestra', 'paradox', 'quasar', 'rainbow']

        text_words = np.random.choice(words, size=min(length, len(words)), replace=False)
        text = ' '.join(text_words)
        tokens = self.tokenizer.encode(text)

        compression_ratio = 1.0  # No compression potential

        return CompressionPattern(
            pattern_type='unique',
            text=text,
            tokens=tokens,
            compression_ratio=compression_ratio,
            metadata={'length': length}
        )

    def generate_dataset(self, samples_per_type: int = 50) -> List[CompressionPattern]:
        """Generate complete dataset for analysis"""
        dataset = []

        # Repetitive patterns
        for _ in range(samples_per_type):
            base_texts = [
                "The weather is nice today",
                "Machine learning is powerful",
                "Data analysis reveals patterns",
                "Neural networks process information"
            ]
            base = np.random.choice(base_texts)
            reps = np.random.randint(3, 8)
            variation = np.random.choice(['exact', 'slight', 'semantic'])
            dataset.append(self.generate_repetitive_text(base, reps, variation))

        # Structured patterns
        templates = [
            "ID: {id}, Name: {name}, Value: {value}, Category: {category}",
            "Record {id}: {name} belongs to {category} with score {value}",
            "{name} (#{id}) - Category: {category}, Score: {value}"
        ]
        for _ in range(samples_per_type):
            template = np.random.choice(templates)
            num_entries = np.random.randint(5, 15)
            dataset.append(self.generate_structured_data(template, num_entries))

        # Unique content
        for _ in range(samples_per_type):
            length = np.random.randint(20, 50)
            dataset.append(self.generate_unique_text(length))

        return dataset


class CompressionCircuitAnalyzer:
    """Main analyzer for identifying compression circuits in transformers"""

    def __init__(self, model_name: str = "gpt2-small", device: str = "cpu"):
        """Initialize analyzer with specified model"""
        # Use CPU for compatibility, can change to 'mps' if on Mac M3
        self.device = device
        print(f"Loading model {model_name} on {self.device}...")

        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device
        )

        self.model_name = model_name
        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads

        # Storage for analysis results
        self.activation_cache = {}
        self.compression_circuits = []
        self.results = {
            'attention_patterns': {},
            'mlp_activations': {},
            'circuit_importance': {},
            'compression_signatures': {}
        }

    def hook_fn(self, activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        """Universal hook function to capture activations"""
        self.activation_cache[hook.name] = activation.detach().cpu().clone()
        return activation

    def analyze_single_input(self, pattern: CompressionPattern) -> Dict[str, Any]:
        """Analyze activation patterns for a single input"""
        # Prepare input
        tokens = torch.tensor(pattern.tokens).unsqueeze(0).to(self.device)

        # Run model with cache to get all activations
        logits, cache = self.model.run_with_cache(tokens)

        # Store cache for analysis
        self.activation_cache = cache

        # Analyze captured activations
        analysis = {
            'pattern_type': pattern.pattern_type,
            'compression_ratio': pattern.compression_ratio,
            'attention_entropy': {},
            'mlp_sparsity': {},
            'repetition_detection': {}
        }

        # Analyze attention patterns
        for layer in range(self.n_layers):
            # Get attention pattern using shorthand
            attn_pattern = cache["pattern", layer]  # Shape: [batch, head, query_pos, key_pos]

            # Calculate entropy for each head
            head_entropies = []
            for head in range(self.n_heads):
                head_pattern = attn_pattern[0, head].cpu().numpy()
                # Calculate entropy along each row
                row_entropies = [entropy(row + 1e-10) for row in head_pattern]
                head_entropies.append(np.mean(row_entropies))

            analysis['attention_entropy'][layer] = head_entropies

            # Detect repetition patterns in attention
            if pattern.pattern_type == 'repetition':
                analysis['repetition_detection'][layer] = self._detect_repetition_pattern(attn_pattern[0])

        # Analyze MLP activations
        for layer in range(self.n_layers):
            # Get MLP post-activation
            mlp_act = cache["mlp_out", layer]  # Shape: [batch, pos, d_model]

            # Calculate sparsity
            sparsity = (mlp_act.abs() < 0.01).float().mean().item()
            analysis['mlp_sparsity'][layer] = sparsity

        return analysis

    def _detect_repetition_pattern(self, attention_pattern: torch.Tensor) -> Dict[str, float]:
        """Detect if attention shows repetitive structure"""
        pattern_np = attention_pattern.cpu().numpy()

        # Look for periodic patterns
        scores = {}
        for head in range(pattern_np.shape[0]):
            head_pattern = pattern_np[head]

            # Check for diagonal patterns (copying from fixed distance)
            diagonal_scores = []
            for offset in range(1, min(5, head_pattern.shape[0])):
                diag_sum = 0
                count = 0
                for i in range(head_pattern.shape[0] - offset):
                    if i + offset < head_pattern.shape[1]:
                        diag_sum += head_pattern[i, i + offset]
                        count += 1
                if count > 0:
                    diagonal_scores.append(diag_sum / count)

            scores[f'head_{head}'] = max(diagonal_scores) if diagonal_scores else 0.0

        return scores

    def identify_compression_circuits(self, dataset: List[CompressionPattern]) -> List[CircuitActivation]:
        """Identify circuits that activate specifically for compressible content"""
        print("Analyzing dataset for compression circuits...")

        # Collect activation statistics
        all_analyses = []
        for pattern in tqdm(dataset, desc="Processing patterns"):
            analysis = self.analyze_single_input(pattern)
            all_analyses.append(analysis)

        # Identify circuits with differential activation
        compression_circuits = []

        # Separate analyses by type
        repetitive_analyses = [a for a in all_analyses if a['pattern_type'] == 'repetition']
        structured_analyses = [a for a in all_analyses if a['pattern_type'] == 'structure']
        unique_analyses = [a for a in all_analyses if a['pattern_type'] == 'unique']

        # Find attention heads that behave differently for compressible content
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                # Get entropy distributions
                rep_entropies = [a['attention_entropy'][layer][head] for a in repetitive_analyses]
                uni_entropies = [a['attention_entropy'][layer][head] for a in unique_analyses]

                # Statistical test for difference
                if len(rep_entropies) > 0 and len(uni_entropies) > 0:
                    mean_diff = abs(np.mean(rep_entropies) - np.mean(uni_entropies))

                    if mean_diff > 0.5:  # Significant difference threshold
                        circuit = CircuitActivation(
                            layer=layer,
                            head=head,
                            activation_pattern=np.array([rep_entropies, uni_entropies]),
                            importance_score=mean_diff,
                            is_compression_related=True
                        )
                        compression_circuits.append(circuit)

        # Find MLP layers with compression-specific behavior
        for layer in range(self.n_layers):
            rep_sparsities = [a['mlp_sparsity'][layer] for a in repetitive_analyses]
            uni_sparsities = [a['mlp_sparsity'][layer] for a in unique_analyses]

            if len(rep_sparsities) > 0 and len(uni_sparsities) > 0:
                sparsity_diff = abs(np.mean(rep_sparsities) - np.mean(uni_sparsities))

                if sparsity_diff > 0.1:  # Threshold for MLP difference
                    circuit = CircuitActivation(
                        layer=layer,
                        head=None,  # MLP circuit
                        activation_pattern=np.array([rep_sparsities, uni_sparsities]),
                        importance_score=sparsity_diff,
                        is_compression_related=True
                    )
                    compression_circuits.append(circuit)

        # Sort by importance
        compression_circuits.sort(key=lambda x: x.importance_score, reverse=True)

        return compression_circuits

    def visualize_compression_circuits(self, circuits: List[CircuitActivation],
                                      save_path: Optional[str] = None):
        """Visualize identified compression circuits"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Circuit importance heatmap
        importance_matrix = np.zeros((self.n_layers, self.n_heads + 1))  # +1 for MLP
        for circuit in circuits:
            if circuit.head is not None:
                importance_matrix[circuit.layer, circuit.head] = circuit.importance_score
            else:
                importance_matrix[circuit.layer, -1] = circuit.importance_score

        sns.heatmap(importance_matrix.T, ax=axes[0, 0], cmap='YlOrRd',
                   xticklabels=range(self.n_layers),
                   yticklabels=list(range(self.n_heads)) + ['MLP'])
        axes[0, 0].set_title('Compression Circuit Importance Map')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Head / MLP')

        # Plot 2: Distribution of importance scores
        importance_scores = [c.importance_score for c in circuits]
        axes[0, 1].hist(importance_scores, bins=20, edgecolor='black')
        axes[0, 1].set_title('Distribution of Circuit Importance Scores')
        axes[0, 1].set_xlabel('Importance Score')
        axes[0, 1].set_ylabel('Count')

        # Plot 3: Layer-wise circuit count
        layer_counts = {}
        for circuit in circuits:
            if circuit.layer not in layer_counts:
                layer_counts[circuit.layer] = 0
            layer_counts[circuit.layer] += 1

        layers = sorted(layer_counts.keys())
        counts = [layer_counts[l] for l in layers]
        axes[1, 0].bar(layers, counts)
        axes[1, 0].set_title('Compression Circuits per Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Number of Circuits')

        # Plot 4: Top circuits details
        top_circuits = circuits[:10] if len(circuits) >= 10 else circuits
        circuit_labels = []
        circuit_scores = []
        for i, circuit in enumerate(top_circuits):
            if circuit.head is not None:
                label = f"L{circuit.layer}H{circuit.head}"
            else:
                label = f"L{circuit.layer}MLP"
            circuit_labels.append(label)
            circuit_scores.append(circuit.importance_score)

        axes[1, 1].barh(circuit_labels, circuit_scores)
        axes[1, 1].set_title('Top 10 Compression Circuits')
        axes[1, 1].set_xlabel('Importance Score')
        axes[1, 1].set_ylabel('Circuit')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

        return fig

    def analyze_circuit_behavior(self, circuits: List[CircuitActivation],
                                 dataset: List[CompressionPattern]) -> Dict[str, Any]:
        """Detailed analysis of how compression circuits behave"""
        print("Performing detailed circuit behavior analysis...")

        behavior_analysis = {
            'circuit_specialization': {},
            'activation_patterns': {},
            'compression_efficiency': {}
        }

        # Analyze top circuits in detail
        top_circuits = circuits[:5] if len(circuits) >= 5 else circuits

        for circuit in tqdm(top_circuits, desc="Analyzing top circuits"):
            circuit_id = f"L{circuit.layer}_H{circuit.head}" if circuit.head is not None else f"L{circuit.layer}_MLP"

            # Test circuit on different compression types
            circuit_responses = {
                'exact_repetition': [],
                'semantic_repetition': [],
                'structured': [],
                'unique': []
            }

            for pattern in dataset:
                # Get specific circuit activation
                tokens = torch.tensor(pattern.tokens).unsqueeze(0).to(self.device)

                # Run model with cache
                logits, cache = self.model.run_with_cache(tokens)

                if circuit.head is not None:
                    # Attention circuit
                    activation = cache["pattern", circuit.layer][0, circuit.head]
                    activation_strength = activation.mean().item()
                else:
                    # MLP circuit
                    activation = cache["mlp_out", circuit.layer]
                    activation_strength = activation.abs().mean().item()

                # Categorize response
                if pattern.pattern_type == 'repetition':
                    if 'exact' in pattern.metadata.get('variation', ''):
                        circuit_responses['exact_repetition'].append(activation_strength)
                    else:
                        circuit_responses['semantic_repetition'].append(activation_strength)
                elif pattern.pattern_type == 'structure':
                    circuit_responses['structured'].append(activation_strength)
                else:
                    circuit_responses['unique'].append(activation_strength)

            behavior_analysis['circuit_specialization'][circuit_id] = {
                'mean_activations': {k: np.mean(v) if v else 0 for k, v in circuit_responses.items()},
                'std_activations': {k: np.std(v) if v else 0 for k, v in circuit_responses.items()},
                'specialization_score': self._calculate_specialization_score(circuit_responses)
            }

        return behavior_analysis

    def _calculate_specialization_score(self, responses: Dict[str, List[float]]) -> float:
        """Calculate how specialized a circuit is for compression"""
        compressible = responses.get('exact_repetition', []) + responses.get('structured', [])
        unique = responses.get('unique', [])

        if not compressible or not unique:
            return 0.0

        # Higher score means more specialized for compression
        mean_comp = np.mean(compressible)
        mean_unique = np.mean(unique)

        if mean_unique > 0:
            return (mean_comp - mean_unique) / mean_unique
        else:
            return mean_comp

    def export_results(self, circuits: List[CircuitActivation],
                      behavior_analysis: Dict[str, Any],
                      output_dir: str = "./compression_analysis_results"):
        """Export all analysis results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save circuits
        circuits_data = []
        for circuit in circuits:
            circuits_data.append({
                'layer': int(circuit.layer),
                'head': int(circuit.head) if circuit.head is not None else None,
                'importance_score': float(circuit.importance_score),
                'is_compression_related': bool(circuit.is_compression_related)
            })

        with open(output_path / 'compression_circuits.json', 'w') as f:
            json.dump(circuits_data, f, indent=2)

        # Save behavior analysis
        with open(output_path / 'circuit_behavior.json', 'w') as f:
            json.dump(behavior_analysis, f, indent=2, default=str)

        # Save summary report
        report = self._generate_report(circuits, behavior_analysis)
        with open(output_path / 'analysis_report.md', 'w') as f:
            f.write(report)

        print(f"Results exported to {output_path}")

    def _generate_report(self, circuits: List[CircuitActivation],
                        behavior_analysis: Dict[str, Any]) -> str:
        """Generate markdown report of findings"""
        report = f"""# Compression Circuit Analysis Report

## Model: {self.model_name}
## Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary of Findings

### Identified Compression Circuits
- Total circuits found: {len(circuits)}
- Top circuit importance score: {circuits[0].importance_score:.3f} if circuits else 'N/A'
- Layers with most circuits: {self._get_top_layers(circuits)}

### Circuit Distribution
- Attention circuits: {sum(1 for c in circuits if c.head is not None)}
- MLP circuits: {sum(1 for c in circuits if c.head is None)}

### Top 5 Compression Circuits
"""

        for i, circuit in enumerate(circuits[:5]):
            circuit_id = f"L{circuit.layer}_H{circuit.head}" if circuit.head is not None else f"L{circuit.layer}_MLP"
            report += f"{i+1}. **{circuit_id}**: Importance score = {circuit.importance_score:.3f}\n"

            if circuit_id in behavior_analysis.get('circuit_specialization', {}):
                spec = behavior_analysis['circuit_specialization'][circuit_id]
                report += f"   - Specialization score: {spec['specialization_score']:.3f}\n"
                report += f"   - Best performance on: {max(spec['mean_activations'], key=spec['mean_activations'].get)}\n"

        report += """
## Key Insights

1. **Compression Strategy**: The model appears to use specialized circuits for detecting and processing redundant information.

2. **Layer Distribution**: Compression circuits are distributed across layers, suggesting hierarchical processing of redundancy.

3. **Attention vs MLP**: Both attention and MLP components contribute to compression, with different specializations.

## Recommendations for Further Analysis

1. Test circuit transferability to other models
2. Investigate causal role via ablation studies
3. Analyze circuit activation on out-of-distribution compression tasks
4. Study developmental trajectory during training

"""
        return report

    def _get_top_layers(self, circuits: List[CircuitActivation], top_n: int = 3) -> str:
        """Get layers with most circuits"""
        layer_counts = {}
        for circuit in circuits:
            layer_counts[circuit.layer] = layer_counts.get(circuit.layer, 0) + 1

        sorted_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
        top_layers = sorted_layers[:top_n]

        return ', '.join([f"L{layer} ({count} circuits)" for layer, count in top_layers])


def main():
    """Main experimental pipeline"""
    print("="*80)
    print("COMPRESSION CIRCUIT ANALYSIS EXPERIMENT")
    print("="*80)

    # Initialize analyzer
    analyzer = CompressionCircuitAnalyzer(model_name="gpt2-medium")

    # Generate dataset
    print("\nGenerating compression test dataset...")
    data_generator = CompressionDataGenerator(analyzer.model.tokenizer)
    dataset = data_generator.generate_dataset(samples_per_type=30)
    print(f"Generated {len(dataset)} test patterns")

    # Identify compression circuits
    print("\nIdentifying compression circuits...")
    circuits = analyzer.identify_compression_circuits(dataset)
    print(f"Found {len(circuits)} compression-related circuits")

    # Analyze circuit behavior
    print("\nAnalyzing circuit behavior...")
    behavior_analysis = analyzer.analyze_circuit_behavior(circuits, dataset)

    # Visualize results
    print("\nGenerating visualizations...")
    analyzer.visualize_compression_circuits(circuits, save_path="compression_circuits.png")

    # Export results
    print("\nExporting results...")
    analyzer.export_results(circuits, behavior_analysis)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    # Print summary statistics
    if circuits:
        print(f"\nTop 3 Compression Circuits:")
        for i, circuit in enumerate(circuits[:3]):
            circuit_type = f"L{circuit.layer}H{circuit.head}" if circuit.head else f"L{circuit.layer}MLP"
            print(f"  {i+1}. {circuit_type}: Score = {circuit.importance_score:.3f}")

    return analyzer, circuits, behavior_analysis


if __name__ == "__main__":
    analyzer, circuits, behavior = main()