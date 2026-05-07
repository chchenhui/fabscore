"""
Enhanced Noise Robustness Analysis with Statistical Testing and Causal Intervention
==================================================================================
Extended experiment with statistical rigor, causal verification, and advanced noise patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from typing import Dict, List, Tuple, Optional, Any
import random
import string
from dataclasses import dataclass, field
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EnhancedNoiseConfig:
    """Extended configuration for noise injection"""
    noise_type: str
    noise_level: float
    seed: int = 42
    adversarial: bool = False
    targeted_layer: Optional[int] = None
    semantic_distance: Optional[float] = None


@dataclass
class StatisticalResult:
    """Store statistical test results"""
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    significant: bool = False


class AdvancedNoiseInjector:
    """Enhanced noise injection with semantic and adversarial patterns"""

    def __init__(self, tokenizer, model=None):
        self.tokenizer = tokenizer
        self.model = model
        random.seed(42)
        np.random.seed(42)

        # Semantic substitution dictionaries
        self.antonyms = {
            'good': 'bad', 'bad': 'good', 'hot': 'cold', 'cold': 'hot',
            'big': 'small', 'small': 'big', 'fast': 'slow', 'slow': 'fast',
            'happy': 'sad', 'sad': 'happy', 'right': 'wrong', 'wrong': 'right'
        }

        self.semantic_groups = {
            'animals': ['dog', 'cat', 'bird', 'fish', 'mouse', 'elephant'],
            'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white'],
            'actions': ['run', 'walk', 'jump', 'sit', 'stand', 'lie']
        }

    def inject_semantic_noise(self, text: str, noise_level: float) -> Tuple[str, List[int], Dict]:
        """Replace words with semantically distant alternatives"""
        words = text.split()
        num_changes = max(1, int(len(words) * noise_level))
        change_positions = []
        semantic_shifts = []

        for _ in range(num_changes):
            if words:
                pos = random.randint(0, len(words) - 1)
                original = words[pos].lower().strip('.,!?')

                # Try antonym substitution
                if original in self.antonyms:
                    replacement = self.antonyms[original]
                    words[pos] = words[pos].replace(original, replacement)
                    change_positions.append(pos)
                    semantic_shifts.append(('antonym', original, replacement))
                else:
                    # Random semantic group substitution
                    for group_name, group_words in self.semantic_groups.items():
                        if original in group_words:
                            other_words = [w for w in group_words if w != original]
                            if other_words:
                                replacement = random.choice(other_words)
                                words[pos] = words[pos].replace(original, replacement)
                                change_positions.append(pos)
                                semantic_shifts.append((group_name, original, replacement))
                                break

        return ' '.join(words), change_positions, {'semantic_shifts': semantic_shifts}

    def inject_adversarial_noise(self, text: str, model, noise_level: float) -> Tuple[str, List[int], Dict]:
        """Generate adversarial perturbations that maximize model confusion"""
        if not model:
            return text, [], {}

        model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        with torch.enable_grad():
            inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
            inputs_embeds.requires_grad = True

            # Forward pass
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=inputs['attention_mask'])
            loss = outputs.hidden_states[-1].mean()

            # Compute gradients
            loss.backward()
            gradients = inputs_embeds.grad

            # Apply perturbation in direction of maximum change
            num_tokens = int(inputs_embeds.shape[1] * noise_level)
            grad_norms = gradients.norm(dim=-1).squeeze()

            # Select tokens with highest gradients
            top_indices = grad_norms.topk(min(num_tokens, len(grad_norms))).indices

            # Perturb embeddings
            epsilon = 0.1
            perturbed_embeds = inputs_embeds.clone()
            for idx in top_indices:
                perturbed_embeds[0, idx] += epsilon * gradients[0, idx].sign()

        # Find nearest tokens for perturbed embeddings
        perturbed_tokens = []
        embedding_matrix = model.get_input_embeddings().weight

        for idx in top_indices:
            perturbed_vec = perturbed_embeds[0, idx]
            distances = torch.cdist(perturbed_vec.unsqueeze(0), embedding_matrix).squeeze()
            nearest_token = distances.argmin().item()
            perturbed_tokens.append((idx.item(), nearest_token))

        # Reconstruct text with adversarial tokens
        tokens = inputs['input_ids'][0].tolist()
        for pos, new_token in perturbed_tokens:
            if pos < len(tokens):
                tokens[pos] = new_token

        adversarial_text = self.tokenizer.decode(tokens, skip_special_tokens=True)

        return adversarial_text, [p[0] for p in perturbed_tokens], {
            'gradient_norms': grad_norms.tolist(),
            'perturbation_strength': epsilon
        }

    def inject_ocr_noise(self, text: str, noise_level: float) -> Tuple[str, List[int]]:
        """Simulate OCR errors with visually similar characters"""
        ocr_confusion = {
            'o': '0', '0': 'o', 'l': '1', '1': 'l', 'i': '1',
            'S': '5', '5': 'S', 'O': '0', 'B': '8', '8': 'B',
            'rn': 'm', 'm': 'rn', 'cl': 'd', 'h': 'b'
        }

        chars = list(text)
        num_errors = int(len(chars) * noise_level)
        error_positions = []

        for _ in range(num_errors):
            pos = random.randint(0, len(chars) - 1)
            char = chars[pos]

            if char in ocr_confusion:
                chars[pos] = ocr_confusion[char]
                error_positions.append(pos)

        return ''.join(chars), error_positions


class CausalInterventionAnalyzer:
    """Perform causal intervention to verify circuit function"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.hooks = []
        self.activations = {}

    def register_hook(self, layer_idx: int, head_indices: Optional[List[int]] = None):
        """Register hook to modify attention heads"""
        def create_hook(name, head_indices):
            def hook_fn(module, input, output):
                if head_indices is not None:
                    # Zero out specific heads
                    attention_output = output[0]
                    for head_idx in head_indices:
                        attention_output[:, head_idx, :, :] = 0
                    return (attention_output,) + output[1:]
                else:
                    # Store activation for analysis
                    self.activations[name] = output[0].detach().cpu()
                return output
            return hook_fn

        layer = self.model.encoder.layer[layer_idx] if hasattr(self.model, 'encoder') else \
                self.model.transformer.h[layer_idx]

        hook = layer.attention.register_forward_hook(
            create_hook(f'layer_{layer_idx}', head_indices)
        )
        self.hooks.append(hook)
        return hook

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def ablate_heads(self, text: str, layer_idx: int, head_indices: List[int]) -> Dict:
        """Ablate specific heads and measure impact"""
        # Get baseline performance
        baseline_output = self.get_model_output(text)

        # Ablate heads
        self.register_hook(layer_idx, head_indices)
        ablated_output = self.get_model_output(text)
        self.remove_hooks()

        # Calculate impact
        impact = {
            'representation_change': torch.norm(
                baseline_output['hidden_states'] - ablated_output['hidden_states']
            ).item(),
            'attention_entropy_change': self.calculate_entropy_change(
                baseline_output['attentions'], ablated_output['attentions']
            ),
            'prediction_change': self.calculate_prediction_change(
                baseline_output, ablated_output
            )
        }

        return impact

    def get_model_output(self, text: str) -> Dict:
        """Get model outputs for analysis"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)

        return {
            'hidden_states': outputs.hidden_states[-1].mean(dim=1),
            'attentions': torch.stack(outputs.attentions).mean(dim=0),
            'logits': outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None
        }

    def calculate_entropy_change(self, baseline_attn, ablated_attn):
        """Calculate change in attention entropy"""
        def entropy(tensor):
            # Normalize and calculate entropy
            probs = F.softmax(tensor.view(-1), dim=0)
            return -(probs * torch.log(probs + 1e-10)).sum().item()

        baseline_entropy = entropy(baseline_attn)
        ablated_entropy = entropy(ablated_attn)

        return abs(baseline_entropy - ablated_entropy)

    def calculate_prediction_change(self, baseline, ablated):
        """Calculate change in model predictions"""
        if baseline['logits'] is None:
            return 0.0

        baseline_preds = torch.argmax(baseline['logits'], dim=-1)
        ablated_preds = torch.argmax(ablated['logits'], dim=-1)

        return (baseline_preds != ablated_preds).float().mean().item()


class StatisticalAnalyzer:
    """Perform statistical analysis on experimental results"""

    @staticmethod
    def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data"""
        n = len(data)
        if n < 2:
            return (data[0], data[0]) if data else (0, 0)

        mean = np.mean(data)
        sem = stats.sem(data)
        interval = sem * stats.t.ppf((1 + confidence) / 2, n - 1)

        return (mean - interval, mean + interval)

    @staticmethod
    def cohens_d(group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    @staticmethod
    def perform_statistical_test(clean_scores: List[float], noisy_scores: List[float]) -> StatisticalResult:
        """Perform comprehensive statistical testing"""
        # Basic statistics
        mean_clean = np.mean(clean_scores)
        mean_noisy = np.mean(noisy_scores)
        std_noisy = np.std(noisy_scores)

        # Confidence interval
        ci = StatisticalAnalyzer.calculate_confidence_interval(noisy_scores)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(clean_scores, noisy_scores)

        # Effect size
        effect_size = StatisticalAnalyzer.cohens_d(clean_scores, noisy_scores)

        return StatisticalResult(
            mean=mean_noisy,
            std=std_noisy,
            confidence_interval=ci,
            p_value=p_value,
            effect_size=effect_size,
            significant=p_value < 0.05
        )

    @staticmethod
    def analyze_distribution(scores: List[float]) -> Dict:
        """Analyze score distribution"""
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'skewness': stats.skew(scores),
            'kurtosis': stats.kurtosis(scores),
            'normality_test': stats.normaltest(scores)
        }


class CrossLayerAnalyzer:
    """Analyze interactions between layers"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def compute_layer_similarity_matrix(self, text: str) -> np.ndarray:
        """Compute similarity between all layer pairs"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = [h.mean(dim=1).squeeze().cpu().numpy() for h in outputs.hidden_states]
        n_layers = len(hidden_states)

        similarity_matrix = np.zeros((n_layers, n_layers))

        for i in range(n_layers):
            for j in range(n_layers):
                similarity_matrix[i, j] = 1 - cosine(hidden_states[i], hidden_states[j])

        return similarity_matrix

    def identify_critical_layers(self, clean_text: str, noisy_text: str) -> Dict:
        """Identify layers critical for error correction"""
        clean_sim = self.compute_layer_similarity_matrix(clean_text)
        noisy_sim = self.compute_layer_similarity_matrix(noisy_text)

        # Compute difference in similarity patterns
        diff_matrix = np.abs(clean_sim - noisy_sim)

        # Identify layers with highest changes
        layer_importance = diff_matrix.sum(axis=1)
        critical_layers = np.argsort(layer_importance)[-5:][::-1]

        return {
            'critical_layers': critical_layers.tolist(),
            'importance_scores': layer_importance.tolist(),
            'similarity_change': diff_matrix
        }

    def analyze_information_flow(self, texts: List[str]) -> Dict:
        """Analyze how information flows through layers"""
        flow_patterns = []

        for text in texts:
            similarity_matrix = self.compute_layer_similarity_matrix(text)

            # Analyze diagonal neighbors (layer-to-layer flow)
            flow = []
            for i in range(len(similarity_matrix) - 1):
                flow.append(similarity_matrix[i, i+1])

            flow_patterns.append(flow)

        avg_flow = np.mean(flow_patterns, axis=0)

        # Identify bottlenecks (low similarity transitions)
        bottlenecks = np.where(avg_flow < np.percentile(avg_flow, 25))[0]

        # Identify highways (high similarity transitions)
        highways = np.where(avg_flow > np.percentile(avg_flow, 75))[0]

        return {
            'average_flow': avg_flow.tolist(),
            'bottleneck_layers': bottlenecks.tolist(),
            'highway_layers': highways.tolist()
        }


class EnhancedExperimentRunner:
    """Run comprehensive noise robustness experiments with statistical analysis"""

    def __init__(self):
        self.results = {}
        self.statistical_results = {}

    def run_comprehensive_experiment(self, model_names: List[str], test_sentences: List[str]):
        """Run extended experiments with all enhancements"""

        print("="*60)
        print("ENHANCED NOISE ROBUSTNESS EXPERIMENT")
        print("="*60)

        for model_name in model_names:
            print(f"\n{'='*50}")
            print(f"Analyzing model: {model_name}")
            print('='*50)

            # Initialize components
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()

            # Initialize analyzers
            noise_injector = AdvancedNoiseInjector(tokenizer, model)
            causal_analyzer = CausalInterventionAnalyzer(model, tokenizer)
            cross_layer_analyzer = CrossLayerAnalyzer(model, tokenizer)

            results = {
                'noise_robustness': {},
                'causal_intervention': {},
                'cross_layer_analysis': {},
                'statistical_tests': {}
            }

            # Test different noise types
            noise_types = ['char_swap', 'semantic', 'ocr', 'adversarial']
            noise_levels = [0.05, 0.10, 0.20]

            for noise_type in noise_types:
                print(f"\nTesting {noise_type} noise...")

                for noise_level in noise_levels:
                    clean_scores = []
                    noisy_scores = []

                    for sentence in tqdm(test_sentences, desc=f"{noise_type} @ {noise_level:.0%}"):
                        # Get clean baseline
                        clean_output = self.get_model_representation(model, tokenizer, sentence)

                        # Inject noise
                        if noise_type == 'semantic':
                            noisy_text, positions, meta = noise_injector.inject_semantic_noise(sentence, noise_level)
                        elif noise_type == 'ocr':
                            noisy_text, positions = noise_injector.inject_ocr_noise(sentence, noise_level)
                            meta = {}
                        elif noise_type == 'adversarial':
                            noisy_text, positions, meta = noise_injector.inject_adversarial_noise(
                                sentence, model, noise_level
                            )
                        else:  # char_swap
                            from noise_robustness_experiment import NoiseInjector, NoiseConfig
                            basic_injector = NoiseInjector(tokenizer)
                            config = NoiseConfig('char_swap', noise_level)
                            noisy_text, positions = basic_injector.inject_noise(sentence, config)
                            meta = {}

                        # Get noisy output
                        noisy_output = self.get_model_representation(model, tokenizer, noisy_text)

                        # Calculate robustness score
                        similarity = F.cosine_similarity(clean_output, noisy_output, dim=0).item()

                        clean_scores.append(1.0)  # Perfect score for clean
                        noisy_scores.append(similarity)

                    # Statistical analysis
                    stats_result = StatisticalAnalyzer.perform_statistical_test(clean_scores, noisy_scores)

                    key = f"{noise_type}_{noise_level}"
                    results['statistical_tests'][key] = stats_result
                    results['noise_robustness'][key] = {
                        'scores': noisy_scores,
                        'mean': stats_result.mean,
                        'std': stats_result.std,
                        'ci': stats_result.confidence_interval,
                        'significant_degradation': stats_result.significant
                    }

            # Causal intervention analysis
            print("\nPerforming causal intervention analysis...")

            # Identify critical heads from previous analysis
            critical_heads = [(3, [0, 1, 2]), (7, [5, 6]), (11, [8, 9, 10])]  # Example

            for layer_idx, head_indices in critical_heads:
                impacts = []
                for sentence in test_sentences[:5]:  # Sample for efficiency
                    impact = causal_analyzer.ablate_heads(sentence, layer_idx, head_indices)
                    impacts.append(impact['representation_change'])

                results['causal_intervention'][f'layer_{layer_idx}'] = {
                    'heads_ablated': head_indices,
                    'mean_impact': np.mean(impacts),
                    'std_impact': np.std(impacts)
                }

            # Cross-layer analysis
            print("\nAnalyzing cross-layer interactions...")

            # Sample sentences for analysis
            sample_sentences = random.sample(test_sentences, min(5, len(test_sentences)))

            critical_layers_list = []
            for sentence in sample_sentences:
                # Test with noisy version
                noisy_text, _ = noise_injector.inject_ocr_noise(sentence, 0.1)
                critical = cross_layer_analyzer.identify_critical_layers(sentence, noisy_text)
                critical_layers_list.extend(critical['critical_layers'])

            # Information flow analysis
            flow_analysis = cross_layer_analyzer.analyze_information_flow(sample_sentences)

            results['cross_layer_analysis'] = {
                'critical_layers': list(set(critical_layers_list)),
                'information_flow': flow_analysis
            }

            self.results[model_name] = results
            self.statistical_results[model_name] = results['statistical_tests']

        # Generate comprehensive report
        self.generate_report()
        self.create_advanced_visualizations()

    def get_model_representation(self, model, tokenizer, text):
        """Get model's final representation"""
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        return outputs.hidden_states[-1].mean(dim=1).squeeze()

    def generate_report(self):
        """Generate comprehensive statistical report"""
        report = []
        report.append("\n" + "="*60)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("="*60)

        for model_name, stats in self.statistical_results.items():
            report.append(f"\n### {model_name.upper()} ###")

            for test_name, result in stats.items():
                noise_type, level = test_name.rsplit('_', 1)
                report.append(f"\n{noise_type} @ {float(level):.0%}:")
                report.append(f"  Mean robustness: {result.mean:.4f} ± {result.std:.4f}")
                report.append(f"  95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
                report.append(f"  Effect size (Cohen's d): {result.effect_size:.3f}")
                report.append(f"  p-value: {result.p_value:.6f}")
                report.append(f"  Significant degradation: {'Yes' if result.significant else 'No'}")

        # Causal intervention results
        report.append("\n" + "="*60)
        report.append("CAUSAL INTERVENTION ANALYSIS")
        report.append("="*60)

        for model_name, results in self.results.items():
            if 'causal_intervention' in results:
                report.append(f"\n### {model_name.upper()} ###")
                for layer_name, impact in results['causal_intervention'].items():
                    report.append(f"\n{layer_name}:")
                    report.append(f"  Heads ablated: {impact['heads_ablated']}")
                    report.append(f"  Mean representation change: {impact['mean_impact']:.4f}")
                    report.append(f"  Std deviation: {impact['std_impact']:.4f}")

        # Cross-layer analysis
        report.append("\n" + "="*60)
        report.append("CROSS-LAYER INTERACTION ANALYSIS")
        report.append("="*60)

        for model_name, results in self.results.items():
            if 'cross_layer_analysis' in results:
                report.append(f"\n### {model_name.upper()} ###")
                analysis = results['cross_layer_analysis']
                report.append(f"Critical layers for error correction: {analysis['critical_layers']}")
                report.append(f"Information bottlenecks: {analysis['information_flow']['bottleneck_layers']}")
                report.append(f"Information highways: {analysis['information_flow']['highway_layers']}")

        # Save report
        report_text = '\n'.join(report)
        with open('enhanced_analysis_report.txt', 'w') as f:
            f.write(report_text)

        print(report_text)

    def create_advanced_visualizations(self):
        """Create comprehensive visualizations"""
        # Statistical significance heatmap
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for idx, (model_name, stats) in enumerate(self.statistical_results.items()):
            row = idx // 2
            col = idx % 2

            # Prepare data for heatmap
            noise_types = list(set([k.rsplit('_', 1)[0] for k in stats.keys()]))
            noise_levels = ['0.05', '0.10', '0.20']

            # Effect size matrix
            effect_matrix = np.zeros((len(noise_types), len(noise_levels)))

            for i, noise_type in enumerate(noise_types):
                for j, level in enumerate(noise_levels):
                    key = f"{noise_type}_{level}"
                    if key in stats:
                        effect_matrix[i, j] = abs(stats[key].effect_size)

            # Plot heatmap
            sns.heatmap(effect_matrix, annot=True, fmt='.2f',
                       xticklabels=['5%', '10%', '20%'],
                       yticklabels=noise_types,
                       cmap='YlOrRd', vmin=0, vmax=2,
                       ax=axes[row, col] if len(self.statistical_results) > 2 else axes[col])
            axes[row, col].set_title(f'{model_name}: Effect Size (Cohen\'s d)')
            axes[row, col].set_xlabel('Noise Level')
            axes[row, col].set_ylabel('Noise Type')

        plt.tight_layout()
        plt.savefig('effect_size_heatmap.png', dpi=150, bbox_inches='tight')
        print("\nSaved visualization: effect_size_heatmap.png")

        # Confidence interval plot
        fig, ax = plt.subplots(figsize=(12, 8))

        x_pos = 0
        x_labels = []
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.statistical_results)))

        for model_idx, (model_name, stats) in enumerate(self.statistical_results.items()):
            for test_name, result in stats.items():
                # Plot mean with confidence interval
                ax.errorbar(x_pos, result.mean,
                           yerr=[[result.mean - result.confidence_interval[0]],
                                 [result.confidence_interval[1] - result.mean]],
                           fmt='o', capsize=5, capthick=2,
                           color=colors[model_idx], label=model_name if x_pos == 0 else "")

                # Mark significant results
                if result.significant:
                    ax.scatter(x_pos, result.mean, s=100, marker='*',
                              color='red', alpha=0.5, zorder=10)

                label = test_name.replace('_', '\n')
                x_labels.append(label)
                x_pos += 1

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylabel('Robustness Score')
        ax.set_title('Robustness Scores with 95% Confidence Intervals\n(* indicates significant degradation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig('confidence_intervals.png', dpi=150, bbox_inches='tight')
        print("Saved visualization: confidence_intervals.png")

        # Save comprehensive results
        with open('enhanced_results.json', 'w') as f:
            # Convert StatisticalResult objects to dicts
            serializable_results = {}
            for model_name, results in self.results.items():
                serializable_results[model_name] = {
                    'noise_robustness': results['noise_robustness'],
                    'causal_intervention': results['causal_intervention'],
                    'cross_layer_analysis': results['cross_layer_analysis']
                }
            json.dump(serializable_results, f, indent=2, default=str)
        print("Saved results: enhanced_results.json")


def main():
    """Run enhanced noise robustness experiment"""

    # Extended test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can process natural language effectively.",
        "Recent advances in artificial intelligence have transformed many industries.",
        "Understanding how neural networks handle errors is crucial for robustness.",
        "The researchers analyzed the activation patterns across different layers.",
        "Complex sentences with multiple clauses require sophisticated processing.",
        "Error correction mechanisms emerge naturally during model training.",
        "Attention heads specialize in detecting and fixing input corruptions.",
        "Robustness to noise is an important property of language models.",
        "The experiment revealed interesting patterns in error propagation.",
        "Semantic understanding helps models recover from syntactic errors.",
        "Different types of noise challenge models in unique ways.",
        "Statistical analysis provides insights into model behavior.",
        "Causal intervention verifies the function of neural circuits.",
        "Cross-layer interactions enable complex error correction.",
        "The model's architecture influences its robustness characteristics.",
        "Training data diversity improves noise resistance.",
        "Tokenization strategies affect character-level robustness.",
        "Adversarial perturbations reveal model vulnerabilities.",
        "Information flow through layers shows processing dynamics."
    ]

    # Models to analyze
    model_names = ['bert-base-uncased', 'roberta-base']

    # Run enhanced experiment
    runner = EnhancedExperimentRunner()
    runner.run_comprehensive_experiment(model_names, test_sentences)

    print("\n" + "="*60)
    print("ENHANCED EXPERIMENT COMPLETE")
    print("="*60)
    print("\nFiles generated:")
    print("- enhanced_analysis_report.txt")
    print("- effect_size_heatmap.png")
    print("- confidence_intervals.png")
    print("- enhanced_results.json")


if __name__ == "__main__":
    main()