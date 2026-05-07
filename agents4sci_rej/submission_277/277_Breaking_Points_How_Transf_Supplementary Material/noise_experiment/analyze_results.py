"""
Comprehensive Analysis of Noise Robustness Experimental Results
===============================================================
Analyze and compare results from multiple experiments to identify patterns.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats
import glob
import os


class ResultsAnalyzer:
    """Analyze experimental results across multiple runs"""

    def __init__(self):
        self.results = {}
        self.dataframes = {}

    def load_results(self, pattern: str = "*results*.json"):
        """Load all result files matching pattern"""
        files = glob.glob(pattern)
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                name = os.path.basename(file).replace('.json', '')
                self.results[name] = data
                print(f"Loaded: {name}")

    def compare_models(self):
        """Compare model performance across experiments"""
        comparison = {}

        for exp_name, exp_data in self.results.items():
            for model_name, model_data in exp_data.items():
                if model_name not in comparison:
                    comparison[model_name] = {}

                if 'robustness_summary' in model_data:
                    for config, scores in model_data['robustness_summary'].items():
                        key = f"{exp_name}_{config}"
                        comparison[model_name][key] = scores

        return comparison

    def analyze_circuit_patterns(self):
        """Identify consistent circuit patterns across experiments"""
        detection_heads = {}
        correction_layers = {}

        for exp_name, exp_data in self.results.items():
            for model_name, model_data in exp_data.items():
                if model_name not in detection_heads:
                    detection_heads[model_name] = []
                    correction_layers[model_name] = []

                if 'detection_heads' in model_data:
                    for layer, count in model_data['detection_heads'].items():
                        detection_heads[model_name].append((int(layer), count))

                if 'correction_layers' in model_data:
                    correction_layers[model_name].extend(model_data['correction_layers'])

        # Find most common patterns
        patterns = {}
        for model_name in detection_heads:
            # Most active detection layers
            if detection_heads[model_name]:
                layer_counts = {}
                for layer, count in detection_heads[model_name]:
                    layer_counts[layer] = layer_counts.get(layer, 0) + count

                top_detection = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:3]

            # Most common correction layers
            if correction_layers[model_name]:
                from collections import Counter
                correction_freq = Counter(correction_layers[model_name])
                top_correction = correction_freq.most_common(3)

                patterns[model_name] = {
                    'top_detection_layers': top_detection if 'top_detection' in locals() else [],
                    'top_correction_layers': top_correction
                }

        return patterns

    def statistical_meta_analysis(self):
        """Perform meta-analysis across experiments"""
        meta_results = {}

        # Collect all robustness scores by condition
        all_scores = {}
        for exp_name, exp_data in self.results.items():
            for model_name, model_data in exp_data.items():
                if 'robustness_summary' in model_data:
                    for config, scores in model_data['robustness_summary'].items():
                        key = f"{model_name}_{config}"
                        if key not in all_scores:
                            all_scores[key] = []
                        all_scores[key].append(scores['mean'])

        # Calculate meta-statistics
        for key, scores in all_scores.items():
            if scores:
                meta_results[key] = {
                    'pooled_mean': np.mean(scores),
                    'pooled_std': np.std(scores),
                    'n_experiments': len(scores),
                    'ci_95': stats.t.interval(0.95, len(scores)-1,
                                             loc=np.mean(scores),
                                             scale=stats.sem(scores)) if len(scores) > 1 else (scores[0], scores[0])
                }

        return meta_results

    def identify_robust_configurations(self, threshold: float = 0.9):
        """Identify configurations where models maintain high robustness"""
        robust_configs = {}

        for exp_name, exp_data in self.results.items():
            for model_name, model_data in exp_data.items():
                if model_name not in robust_configs:
                    robust_configs[model_name] = []

                if 'robustness_summary' in model_data:
                    for config, scores in model_data['robustness_summary'].items():
                        if scores['mean'] >= threshold:
                            robust_configs[model_name].append({
                                'config': config,
                                'score': scores['mean'],
                                'experiment': exp_name
                            })

        return robust_configs

    def generate_insights(self):
        """Generate key insights from all analyses"""
        insights = []

        # Model comparison
        comparison = self.compare_models()
        best_overall = {}
        for model, configs in comparison.items():
            if configs:
                avg_robustness = np.mean([v['mean'] for v in configs.values() if 'mean' in v])
                best_overall[model] = avg_robustness

        if best_overall:
            best_model = max(best_overall, key=best_overall.get)
            insights.append(f"Most robust model overall: {best_model} (avg: {best_overall[best_model]:.3f})")

        # Circuit patterns
        patterns = self.analyze_circuit_patterns()
        for model, pattern in patterns.items():
            if pattern.get('top_detection_layers'):
                layers = [str(l[0]) for l in pattern['top_detection_layers'][:2]]
                insights.append(f"{model}: Primary detection in layers {', '.join(layers)}")

        # Robust configurations
        robust = self.identify_robust_configurations()
        for model, configs in robust.items():
            if configs:
                best_config = max(configs, key=lambda x: x['score'])
                insights.append(f"{model} maintains {best_config['score']:.3f} robustness with {best_config['config']}")

        # Meta-analysis
        meta = self.statistical_meta_analysis()
        most_consistent = min(meta.items(), key=lambda x: x[1]['pooled_std'] if x[1]['n_experiments'] > 1 else float('inf'))
        if most_consistent[1]['n_experiments'] > 1:
            insights.append(f"Most consistent: {most_consistent[0]} (std: {most_consistent[1]['pooled_std']:.4f})")

        return insights

    def create_comparison_visualizations(self):
        """Create comprehensive comparison visualizations"""
        fig = plt.figure(figsize=(18, 12))

        # 1. Model comparison heatmap
        ax1 = plt.subplot(2, 3, 1)
        comparison = self.compare_models()
        if comparison:
            models = list(comparison.keys())
            configs = list(set(sum([list(m.keys()) for m in comparison.values()], [])))

            matrix = np.zeros((len(models), len(configs)))
            for i, model in enumerate(models):
                for j, config in enumerate(configs):
                    if config in comparison[model] and 'mean' in comparison[model][config]:
                        matrix[i, j] = comparison[model][config]['mean']

            sns.heatmap(matrix, xticklabels=[c.split('_')[-1] for c in configs],
                       yticklabels=models, annot=True, fmt='.3f', cmap='RdYlGn',
                       vmin=0, vmax=1, ax=ax1)
            ax1.set_title('Robustness Comparison Across Experiments')

        # 2. Circuit activation patterns
        ax2 = plt.subplot(2, 3, 2)
        patterns = self.analyze_circuit_patterns()
        if patterns:
            for model, pattern in patterns.items():
                if pattern.get('top_correction_layers'):
                    layers = [l[0] for l in pattern['top_correction_layers']]
                    counts = [l[1] for l in pattern['top_correction_layers']]
                    ax2.bar([f"L{l}" for l in layers], counts, label=model, alpha=0.7)

            ax2.set_xlabel('Layer')
            ax2.set_ylabel('Correction Frequency')
            ax2.set_title('Error Correction Layer Activity')
            ax2.legend()

        # 3. Meta-analysis confidence intervals
        ax3 = plt.subplot(2, 3, 3)
        meta = self.statistical_meta_analysis()
        if meta:
            sorted_meta = sorted(meta.items(), key=lambda x: x[1]['pooled_mean'])
            y_pos = np.arange(len(sorted_meta))

            means = [v[1]['pooled_mean'] for v in sorted_meta]
            ci_lower = [v[1]['ci_95'][0] if v[1]['n_experiments'] > 1 else v[1]['pooled_mean']
                       for v in sorted_meta]
            ci_upper = [v[1]['ci_95'][1] if v[1]['n_experiments'] > 1 else v[1]['pooled_mean']
                       for v in sorted_meta]
            errors = [[means[i] - ci_lower[i] for i in range(len(means))],
                     [ci_upper[i] - means[i] for i in range(len(means))]]

            ax3.barh(y_pos, means, xerr=errors, capsize=3)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels([k.split('_')[-1] for k, v in sorted_meta], fontsize=8)
            ax3.set_xlabel('Pooled Robustness Score')
            ax3.set_title('Meta-Analysis: Pooled Results')
            ax3.set_xlim([0, 1])

        # 4. Robustness distribution
        ax4 = plt.subplot(2, 3, 4)
        all_robustness = []
        labels = []
        for exp_name, exp_data in self.results.items():
            for model_name, model_data in exp_data.items():
                if 'robustness_summary' in model_data:
                    scores = [v['mean'] for v in model_data['robustness_summary'].values()]
                    all_robustness.append(scores)
                    labels.append(f"{model_name[:4]}")

        if all_robustness:
            ax4.violinplot(all_robustness, positions=range(len(all_robustness)),
                          showmeans=True, showmedians=True)
            ax4.set_xticks(range(len(labels)))
            ax4.set_xticklabels(labels, rotation=45)
            ax4.set_ylabel('Robustness Score')
            ax4.set_title('Robustness Distribution by Model')
            ax4.set_ylim([0, 1.1])

        # 5. Noise type impact
        ax5 = plt.subplot(2, 3, 5)
        noise_impact = {}
        for exp_name, exp_data in self.results.items():
            for model_name, model_data in exp_data.items():
                if 'robustness_summary' in model_data:
                    for config, scores in model_data['robustness_summary'].items():
                        noise_type = config.split('_')[0]
                        if noise_type not in noise_impact:
                            noise_impact[noise_type] = []
                        noise_impact[noise_type].append(scores['mean'])

        if noise_impact:
            noise_types = list(noise_impact.keys())
            avg_impact = [np.mean(noise_impact[nt]) for nt in noise_types]
            std_impact = [np.std(noise_impact[nt]) for nt in noise_types]

            ax5.bar(noise_types, avg_impact, yerr=std_impact, capsize=5)
            ax5.set_ylabel('Average Robustness')
            ax5.set_title('Impact by Noise Type')
            ax5.set_ylim([0, 1.1])

        # 6. Key insights
        ax6 = plt.subplot(2, 3, 6)
        insights = self.generate_insights()
        ax6.axis('off')
        insights_text = '\n\n'.join(insights[:6])  # Show top 6 insights
        ax6.text(0.1, 0.9, 'KEY INSIGHTS', fontsize=14, fontweight='bold',
                transform=ax6.transAxes)
        ax6.text(0.1, 0.1, insights_text, fontsize=10, wrap=True,
                transform=ax6.transAxes, verticalalignment='bottom')

        plt.suptitle('Comprehensive Noise Robustness Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=150, bbox_inches='tight')
        print("\nSaved: comprehensive_analysis.png")

    def save_analysis_report(self):
        """Save comprehensive analysis report"""
        report = []
        report.append("="*60)
        report.append("COMPREHENSIVE ANALYSIS REPORT")
        report.append("="*60)

        report.append("\n## Model Comparison")
        comparison = self.compare_models()
        for model, configs in comparison.items():
            report.append(f"\n### {model}")
            for config, scores in configs.items():
                if 'mean' in scores:
                    report.append(f"  {config}: {scores['mean']:.4f} ± {scores.get('std', 0):.4f}")

        report.append("\n## Circuit Patterns")
        patterns = self.analyze_circuit_patterns()
        for model, pattern in patterns.items():
            report.append(f"\n### {model}")
            if pattern.get('top_detection_layers'):
                report.append(f"  Detection layers: {pattern['top_detection_layers']}")
            if pattern.get('top_correction_layers'):
                report.append(f"  Correction layers: {pattern['top_correction_layers']}")

        report.append("\n## Meta-Analysis")
        meta = self.statistical_meta_analysis()
        for key, stats in sorted(meta.items(), key=lambda x: x[1]['pooled_mean'], reverse=True)[:10]:
            report.append(f"\n{key}:")
            report.append(f"  Pooled mean: {stats['pooled_mean']:.4f}")
            report.append(f"  N experiments: {stats['n_experiments']}")
            if stats['n_experiments'] > 1:
                report.append(f"  95% CI: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")

        report.append("\n## Key Insights")
        for i, insight in enumerate(self.generate_insights(), 1):
            report.append(f"{i}. {insight}")

        with open('comprehensive_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        print("Saved: comprehensive_analysis_report.txt")


def main():
    """Run comprehensive analysis"""
    analyzer = ResultsAnalyzer()

    # Load all results
    print("Loading experimental results...")
    analyzer.load_results()

    if not analyzer.results:
        print("No results found. Please run experiments first.")
        return

    # Perform analyses
    print("\nPerforming meta-analysis...")
    meta_results = analyzer.statistical_meta_analysis()
    print(f"Analyzed {len(meta_results)} configurations")

    print("\nIdentifying circuit patterns...")
    patterns = analyzer.analyze_circuit_patterns()
    print(f"Found patterns for {len(patterns)} models")

    print("\nIdentifying robust configurations...")
    robust = analyzer.identify_robust_configurations()
    total_robust = sum(len(configs) for configs in robust.values())
    print(f"Found {total_robust} robust configurations")

    # Generate visualizations
    print("\nCreating visualizations...")
    analyzer.create_comparison_visualizations()

    # Save report
    print("\nGenerating report...")
    analyzer.save_analysis_report()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("- comprehensive_analysis.png")
    print("- comprehensive_analysis_report.txt")


if __name__ == "__main__":
    main()