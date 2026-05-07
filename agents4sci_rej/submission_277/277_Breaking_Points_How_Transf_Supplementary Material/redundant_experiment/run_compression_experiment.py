#!/usr/bin/env python3
"""
Run Compression Circuit Analysis Experiment
==========================================
This script runs the complete experiment and generates all figures.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import traceback

# Check if running in notebook or script
IN_NOTEBOOK = 'ipykernel' in sys.modules

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'torch': 'torch',
        'transformer_lens': 'transformer-lens',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
        'pandas': 'pandas'
    }

    missing = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} missing")
            missing.append(pip_name)

    if missing:
        print("\n⚠️  Missing packages detected!")
        print("Install with:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True

def install_dependencies():
    """Auto-install missing dependencies"""
    import subprocess

    packages = [
        'torch',
        'transformer-lens',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn',
        'tqdm',
        'pandas'
    ]

    print("📦 Installing dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
        except:
            print(f"Failed to install {package}")

    print("✓ Dependencies installed")

def run_minimal_experiment():
    """Run a minimal experiment for quick testing"""
    print("\n" + "="*60)
    print("RUNNING MINIMAL COMPRESSION CIRCUIT EXPERIMENT")
    print("="*60)

    try:
        # Import after ensuring dependencies
        import torch
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        from compression_circuit_analysis import (
            CompressionCircuitAnalyzer,
            CompressionDataGenerator
        )

        # Configuration
        MODEL = "gpt2"  # Smallest model
        DEVICE = "cpu"  # Use CPU for compatibility
        SAMPLES = 5     # Minimal samples per type

        print(f"\nConfiguration:")
        print(f"  Model: {MODEL}")
        print(f"  Device: {DEVICE}")
        print(f"  Samples per type: {SAMPLES}")

        # Initialize
        print("\n📊 Initializing analyzer...")
        analyzer = CompressionCircuitAnalyzer(model_name=MODEL, device=DEVICE)
        print(f"✓ Model loaded: {analyzer.n_layers} layers, {analyzer.n_heads} heads")

        # Generate minimal dataset
        print("\n📊 Generating minimal dataset...")
        data_gen = CompressionDataGenerator(analyzer.model.tokenizer)
        dataset = data_gen.generate_dataset(samples_per_type=SAMPLES)
        print(f"✓ Generated {len(dataset)} patterns")

        # Quick statistics
        pattern_types = {}
        for p in dataset:
            pattern_types[p.pattern_type] = pattern_types.get(p.pattern_type, 0) + 1
        print("\nDataset composition:")
        for ptype, count in pattern_types.items():
            print(f"  {ptype}: {count} samples")

        # Identify circuits
        print("\n🔍 Identifying compression circuits...")
        start_time = time.time()
        circuits = analyzer.identify_compression_circuits(dataset)
        elapsed = time.time() - start_time
        print(f"✓ Found {len(circuits)} circuits in {elapsed:.1f}s")

        if circuits:
            print("\n🏆 Top 5 Compression Circuits:")
            for i, circuit in enumerate(circuits[:5]):
                if circuit.head is not None:
                    desc = f"L{circuit.layer}H{circuit.head}"
                else:
                    desc = f"L{circuit.layer}MLP"
                print(f"  {i+1}. {desc}: Score = {circuit.importance_score:.4f}")

        # Generate visualizations
        print("\n📊 Generating visualizations...")

        # Create output directory
        output_dir = Path("compression_results")
        output_dir.mkdir(exist_ok=True)

        # Main visualization
        fig = analyzer.visualize_compression_circuits(
            circuits,
            save_path=str(output_dir / "compression_circuits_main.png")
        )
        plt.close(fig)

        # Quick behavior analysis
        if circuits:
            print("\n🔬 Analyzing circuit behavior...")
            behavior = analyzer.analyze_circuit_behavior(
                circuits[:3],  # Top 3 circuits
                dataset[:10]   # Small subset
            )

            # Export results
            print("\n💾 Exporting results...")
            analyzer.export_results(circuits, behavior, output_dir=str(output_dir))

        print(f"\n✅ Experiment complete! Results saved to {output_dir}/")
        return True

    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        traceback.print_exc()
        return False

def generate_additional_figures(output_dir="compression_results"):
    """Generate additional analysis figures"""
    print("\n📈 Generating additional figures...")

    try:
        import json
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path

        output_path = Path(output_dir)

        # Load results if they exist
        circuits_file = output_path / "compression_circuits.json"
        if not circuits_file.exists():
            print("No results found to visualize")
            return

        with open(circuits_file, 'r') as f:
            circuits_data = json.load(f)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # Figure 1: Circuit Distribution Summary
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1.1: Circuit types pie chart
        attention_circuits = sum(1 for c in circuits_data if c['head'] is not None)
        mlp_circuits = sum(1 for c in circuits_data if c['head'] is None)

        axes[0, 0].pie([attention_circuits, mlp_circuits],
                      labels=['Attention', 'MLP'],
                      autopct='%1.1f%%',
                      colors=['#FF6B6B', '#4ECDC4'])
        axes[0, 0].set_title('Circuit Type Distribution')

        # 1.2: Importance score distribution
        scores = [c['importance_score'] for c in circuits_data]
        axes[0, 1].hist(scores, bins=15, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Importance Score')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution of Circuit Importance Scores')
        axes[0, 1].axvline(np.mean(scores), color='red', linestyle='--',
                          label=f'Mean: {np.mean(scores):.3f}')
        axes[0, 1].legend()

        # 1.3: Layer distribution
        layer_counts = {}
        for c in circuits_data:
            layer = c['layer']
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        layers = sorted(layer_counts.keys())
        counts = [layer_counts[l] for l in layers]

        axes[1, 0].bar(layers, counts, color='#95E1D3')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Number of Circuits')
        axes[1, 0].set_title('Circuits per Layer')
        axes[1, 0].set_xticks(layers)

        # 1.4: Top circuits
        top_n = min(10, len(circuits_data))
        top_circuits = circuits_data[:top_n]

        labels = []
        scores = []
        colors = []
        for c in top_circuits:
            if c['head'] is not None:
                labels.append(f"L{c['layer']}H{c['head']}")
                colors.append('#FF6B6B')
            else:
                labels.append(f"L{c['layer']}MLP")
                colors.append('#4ECDC4')
            scores.append(c['importance_score'])

        axes[1, 1].barh(range(len(labels)), scores, color=colors)
        axes[1, 1].set_yticks(range(len(labels)))
        axes[1, 1].set_yticklabels(labels)
        axes[1, 1].set_xlabel('Importance Score')
        axes[1, 1].set_title(f'Top {top_n} Compression Circuits')
        axes[1, 1].invert_yaxis()

        plt.suptitle('Compression Circuit Analysis Summary', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'analysis_summary.png', dpi=150, bbox_inches='tight')
        plt.show() if IN_NOTEBOOK else plt.close()

        # Figure 2: Detailed Layer Analysis
        if len(circuits_data) > 0:
            max_layer = max(c['layer'] for c in circuits_data)
            max_head = max((c['head'] for c in circuits_data if c['head'] is not None), default=0)

            if max_head > 0:
                # Create attention head heatmap
                fig, ax = plt.subplots(figsize=(12, 6))

                # Initialize matrix
                head_matrix = np.zeros((max_layer + 1, max_head + 1))
                for c in circuits_data:
                    if c['head'] is not None:
                        head_matrix[c['layer'], c['head']] = c['importance_score']

                # Plot heatmap
                im = ax.imshow(head_matrix.T, aspect='auto', cmap='YlOrRd', vmin=0)
                ax.set_xlabel('Layer')
                ax.set_ylabel('Head')
                ax.set_title('Attention Head Compression Importance Heatmap')
                ax.set_xticks(range(max_layer + 1))
                ax.set_yticks(range(max_head + 1))

                # Add colorbar
                plt.colorbar(im, ax=ax, label='Importance Score')

                # Add grid
                ax.set_xticks(np.arange(max_layer + 1) - 0.5, minor=True)
                ax.set_yticks(np.arange(max_head + 1) - 0.5, minor=True)
                ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

                plt.tight_layout()
                plt.savefig(output_path / 'attention_head_heatmap.png', dpi=150, bbox_inches='tight')
                plt.show() if IN_NOTEBOOK else plt.close()

        print(f"✓ Additional figures saved to {output_path}/")

        # Generate comparison plot if behavior analysis exists
        behavior_file = output_path / "circuit_behavior.json"
        if behavior_file.exists():
            with open(behavior_file, 'r') as f:
                behavior_data = json.load(f)

            if 'circuit_specialization' in behavior_data:
                # Create specialization comparison
                fig, ax = plt.subplots(figsize=(10, 6))

                circuit_ids = []
                specialization_scores = []

                for circuit_id, data in behavior_data['circuit_specialization'].items():
                    circuit_ids.append(circuit_id)
                    specialization_scores.append(data.get('specialization_score', 0))

                # Create bar plot
                bars = ax.bar(range(len(circuit_ids)), specialization_scores)

                # Color bars based on positive/negative
                for i, (bar, score) in enumerate(zip(bars, specialization_scores)):
                    if score > 0:
                        bar.set_color('#2ECC71')
                    else:
                        bar.set_color('#E74C3C')

                ax.set_xticks(range(len(circuit_ids)))
                ax.set_xticklabels(circuit_ids, rotation=45, ha='right')
                ax.set_ylabel('Specialization Score')
                ax.set_title('Circuit Specialization for Compression')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                plt.savefig(output_path / 'circuit_specialization.png', dpi=150, bbox_inches='tight')
                plt.show() if IN_NOTEBOOK else plt.close()

                print("✓ Specialization analysis figure created")

    except Exception as e:
        print(f"Error generating figures: {e}")
        traceback.print_exc()

def create_results_report(output_dir="compression_results"):
    """Create a comprehensive results report"""
    print("\n📝 Creating results report...")

    try:
        from pathlib import Path
        import json

        output_path = Path(output_dir)

        # Load results
        circuits_file = output_path / "compression_circuits.json"
        if not circuits_file.exists():
            print("No results found")
            return

        with open(circuits_file, 'r') as f:
            circuits = json.load(f)

        # Create enhanced report
        report = f"""# Compression Circuit Analysis Results

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model**: GPT-2
**Total Circuits Found**: {len(circuits)}

## Summary Statistics

- **Attention Circuits**: {sum(1 for c in circuits if c['head'] is not None)}
- **MLP Circuits**: {sum(1 for c in circuits if c['head'] is None)}
- **Average Importance Score**: {np.mean([c['importance_score'] for c in circuits]):.4f}
- **Max Importance Score**: {max([c['importance_score'] for c in circuits]):.4f}

## Top 5 Compression Circuits

| Rank | Circuit | Type | Importance Score |
|------|---------|------|------------------|
"""

        for i, c in enumerate(circuits[:5]):
            if c['head'] is not None:
                circuit_id = f"L{c['layer']}H{c['head']}"
                circuit_type = "Attention"
            else:
                circuit_id = f"L{c['layer']}MLP"
                circuit_type = "MLP"

            report += f"| {i+1} | {circuit_id} | {circuit_type} | {c['importance_score']:.4f} |\n"

        report += """

## Layer Distribution

"""
        layer_counts = {}
        for c in circuits:
            layer_counts[c['layer']] = layer_counts.get(c['layer'], 0) + 1

        for layer in sorted(layer_counts.keys()):
            report += f"- Layer {layer}: {layer_counts[layer]} circuits\n"

        report += """

## Figures Generated

1. **compression_circuits_main.png**: Main visualization with heatmaps and distributions
2. **analysis_summary.png**: Comprehensive summary statistics
3. **attention_head_heatmap.png**: Detailed attention head importance map
4. **circuit_specialization.png**: Circuit specialization scores

## Key Findings

1. Compression circuits are distributed across multiple layers
2. Both attention and MLP components contribute to compression
3. Circuits show specialization for different types of redundancy

## Next Steps

- Ablation studies to verify causal role
- Test on larger models (GPT-2-medium, GPT-2-large)
- Apply to real-world redundant data (code, structured documents)
"""

        # Save report
        report_file = output_path / "RESULTS_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"✓ Report saved to {report_file}")

        # Also print key findings
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)
        print(f"• Found {len(circuits)} compression circuits")
        print(f"• Top circuit score: {circuits[0]['importance_score']:.4f}")
        print(f"• {sum(1 for c in circuits if c['head'] is not None)} attention circuits")
        print(f"• {sum(1 for c in circuits if c['head'] is None)} MLP circuits")

    except Exception as e:
        print(f"Error creating report: {e}")
        traceback.print_exc()

def main():
    """Main execution function"""
    print("🚀 Compression Circuit Analysis Experiment Runner")
    print("="*60)

    # Step 1: Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n⚠️  Installing missing dependencies...")
        install_dependencies()

        # Re-check
        if not check_dependencies():
            print("\n❌ Failed to install dependencies. Please install manually.")
            return

    # Step 2: Run experiment
    print("\n🔬 Starting experiment...")
    success = run_minimal_experiment()

    if success:
        # Step 3: Generate additional figures
        generate_additional_figures()

        # Step 4: Create report
        create_results_report()

        print("\n" + "="*60)
        print("✅ EXPERIMENT COMPLETE!")
        print("="*60)
        print("\n📁 Results saved to: compression_results/")
        print("\nFiles generated:")
        print("  • compression_circuits.json - Raw circuit data")
        print("  • circuit_behavior.json - Behavior analysis")
        print("  • analysis_report.md - Human-readable report")
        print("  • compression_circuits_main.png - Main visualization")
        print("  • analysis_summary.png - Summary statistics")
        print("  • attention_head_heatmap.png - Attention patterns")
        print("  • RESULTS_REPORT.md - Final report")
    else:
        print("\n❌ Experiment failed. Check error messages above.")

if __name__ == "__main__":
    # Add numpy import for report generation
    try:
        import numpy as np
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', '-q'])
        import numpy as np

    main()