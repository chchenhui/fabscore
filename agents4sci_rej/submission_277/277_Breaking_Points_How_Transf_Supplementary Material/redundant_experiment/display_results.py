#!/usr/bin/env python3
"""
Display and summarize compression circuit analysis results
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def display_results():
    """Display all generated figures and summarize findings"""

    results_dir = Path("compression_results")

    print("="*70)
    print("COMPRESSION CIRCUIT ANALYSIS - RESULTS SUMMARY")
    print("="*70)

    # Load and display circuit data
    with open(results_dir / "compression_circuits.json", 'r') as f:
        circuits = json.load(f)

    print(f"\n📊 STATISTICS:")
    print(f"  • Total circuits found: {len(circuits)}")
    print(f"  • Attention circuits: {sum(1 for c in circuits if c['head'] is not None)}")
    print(f"  • MLP circuits: {sum(1 for c in circuits if c['head'] is None)}")

    scores = [c['importance_score'] for c in circuits]
    print(f"  • Average importance: {np.mean(scores):.4f}")
    print(f"  • Maximum importance: {max(scores):.4f}")
    print(f"  • Minimum importance: {min(scores):.4f}")

    # Top circuits
    print(f"\n🏆 TOP 10 COMPRESSION CIRCUITS:")
    print("-"*50)
    print(f"{'Rank':<6} {'Circuit':<12} {'Type':<10} {'Score':<10}")
    print("-"*50)

    for i, circuit in enumerate(circuits[:10], 1):
        if circuit['head'] is not None:
            circuit_id = f"L{circuit['layer']}H{circuit['head']}"
            circuit_type = "Attention"
        else:
            circuit_id = f"L{circuit['layer']}MLP"
            circuit_type = "MLP"

        print(f"{i:<6} {circuit_id:<12} {circuit_type:<10} {circuit['importance_score']:.4f}")

    # Layer distribution
    print(f"\n📍 LAYER DISTRIBUTION:")
    layer_counts = {}
    layer_scores = {}

    for c in circuits:
        layer = c['layer']
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if layer not in layer_scores:
            layer_scores[layer] = []
        layer_scores[layer].append(c['importance_score'])

    for layer in sorted(layer_counts.keys()):
        avg_score = np.mean(layer_scores[layer])
        print(f"  Layer {layer:2d}: {layer_counts[layer]:2d} circuits (avg score: {avg_score:.3f})")

    # Load behavior analysis
    with open(results_dir / "circuit_behavior.json", 'r') as f:
        behavior = json.load(f)

    if 'circuit_specialization' in behavior:
        print(f"\n🔬 CIRCUIT SPECIALIZATION:")
        print("-"*50)

        for circuit_id, data in behavior['circuit_specialization'].items():
            spec_score = data.get('specialization_score', 0)
            print(f"\n{circuit_id}:")
            print(f"  Specialization score: {spec_score:.3f}")

            if 'mean_activations' in data:
                print("  Mean activations by pattern type:")
                sorted_acts = sorted(data['mean_activations'].items(),
                                   key=lambda x: x[1], reverse=True)
                for pattern_type, activation in sorted_acts:
                    if activation > 0:
                        print(f"    • {pattern_type:20s}: {activation:.4f}")

    # Display figure information
    print(f"\n📈 GENERATED FIGURES:")
    print("-"*50)

    figures = [
        ("compression_circuits_main.png", "Main visualization with heatmaps"),
        ("analysis_summary.png", "Comprehensive summary statistics"),
        ("attention_head_heatmap.png", "Attention head importance map"),
        ("circuit_specialization.png", "Circuit specialization scores")
    ]

    for fig_name, description in figures:
        fig_path = results_dir / fig_name
        if fig_path.exists():
            # Get image dimensions
            with Image.open(fig_path) as img:
                width, height = img.size
            print(f"  ✓ {fig_name:<35s} ({width}x{height})")
            print(f"    {description}")
        else:
            print(f"  ✗ {fig_name} - Not found")

    print(f"\n💡 KEY INSIGHTS:")
    print("-"*50)

    # Analyze patterns
    early_layers = sum(1 for c in circuits if c['layer'] < 4)
    middle_layers = sum(1 for c in circuits if 4 <= c['layer'] < 8)
    late_layers = sum(1 for c in circuits if c['layer'] >= 8)

    print(f"1. Circuit distribution across network depth:")
    print(f"   • Early layers (0-3): {early_layers} circuits ({early_layers/len(circuits)*100:.1f}%)")
    print(f"   • Middle layers (4-7): {middle_layers} circuits ({middle_layers/len(circuits)*100:.1f}%)")
    print(f"   • Late layers (8-11): {late_layers} circuits ({late_layers/len(circuits)*100:.1f}%)")

    # Head analysis
    head_counts = {}
    for c in circuits:
        if c['head'] is not None:
            head = c['head']
            head_counts[head] = head_counts.get(head, 0) + 1

    if head_counts:
        most_active_head = max(head_counts, key=head_counts.get)
        print(f"\n2. Most active attention head: Head {most_active_head} ({head_counts[most_active_head]} occurrences)")

    # Score analysis
    high_score_circuits = [c for c in circuits if c['importance_score'] > 0.8]
    print(f"\n3. High-importance circuits (score > 0.8): {len(high_score_circuits)} circuits")

    if high_score_circuits:
        avg_layer = np.mean([c['layer'] for c in high_score_circuits])
        print(f"   • Average layer of high-importance circuits: {avg_layer:.1f}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    # Create combined figure display
    print("\n📊 Creating combined results visualization...")

    fig = plt.figure(figsize=(20, 12))

    # Load and display the four main figures
    fig_files = [
        "compression_circuits_main.png",
        "analysis_summary.png",
        "attention_head_heatmap.png",
        "circuit_specialization.png"
    ]

    for i, fig_file in enumerate(fig_files, 1):
        fig_path = results_dir / fig_file
        if fig_path.exists():
            ax = fig.add_subplot(2, 2, i)
            img = Image.open(fig_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(fig_file.replace('_', ' ').replace('.png', '').title(),
                        fontsize=12, pad=10)

    plt.suptitle('Compression Circuit Analysis Results', fontsize=16, y=0.98)
    plt.tight_layout()

    # Save combined figure
    combined_path = results_dir / "combined_results.png"
    plt.savefig(combined_path, dpi=100, bbox_inches='tight')
    print(f"✓ Combined visualization saved to {combined_path}")

    plt.show()

if __name__ == "__main__":
    display_results()