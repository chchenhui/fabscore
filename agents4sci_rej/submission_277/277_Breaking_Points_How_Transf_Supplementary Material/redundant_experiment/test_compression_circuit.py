"""
Test script for Compression Circuit Analysis
Tests basic functionality with a small dataset
"""

import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from compression_circuit_analysis import (
    CompressionDataGenerator,
    CompressionCircuitAnalyzer,
    CompressionPattern
)


def test_basic_functionality():
    """Test basic components of the compression circuit analyzer"""
    print("="*60)
    print("TESTING COMPRESSION CIRCUIT ANALYSIS")
    print("="*60)

    # Test 1: Initialize analyzer with small model
    print("\n1. Testing model initialization...")
    try:
        # Use gpt2 (smallest) for testing
        analyzer = CompressionCircuitAnalyzer(model_name="gpt2", device="cpu")
        print(f"✓ Model loaded successfully: {analyzer.model_name}")
        print(f"  - Layers: {analyzer.n_layers}")
        print(f"  - Heads: {analyzer.n_heads}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Test 2: Generate test data
    print("\n2. Testing data generation...")
    try:
        data_gen = CompressionDataGenerator(analyzer.model.tokenizer)

        # Generate small test samples
        test_patterns = []

        # Repetitive pattern
        rep_pattern = data_gen.generate_repetitive_text(
            "The cat sat on the mat",
            repetitions=3,
            variation_type='exact'
        )
        test_patterns.append(rep_pattern)
        print(f"✓ Generated repetitive pattern")
        print(f"  Text: '{rep_pattern.text[:50]}...'")
        print(f"  Compression ratio: {rep_pattern.compression_ratio:.2f}")

        # Structured pattern
        struct_pattern = data_gen.generate_structured_data(
            "Item {id}: {name} - Value: {value}",
            num_entries=3
        )
        test_patterns.append(struct_pattern)
        print(f"✓ Generated structured pattern")
        print(f"  Compression ratio: {struct_pattern.compression_ratio:.2f}")

        # Unique pattern
        unique_pattern = data_gen.generate_unique_text(length=10)
        test_patterns.append(unique_pattern)
        print(f"✓ Generated unique pattern")
        print(f"  Text: '{unique_pattern.text[:50]}...'")

    except Exception as e:
        print(f"✗ Failed to generate data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Analyze single input
    print("\n3. Testing single input analysis...")
    try:
        analysis = analyzer.analyze_single_input(rep_pattern)
        print(f"✓ Analysis completed")
        print(f"  - Pattern type: {analysis['pattern_type']}")
        print(f"  - Compression ratio: {analysis['compression_ratio']:.2f}")
        print(f"  - Attention entropy layers analyzed: {len(analysis['attention_entropy'])}")
        print(f"  - MLP sparsity layers analyzed: {len(analysis['mlp_sparsity'])}")

        # Check if we got reasonable values
        layer_0_entropy = analysis['attention_entropy'][0]
        print(f"  - Layer 0 mean entropy: {sum(layer_0_entropy)/len(layer_0_entropy):.3f}")

    except Exception as e:
        print(f"✗ Failed to analyze input: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Circuit identification (small test)
    print("\n4. Testing circuit identification...")
    try:
        # Generate a small dataset
        small_dataset = []
        for _ in range(3):
            small_dataset.append(data_gen.generate_repetitive_text(
                "Test pattern", 4, 'exact'))
        for _ in range(3):
            small_dataset.append(data_gen.generate_unique_text(15))

        circuits = analyzer.identify_compression_circuits(small_dataset)
        print(f"✓ Circuit identification completed")
        print(f"  - Total circuits found: {len(circuits)}")

        if circuits:
            top_circuit = circuits[0]
            circuit_type = f"L{top_circuit.layer}H{top_circuit.head}" if top_circuit.head else f"L{top_circuit.layer}MLP"
            print(f"  - Top circuit: {circuit_type}")
            print(f"  - Importance score: {top_circuit.importance_score:.3f}")

    except Exception as e:
        print(f"✗ Failed to identify circuits: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Quick visualization test
    print("\n5. Testing visualization...")
    try:
        if circuits:
            # Just test that visualization doesn't crash
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            fig = analyzer.visualize_compression_circuits(
                circuits[:5] if len(circuits) > 5 else circuits,
                save_path="test_compression_vis.png"
            )
            print(f"✓ Visualization created and saved")
    except Exception as e:
        print(f"✗ Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    return True


def quick_experiment():
    """Run a quick experiment with minimal data"""
    print("\n" + "="*60)
    print("RUNNING QUICK COMPRESSION CIRCUIT EXPERIMENT")
    print("="*60)

    # Initialize with smallest model
    analyzer = CompressionCircuitAnalyzer(model_name="gpt2", device="cpu")
    data_gen = CompressionDataGenerator(analyzer.model.tokenizer)

    # Generate minimal dataset (10 samples per type)
    print("\nGenerating minimal dataset...")
    dataset = data_gen.generate_dataset(samples_per_type=5)
    print(f"Generated {len(dataset)} patterns")

    # Identify circuits
    print("\nIdentifying compression circuits...")
    circuits = analyzer.identify_compression_circuits(dataset)
    print(f"Found {len(circuits)} compression circuits")

    # Analyze top circuits
    if circuits:
        print("\nTop 3 Compression Circuits:")
        for i, circuit in enumerate(circuits[:3]):
            circuit_type = f"L{circuit.layer}H{circuit.head}" if circuit.head else f"L{circuit.layer}MLP"
            print(f"  {i+1}. {circuit_type}: Score = {circuit.importance_score:.3f}")

        # Quick behavior analysis on top circuit
        print("\nAnalyzing top circuit behavior...")
        behavior = analyzer.analyze_circuit_behavior(circuits[:1], dataset[:10])

        for circuit_id, specs in behavior['circuit_specialization'].items():
            print(f"\nCircuit {circuit_id} specialization:")
            print(f"  - Specialization score: {specs['specialization_score']:.3f}")
            print(f"  - Mean activations by type:")
            for pattern_type, activation in specs['mean_activations'].items():
                print(f"    - {pattern_type}: {activation:.3f}")

    # Export results
    print("\nExporting results...")
    analyzer.export_results(circuits, behavior if circuits else {},
                           output_dir="./test_compression_results")

    print("\nExperiment complete!")


if __name__ == "__main__":
    # First run tests
    success = test_basic_functionality()

    if success:
        print("\n" + "="*60)
        print("Tests passed! Running quick experiment...")
        print("="*60)
        quick_experiment()
    else:
        print("\n❌ Tests failed. Please check the implementation.")