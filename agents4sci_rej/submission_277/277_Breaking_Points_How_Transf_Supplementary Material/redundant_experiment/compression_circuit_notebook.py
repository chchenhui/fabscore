"""
Interactive Jupyter Notebook for Compression Circuit Analysis
=============================================================

This notebook provides an interactive interface for exploring compression circuits
in transformer models. Run each cell sequentially to conduct the analysis.

To use:
1. Install requirements: pip install -r requirements_compression.txt
2. Run in Jupyter: jupyter notebook compression_circuit_notebook.ipynb
   OR convert this to notebook: jupytext --to notebook compression_circuit_notebook.py
"""

# %% [markdown]
# # Compression Circuit Analysis: Interactive Exploration
#
# This notebook allows you to interactively explore how transformer models handle redundant information.

# %% Setup and Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

# Import our analysis modules
from compression_circuit_analysis import (
    CompressionDataGenerator,
    CompressionCircuitAnalyzer,
    CompressionPattern
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Setup complete!")

# %% [markdown]
# ## 1. Initialize Model and Analyzer
#
# We'll start with GPT-2 small for efficiency. You can change to larger models if you have more compute.

# %% Initialize Model
# Choose model size based on available memory
MODEL_OPTIONS = {
    'tiny': 'gpt2',  # 124M parameters
    'small': 'gpt2-small',  # Same as gpt2
    'medium': 'gpt2-medium',  # 355M parameters
    'large': 'gpt2-large',  # 774M parameters
}

# Select model
model_size = 'tiny'  # Change this based on your hardware
model_name = MODEL_OPTIONS[model_size]

# For M3 Max, you can use 'mps' device
device = 'cpu'  # Change to 'mps' on Mac or 'cuda' on NVIDIA

print(f"Loading {model_name}...")
analyzer = CompressionCircuitAnalyzer(model_name=model_name, device=device)
print(f"✅ Model loaded: {analyzer.n_layers} layers, {analyzer.n_heads} heads")

# %% [markdown]
# ## 2. Generate Test Dataset
#
# We'll create various types of redundant and unique patterns to test the model's compression behavior.

# %% Generate Dataset
data_gen = CompressionDataGenerator(analyzer.model.tokenizer)

# Generate different pattern types
print("Generating test patterns...")

# Parameters for dataset generation
SAMPLES_PER_TYPE = 20  # Increase for more robust analysis

dataset = []

# 1. Exact repetition
print("  - Generating exact repetitions...")
for i in range(SAMPLES_PER_TYPE):
    base_texts = [
        "The quick brown fox",
        "Machine learning is powerful",
        "Data reveals patterns",
        "Neural networks compute"
    ]
    pattern = data_gen.generate_repetitive_text(
        np.random.choice(base_texts),
        repetitions=np.random.randint(3, 6),
        variation_type='exact'
    )
    dataset.append(pattern)

# 2. Semantic repetition
print("  - Generating semantic repetitions...")
for i in range(SAMPLES_PER_TYPE):
    pattern = data_gen.generate_repetitive_text(
        "The result is good",
        repetitions=4,
        variation_type='semantic'
    )
    dataset.append(pattern)

# 3. Structured data
print("  - Generating structured patterns...")
for i in range(SAMPLES_PER_TYPE):
    pattern = data_gen.generate_structured_data(
        "Entry {id}: {name} [{category}] = {value}",
        num_entries=np.random.randint(5, 10)
    )
    dataset.append(pattern)

# 4. Unique content
print("  - Generating unique content...")
for i in range(SAMPLES_PER_TYPE):
    pattern = data_gen.generate_unique_text(
        length=np.random.randint(20, 40)
    )
    dataset.append(pattern)

print(f"✅ Generated {len(dataset)} test patterns")

# Display sample patterns
print("\n📊 Sample patterns:")
for pattern_type in ['repetition', 'structure', 'unique']:
    sample = next(p for p in dataset if p.pattern_type == pattern_type)
    print(f"\n{pattern_type.upper()}:")
    print(f"  Text: '{sample.text[:100]}...'")
    print(f"  Compression ratio: {sample.compression_ratio:.2f}")

# %% [markdown]
# ## 3. Analyze Individual Patterns
#
# Let's analyze how the model processes different types of patterns.

# %% Analyze Patterns
# Analyze a few sample patterns in detail
sample_analyses = []

print("Analyzing sample patterns...")
for i, pattern in enumerate(dataset[:5]):
    print(f"  Pattern {i+1}/{5}: {pattern.pattern_type}")
    analysis = analyzer.analyze_single_input(pattern)
    sample_analyses.append(analysis)

# Visualize attention entropy across layers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Attention entropy by layer
for analysis in sample_analyses:
    pattern_type = analysis['pattern_type']
    mean_entropies = [np.mean(heads) for heads in analysis['attention_entropy'].values()]
    axes[0].plot(mean_entropies, label=pattern_type, marker='o', alpha=0.7)

axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Mean Attention Entropy')
axes[0].set_title('Attention Entropy Across Layers')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: MLP sparsity by layer
for analysis in sample_analyses:
    pattern_type = analysis['pattern_type']
    sparsities = list(analysis['mlp_sparsity'].values())
    axes[1].plot(sparsities, label=pattern_type, marker='s', alpha=0.7)

axes[1].set_xlabel('Layer')
axes[1].set_ylabel('MLP Sparsity')
axes[1].set_title('MLP Activation Sparsity Across Layers')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Identify Compression Circuits
#
# Now we'll identify which circuits (attention heads and MLP layers) are specifically involved in processing redundant information.

# %% Identify Circuits
print("🔍 Identifying compression circuits...")
circuits = analyzer.identify_compression_circuits(dataset)
print(f"✅ Found {len(circuits)} compression-related circuits")

# Display top circuits
print("\n🏆 Top 10 Compression Circuits:")
print("-" * 50)
for i, circuit in enumerate(circuits[:10]):
    if circuit.head is not None:
        circuit_id = f"Layer {circuit.layer}, Head {circuit.head}"
        circuit_type = "Attention"
    else:
        circuit_id = f"Layer {circuit.layer}"
        circuit_type = "MLP"

    print(f"{i+1:2d}. {circuit_type:9s} | {circuit_id:20s} | Score: {circuit.importance_score:.4f}")

# %% [markdown]
# ## 5. Visualize Circuit Distribution

# %% Visualize Circuits
# Create comprehensive visualization
fig = analyzer.visualize_compression_circuits(circuits, save_path="compression_circuits_notebook.png")

# Additional custom visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Circuit type distribution
circuit_types = ['Attention' if c.head is not None else 'MLP' for c in circuits]
type_counts = {t: circuit_types.count(t) for t in set(circuit_types)}
axes[0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
axes[0].set_title('Circuit Type Distribution')

# 2. Layer distribution heatmap
layer_dist = np.zeros(analyzer.n_layers)
for circuit in circuits:
    layer_dist[circuit.layer] += circuit.importance_score

axes[1].bar(range(analyzer.n_layers), layer_dist)
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Total Importance Score')
axes[1].set_title('Compression Importance by Layer')

# 3. Head importance matrix (for attention circuits only)
head_matrix = np.zeros((analyzer.n_layers, analyzer.n_heads))
for circuit in circuits:
    if circuit.head is not None:
        head_matrix[circuit.layer, circuit.head] = circuit.importance_score

im = axes[2].imshow(head_matrix.T, aspect='auto', cmap='YlOrRd')
axes[2].set_xlabel('Layer')
axes[2].set_ylabel('Head')
axes[2].set_title('Attention Head Compression Importance')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Detailed Circuit Behavior Analysis
#
# Let's analyze how the top compression circuits behave on different types of input.

# %% Circuit Behavior
print("🔬 Analyzing circuit behavior...")
behavior_analysis = analyzer.analyze_circuit_behavior(
    circuits[:5],  # Analyze top 5 circuits
    dataset[:20]   # Use subset for speed
)

# Display specialization scores
print("\n📊 Circuit Specialization Analysis:")
print("=" * 60)

for circuit_id, specs in behavior_analysis['circuit_specialization'].items():
    print(f"\n{circuit_id}:")
    print(f"  Specialization Score: {specs['specialization_score']:.3f}")
    print("  Mean Activations by Pattern Type:")

    # Sort by activation strength
    sorted_activations = sorted(
        specs['mean_activations'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for pattern_type, activation in sorted_activations:
        if activation > 0:
            print(f"    - {pattern_type:20s}: {activation:.4f}")

# %% [markdown]
# ## 7. Interactive Circuit Explorer
#
# Use this section to explore specific circuits in detail.

# %% Interactive Explorer
def explore_circuit(circuit_idx=0):
    """Interactively explore a specific circuit"""
    if circuit_idx >= len(circuits):
        print(f"Circuit index {circuit_idx} out of range. Max: {len(circuits)-1}")
        return

    circuit = circuits[circuit_idx]

    # Circuit info
    if circuit.head is not None:
        circuit_name = f"Layer {circuit.layer}, Head {circuit.head} (Attention)"
    else:
        circuit_name = f"Layer {circuit.layer} (MLP)"

    print(f"🔍 Exploring Circuit: {circuit_name}")
    print(f"   Importance Score: {circuit.importance_score:.4f}")

    # Test on specific patterns
    test_patterns = {
        'exact_rep': data_gen.generate_repetitive_text("Test pattern", 5, 'exact'),
        'semantic_rep': data_gen.generate_repetitive_text("Good result", 4, 'semantic'),
        'unique': data_gen.generate_unique_text(30)
    }

    print("\n📊 Circuit Response to Test Patterns:")
    for pattern_name, pattern in test_patterns.items():
        tokens = torch.tensor(pattern.tokens).unsqueeze(0).to(analyzer.device)
        logits, cache = analyzer.model.run_with_cache(tokens)

        if circuit.head is not None:
            activation = cache["pattern", circuit.layer][0, circuit.head]
            response = activation.mean().item()
        else:
            activation = cache["mlp_out", circuit.layer]
            response = activation.abs().mean().item()

        print(f"  {pattern_name:15s}: {response:.4f}")

    # Visualize activation pattern for repetitive input
    if circuit.head is not None:
        tokens = torch.tensor(test_patterns['exact_rep'].tokens).unsqueeze(0).to(analyzer.device)
        logits, cache = analyzer.model.run_with_cache(tokens)
        pattern = cache["pattern", circuit.layer][0, circuit.head].cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.imshow(pattern, aspect='auto', cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title(f'Attention Pattern for {circuit_name}')
        plt.show()

# Explore top circuit
explore_circuit(0)

# %% [markdown]
# ## 8. Export Results and Generate Report

# %% Export Results
print("💾 Exporting results...")

# Export to files
output_dir = "./compression_analysis_notebook_results"
analyzer.export_results(circuits, behavior_analysis, output_dir=output_dir)

# Generate summary statistics
summary = {
    "model": model_name,
    "device": device,
    "dataset_size": len(dataset),
    "total_circuits_found": len(circuits),
    "top_circuit_score": circuits[0].importance_score if circuits else 0,
    "attention_circuits": sum(1 for c in circuits if c.head is not None),
    "mlp_circuits": sum(1 for c in circuits if c.head is None),
    "pattern_types_analyzed": list(set(p.pattern_type for p in dataset))
}

print("\n📈 Summary Statistics:")
for key, value in summary.items():
    print(f"  {key}: {value}")

# Save summary
with open(Path(output_dir) / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Results exported to {output_dir}/")

# %% [markdown]
# ## 9. Conclusions and Next Steps
#
# ### Key Findings:
# 1. **Circuit Distribution**: Compression circuits are distributed across multiple layers
# 2. **Specialization**: Different circuits specialize in different types of redundancy
# 3. **Layer Patterns**: Early layers focus on local patterns, later layers on global structure
#
# ### Next Steps:
# 1. **Ablation Studies**: Test causal role of identified circuits
# 2. **Transfer Analysis**: Check if circuits transfer to other models
# 3. **Intervention**: Try to enhance/suppress compression behavior
# 4. **Scaling**: Test on larger models and datasets

# %% Next Steps Code
print("""
🚀 Suggested Next Experiments:

1. Ablation Study:
   - Disable top compression circuits
   - Measure impact on model performance

2. Circuit Enhancement:
   - Amplify compression circuit activations
   - Test if this improves handling of redundant input

3. Cross-Model Analysis:
   - Compare circuits across GPT-2 sizes
   - Look for universal compression patterns

4. Real-World Applications:
   - Test on code (high redundancy)
   - Test on poetry (repetitive structure)
   - Test on data tables (structured redundancy)

Run additional experiments by modifying the notebook parameters above!
""")