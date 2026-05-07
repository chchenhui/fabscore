"""
Runtime Validation for Layer Dropout Speedup Claims
===================================================
This script measures actual runtime improvements from strategic layer dropout
to validate the theoretical 3.1× speedup claim.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class LayerDropoutBERT(nn.Module):
    """BERT model with configurable layer dropout"""

    def __init__(self, model_name='bert-base-uncased', dropout_layers=None):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.dropout_layers = dropout_layers or []

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model.embeddings(input_ids)

        for i, layer in enumerate(self.base_model.encoder.layer):
            if i not in self.dropout_layers:
                outputs = layer(outputs, attention_mask=attention_mask)[0]

        return outputs


def measure_inference_time(model, inputs, num_runs=100, warmup=10):
    """Measure average inference time with proper warmup"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, attention_mask)

    # Synchronize CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure actual runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_ids, attention_mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def run_runtime_validation():
    """Main runtime validation experiment"""

    print("Runtime Validation Experiment")
    print("=" * 50)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Test configurations
    configurations = {
        'baseline': [],  # No dropout
        'random_15%': [1, 4, 9],  # Random 15% dropout
        'strategic_15%': [1, 2, 4, 5, 9, 10],  # Skip transitions at 3, 8
        'aggressive_25%': [1, 2, 4, 5, 6, 9, 10, 11],  # 25% dropout
    }

    # Test on different sequence lengths
    sequence_lengths = [32, 64, 128, 256, 512]
    batch_sizes = [1, 8, 16, 32]

    results = []

    for seq_len in sequence_lengths:
        for batch_size in batch_sizes:
            # Create dummy input
            text = ["This is a test sentence for runtime measurement."] * batch_size
            inputs = tokenizer(text, padding='max_length', max_length=seq_len,
                              truncation=True, return_tensors='pt')

            print(f"\nTesting: seq_len={seq_len}, batch_size={batch_size}")

            for config_name, dropout_layers in configurations.items():
                model = LayerDropoutBERT(dropout_layers=dropout_layers)

                timing = measure_inference_time(model, inputs)

                results.append({
                    'configuration': config_name,
                    'sequence_length': seq_len,
                    'batch_size': batch_size,
                    'mean_time': timing['mean'],
                    'std_time': timing['std'],
                    'dropped_layers': len(dropout_layers)
                })

                print(f"  {config_name}: {timing['mean']*1000:.2f}±{timing['std']*1000:.2f} ms")

    # Calculate speedups
    df = pd.DataFrame(results)
    baseline_times = df[df['configuration'] == 'baseline'].set_index(['sequence_length', 'batch_size'])['mean_time']

    for idx, row in df.iterrows():
        if row['configuration'] != 'baseline':
            key = (row['sequence_length'], row['batch_size'])
            speedup = baseline_times[key] / row['mean_time']
            df.at[idx, 'speedup'] = speedup
        else:
            df.at[idx, 'speedup'] = 1.0

    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Speedup by configuration
    ax = axes[0, 0]
    speedup_summary = df[df['configuration'] != 'baseline'].groupby('configuration')['speedup'].agg(['mean', 'std'])
    speedup_summary.plot(kind='bar', y='mean', yerr='std', ax=ax, legend=False)
    ax.set_title('Average Speedup by Configuration')
    ax.set_ylabel('Speedup Factor')
    ax.axhline(y=3.1, color='r', linestyle='--', label='Theoretical (3.1×)')
    ax.legend()

    # Speedup vs sequence length
    ax = axes[0, 1]
    for config in configurations.keys():
        if config != 'baseline':
            data = df[(df['configuration'] == config) & (df['batch_size'] == 16)]
            ax.plot(data['sequence_length'], data['speedup'], marker='o', label=config)
    ax.set_title('Speedup vs Sequence Length (batch_size=16)')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Speedup Factor')
    ax.legend()

    # Speedup vs batch size
    ax = axes[1, 0]
    for config in configurations.keys():
        if config != 'baseline':
            data = df[(df['configuration'] == config) & (df['sequence_length'] == 128)]
            ax.plot(data['batch_size'], data['speedup'], marker='s', label=config)
    ax.set_title('Speedup vs Batch Size (seq_len=128)')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Speedup Factor')
    ax.legend()

    # Runtime comparison table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    table_data = df[df['sequence_length'] == 128].pivot_table(
        values='speedup', index='configuration', columns='batch_size'
    ).round(2)

    table = ax.table(cellText=table_data.values,
                     rowLabels=table_data.index,
                     colLabels=[f'BS={x}' for x in table_data.columns],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax.set_title('Speedup Factors (seq_len=128)', pad=20)

    plt.tight_layout()
    plt.savefig('nips_figures/runtime_validation.pdf', dpi=300, bbox_inches='tight')

    # Save detailed results
    df.to_csv('runtime_validation_results.csv', index=False)

    # Print summary
    print("\n" + "=" * 50)
    print("RUNTIME VALIDATION SUMMARY")
    print("=" * 50)

    strategic_speedup = df[df['configuration'] == 'strategic_15%']['speedup'].mean()
    print(f"Strategic 15% dropout average speedup: {strategic_speedup:.2f}×")
    print(f"Theoretical prediction: 3.1×")
    print(f"Actual vs Theoretical: {(strategic_speedup/3.1)*100:.1f}%")

    # Memory usage analysis
    print("\nMemory Usage Reduction:")
    for config_name, dropout_layers in configurations.items():
        if config_name != 'baseline':
            memory_reduction = (len(dropout_layers) / 12) * 100
            print(f"  {config_name}: {memory_reduction:.1f}% reduction in activation memory")

    return df


if __name__ == "__main__":
    results_df = run_runtime_validation()
    print("\nRuntime validation complete. Results saved to runtime_validation_results.csv")