"""
Minimal verification script for Claim 146:
Strategic dropout achieves 1.28x average speedup, approaching theoretical 1.33x max.
"""
import sys
import torch
import torch.nn as nn
import time
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModel, BertModel
except ImportError:
    print("ERROR: transformers not available")
    sys.exit(1)


class LayerDropoutBERT(nn.Module):
    """BERT model with configurable layer dropout - fixed for newer transformers"""
    def __init__(self, model_name='bert-base-uncased', dropout_layers=None):
        super().__init__()
        self.base_model = BertModel.from_pretrained(model_name)
        self.dropout_layers = dropout_layers or []

    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        embedding_output = self.base_model.embeddings(input_ids)

        # Extend attention mask properly
        extended_mask = None
        if attention_mask is not None:
            extended_mask = self.base_model.get_extended_attention_mask(
                attention_mask, input_ids.shape
            )

        # Run through layers with selective dropout
        hidden_states = embedding_output
        for i, layer in enumerate(self.base_model.encoder.layer):
            if i not in self.dropout_layers:
                layer_outputs = layer(hidden_states, attention_mask=extended_mask)
                hidden_states = layer_outputs[0]

        return hidden_states


def measure_inference_time(model, inputs, num_runs=20, warmup=5):
    """Measure average inference time"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, attention_mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_ids, attention_mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    return np.mean(times)


def main():
    print("=" * 60)
    print("Claim 146 Verification: Strategic Layer Dropout Speedup")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Configurations from runtime_validation.py
    configurations = {
        'baseline': [],               # No dropout (12/12 layers)
        'random_15%': [1, 4, 9],    # 3 layers dropped (theoretical 12/9 = 1.33x)
        'strategic_15%': [1, 2, 4, 5, 9, 10],  # 6 layers dropped (theoretical 12/6 = 2x)
        'aggressive_25%': [1, 2, 4, 5, 6, 9, 10, 11],  # 8 layers dropped (theoretical 12/4 = 3x)
    }

    # Test with batch sizes as in the paper (Figure 1 right plot)
    batch_sizes = [1, 8, 16, 32]
    seq_len = 128

    results = {}

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}, seq_len: {seq_len}")
        text = ["This is a test sentence for runtime measurement."] * batch_size
        inputs = tokenizer(text, padding='max_length', max_length=seq_len,
                          truncation=True, return_tensors='pt')

        results[batch_size] = {}
        for config_name, dropout_layers in configurations.items():
            print(f"  Testing {config_name} ({len(dropout_layers)} layers dropped)...", flush=True)
            model = LayerDropoutBERT(dropout_layers=dropout_layers)
            mean_time = measure_inference_time(model, inputs)
            results[batch_size][config_name] = mean_time
            print(f"    Mean time: {mean_time*1000:.2f} ms")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SPEEDUP RESULTS")
    print("=" * 60)

    for batch_size in batch_sizes:
        baseline_time = results[batch_size]['baseline']
        print(f"\nBatch size {batch_size}:")
        for config_name in ['random_15%', 'strategic_15%', 'aggressive_25%']:
            speedup = baseline_time / results[batch_size][config_name]
            print(f"  {config_name}: {speedup:.3f}x speedup")

    # Calculate averages
    print("\n" + "=" * 60)
    print("AVERAGE SPEEDUPS (across batch sizes)")
    print("=" * 60)

    all_speedups = {}
    for config_name in ['random_15%', 'strategic_15%', 'aggressive_25%']:
        speedups = []
        for batch_size in batch_sizes:
            baseline_time = results[batch_size]['baseline']
            speedup = baseline_time / results[batch_size][config_name]
            speedups.append(speedup)
        avg_speedup = np.mean(speedups)
        max_speedup = np.max(speedups)
        all_speedups[config_name] = speedups
        print(f"  {config_name}: avg={avg_speedup:.3f}x, max={max_speedup:.3f}x (batch speedups: {[f'{s:.3f}' for s in speedups]})")

    # Key claim check
    strategic_avg = np.mean(all_speedups['strategic_15%'])
    strategic_max = np.max(all_speedups['strategic_15%'])
    random_avg = np.mean(all_speedups['random_15%'])
    random_max = np.max(all_speedups['random_15%'])

    print("\n" + "=" * 60)
    print("CLAIM 146 VERIFICATION")
    print("Paper: strategic dropout achieves 1.28x avg speedup, theoretical 1.33x max")
    print()
    print(f"  strategic_15% avg: {strategic_avg:.3f}x (paper: 1.28x)")
    print(f"  strategic_15% max: {strategic_max:.3f}x (paper: approaching 1.33x)")
    print(f"  random_15% avg: {random_avg:.3f}x")
    print(f"  random_15% max: {random_max:.3f}x")
    print()
    print("Configuration analysis:")
    print(f"  random_15% drops 3/12 layers -> theoretical max = {12/9:.3f}x")
    print(f"  strategic_15% drops 6/12 layers -> theoretical max = {12/6:.3f}x")
    print(f"  Code docstring says 'validate the theoretical 3.1x speedup claim' (MISMATCH)")
    print()

    # Check what matches the paper's 1.33x theoretical
    print("Which config matches paper's 1.33x theoretical?")
    print(f"  random_15% (3 layers dropped): 12/(12-3) = {12/9:.3f}x theoretical <- matches 1.33x")
    print(f"  strategic_15% (6 layers dropped): 12/(12-6) = {12/6:.3f}x theoretical <- does NOT match 1.33x")
    print()
    print("VERDICT ANALYSIS:")
    if abs(strategic_avg - 1.28) < 0.2:
        print("  strategic_15% avg speedup is close to paper claim of 1.28x")
    else:
        print(f"  strategic_15% avg speedup ({strategic_avg:.3f}x) CONFLICTS with paper claim (1.28x)")
    print("=" * 60)

    # Save results
    import json
    output = {
        "batch_sizes": batch_sizes,
        "seq_len": seq_len,
        "speedup_by_batch": {
            config_name: {
                batch_size: results[batch_size]['baseline'] / results[batch_size][config_name]
                for batch_size in batch_sizes
            }
            for config_name in ['random_15%', 'strategic_15%', 'aggressive_25%']
        },
        "average_speedups": {
            config_name: float(np.mean(all_speedups[config_name]))
            for config_name in ['random_15%', 'strategic_15%', 'aggressive_25%']
        },
        "max_speedups": {
            config_name: float(np.max(all_speedups[config_name]))
            for config_name in ['random_15%', 'strategic_15%', 'aggressive_25%']
        }
    }
    with open('/home/chenhui/fabscore/agents4sci_rej/submission_277/fabscore_claude/workspace/claim_146_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Results saved to claim_146_results.json")

    return results


if __name__ == "__main__":
    main()
