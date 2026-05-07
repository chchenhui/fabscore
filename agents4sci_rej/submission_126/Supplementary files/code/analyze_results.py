import json
import os
import numpy as np

def analyze_results(results_dir='../Results/multiseed_finetune'):
    accuracies = []
    for filename in os.listdir(results_dir):
        if filename.startswith('results_') and filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                try:
                    results = json.load(f)
                    accuracies.append(results['test']['accuracy'])
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {filename}")
                except KeyError:
                    print(f"Warning: 'test' or 'accuracy' key not found in {filename}")

    if not accuracies:
        print("No results found to analyze.")
        return

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print(f"Analysis of {len(accuracies)} runs:")
    print(f"  Mean test accuracy: {mean_accuracy:.4f}")
    print(f"  Standard deviation of test accuracy: {std_accuracy:.4f}")

if __name__ == '__main__':
    analyze_results()