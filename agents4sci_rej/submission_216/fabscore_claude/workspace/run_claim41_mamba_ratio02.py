"""
Wrapper script to run improved_proposed_method.py with ratio=0.2 for Mamba-1.4b-hf only.
For claim 41 verification: Figure 5 score distributions.
"""
import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# Add the Mink%++ directory to sys.path
module_dir = '/home/chenhui/fabscore/agents4sci_rej/submission_216/Mink%++'
sys.path.insert(0, module_dir)

import improved_proposed_method as imp

# Override Config ratio to 0.2
config = imp.Config()
config.ratio = 0.2
config.models = ['state-spaces/mamba-1.4b-hf']  # Only Mamba for Figure 5

output_dir = '/home/chenhui/fabscore/agents4sci_rej/submission_216/fabscore_claude/workspace/ratio02_result'
os.makedirs(output_dir, exist_ok=True)

all_results = {}
all_scores = {}

# load model
for model_name in config.models:
    model, tokenizer = imp.load_model(model_name, config)
    model_id = model_name.split('/')[-1]
    all_results[model_id] = {}
    all_scores[model_id] = {}

    # Process each dataset
    for dataset_name in config.datasets:
        try:
            results, scores, labels = imp.process_dataset(model, tokenizer, dataset_name, config)

            all_results[model_id][dataset_name] = {
                'method': results['method'],
                'auroc': results['auroc'],
                'fpr95': results['fpr95'],
                'tpr05': results['tpr05']
            }

            all_scores[model_id][dataset_name] = {}
            for method, method_scores in scores.items():
                score_dict = {"training": [], "non-training": []}
                for label, score in zip(labels, method_scores):
                    if label == 1:
                        score_dict["training"].append(score)
                    elif label == 0:
                        score_dict["non-training"].append(score)
                all_scores[model_id][dataset_name][method] = score_dict

            df = pd.DataFrame(results)
            print(f"\nResults for {dataset_name}:")
            print(df)

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

# Save results
result_file = os.path.join(output_dir, "results.json")
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# Save scores
scores_file = os.path.join(output_dir, "scores.pkl")
with open(scores_file, 'wb') as f:
    pickle.dump(all_scores, f)

print(f"\nResults saved to {result_file}")
print(f"Scores saved to {scores_file}")
print("\nFull results:")
print(json.dumps(all_results, indent=2))
