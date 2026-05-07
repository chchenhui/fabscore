import json
import os

# Extracted from metaprompt.py
COMPARISON_TEMPLATE = """
| Model | Validation Accuracy | Test Accuracy |
|---|---|---|
| GoogleNet24 | | |
| VGG1625 | | |
| ResNet5026 | | |
| DenseNet20127 | | |
| Inceptionv328 | | |
| Xception29 | | |
| InceptionResnetV230 | | |
| NasnetLarge31 | | |
| EfficientNetB732 | | |
| Vision transformer17 | | |
| CONVT33 | | |
| Beit_large34 | | |
| RVT | | |
| Our model | | |
"""

ABLATION_TEMPLATE = """
| Ablation Study | Validation Accuracy | Test Accuracy |
|---|---|---|
"""

# Performance Comparison Table
performance_data = {
    "GoogleNet24": {"Validation R_Accuracy": 77.5, "Test R_Accuracy": 74.8},
    "VGG1625": {"Validation R_Accuracy": 78.5, "Test R_Accuracy": 75.0},
    "ResNet5026": {"Validation R_Accuracy": 80.0, "Test R_Accuracy": 78.3},
    "DenseNet20127": {"Validation R_Accuracy": 77.0, "Test R_Accuracy": 80.8},
    "Inceptionv328": {"Validation R_Accuracy": 79.0, "Test R_Accuracy": 80.0},
    "Xception29": {"Validation R_Accuracy": 78.5, "Test R_Accuracy": 75.3},
    "InceptionResnetV230": {"Validation R_Accuracy": 77.5, "Test R_Accuracy": 78.8},
    "NasnetLarge31": {"Validation R_Accuracy": 80.0, "Test R_Accuracy": 82.5},
    "EfficientNetB732": {"Validation R_Accuracy": 82.0, "Test R_Accuracy": 84.5},
    "Vision transformer17": {"Validation R_Accuracy": 80.0, "Test R_Accuracy": 77.5},
    "CONVT33": {"Validation R_Accuracy": 82.0, "Test R_Accuracy": 85.3},
    "Beit_large34": {"Validation R_Accuracy": 71.0, "Test R_Accuracy": 74.3},
    "RVT": {"Validation R_Accuracy": 85.0, "Test R_Accuracy": 87.3}
}

with open(os.path.join('Results', 'results.json'), 'r') as f:
    our_results = json.load(f)
    performance_data["Our model"] = {
        "Validation R_Accuracy": our_results['validation']['accuracy'] * 100,
        "Test R_Accuracy": our_results['test']['accuracy'] * 100
    }

with open(os.path.join('Results', 'performance_comparison.json'), 'w') as f:
    json.dump(performance_data, f, indent=4)

# Ablation Study Tables
ablation_results_path = 'ablation_results'
ablation_data = {}
for folder in os.listdir(ablation_results_path):
    folder_path = os.path.join(ablation_results_path, folder)
    if os.path.isdir(folder_path):
        for subfolder in os.listdir(folder_path):
            if subfolder.startswith("results_"):
                results_file = os.path.join(folder_path, subfolder, "results.json")
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        ablation_data[folder] = {"accuracy": data["test"]["accuracy"]}

with open(os.path.join('Results', 'results.json'), 'r') as f:
    our_results = json.load(f)
    ablation_data["Our model"] = {"accuracy": our_results['test']['accuracy']}

# Multi-task Ablation
multi_task_ablation_studies = [
    "Classification_Only", "VHS_Regression_Only", "Keypoint_Only",
    "Classification_plus_VHS_Regression", "Keypoint_plus_Classification",
    "Keypoint_plus_VHS_Regression", "Our model"
]
multi_task_ablation = {"accuracy": {k: ablation_data.get(k) for k in multi_task_ablation_studies}}
with open(os.path.join('Results', 'multi_task_ablation.json'), 'w') as f:
    json.dump(multi_task_ablation, f, indent=4)

# Cross Attention and KP Head Ablation
cross_attention_kp_head_ablation_studies = ["No_Cross_Attention", "Simple_KP_Head", "Our model"]
cross_attention_kp_head_ablation = {"accuracy": {k: ablation_data.get(k) for k in cross_attention_kp_head_ablation_studies}}
with open(os.path.join('Results', 'cross_attention_kp_head_ablation.json'), 'w') as f:
    json.dump(cross_attention_kp_head_ablation, f, indent=4)

# Loss Ablation
loss_ablation_studies = [
    "Fixed_Loss_Weights_CLS_Heavy", "Fixed_Loss_Weights_Equal",
    "Fixed_Loss_Weights_KP_Heavy", "Fixed_Loss_Weights_VHS_Heavy", "Our model"
]
loss_ablation = {"accuracy": {k: ablation_data.get(k) for k in loss_ablation_studies}}
with open(os.path.join('Results', 'loss_ablation.json'), 'w') as f:
    json.dump(loss_ablation, f, indent=4)

print("Tables created successfully in the Results folder.")
