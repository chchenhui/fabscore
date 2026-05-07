import torch
import json

merge_results_path = "results/merge_results.json"
with open(merge_results_path) as f:
    merge_data = json.load(f)
    merge_results = merge_data["results"]

print("Checking first few target vectors...")
for i, result in enumerate(merge_results[:5]):
    model_name = result["model_name"]
    vector_path = f"models/merged/{model_name}/compressed_vector.pt"

    vector = torch.load(vector_path)
    print(
        f"{model_name}: dtype={vector.dtype}, norm={torch.norm(vector.float()).item():.6f}, shape={vector.shape}"
    )
    print(f"  Non-zero elements: {torch.count_nonzero(vector).item()}")
    print(f"  Min/Max values: {vector.min().item():.6f} / {vector.max().item():.6f}")
    print(f"  Tasks: {result['tasks']}")
    print()
