import os
import json

import fire
import torch

from src.compress_task_vectors import (
    load_base_model_weights,
    extract_target_vector_from_merged,
    apply_binary_mask,
)


def compress_target_vectors(mask_path: str = "models/unified_selection_mask.pt"):
    print("Starting target vector extraction and compression...")
    if not os.path.exists(mask_path):
        print(f"Binary mask not found at {mask_path}")
        return

    binary_mask = torch.load(mask_path)
    print(f"Loaded binary mask with {len(binary_mask)} entries")

    base_weights = load_base_model_weights()
    if base_weights is None:
        return

    print(f"Loaded base weights for {len(base_weights)} parameters")

    merge_results_path = "results/merge_results.json"
    assert os.path.exists(merge_results_path)

    with open(merge_results_path) as f:
        merge_data = json.load(f)
        merge_results = merge_data["results"]

    compressed_targets = {}
    success_count = 0

    for i, result in enumerate(merge_results):
        model_name = result["model_name"]
        merged_path = f"models/merged/{model_name}"

        if not os.path.exists(merged_path):
            print(f"Merged model not found: {merged_path}")
            continue

        print(f"Processing {i+1}/{len(merge_results)}: {model_name}")

        target_vector = extract_target_vector_from_merged(merged_path, base_weights)
        if target_vector is None:
            continue

        compressed_vector = apply_binary_mask(target_vector, binary_mask)
        output_path = f"models/merged/{model_name}/compressed_vector.pt"
        torch.save(compressed_vector, output_path)

        compressed_targets[model_name] = {
            "path": output_path,
            "selected_params": len(compressed_vector),
            "group": result["group"],
            "method": result["method"],
            "tasks": result["tasks"],
        }

        success_count += 1
        print(f"  Saved with {len(compressed_vector)} selected parameters")

    metadata = {
        "compressed_target_vectors": compressed_targets,
        "total_processed": success_count,
        "binary_mask_path": mask_path,
    }

    with open("results/compress_target_vectors_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully processed {success_count} target vectors")
    return metadata


if __name__ == "__main__":
    fire.Fire(compress_target_vectors)
