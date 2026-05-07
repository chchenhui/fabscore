import json
from collections import defaultdict
from typing import Dict, Any

import fire
import torch
from transformers import AutoModelForCausalLM


def load_base_model_weights(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    print("Loading base model weights...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    base_weights = {}
    for name, param in model.named_parameters():
        if any(n in name for n in ("q_proj", "v_proj", "k_proj", "o_proj")):
            base_weights[name] = param.detach().clone()
    del model
    torch.cuda.empty_cache()
    return base_weights


def extract_target_vector_from_merged(
    merged_model_path: str, base_weights: Dict[str, Any]
):
    print(f"Extracting target vector from {merged_model_path}")
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path, torch_dtype=torch.bfloat16
    )
    target_vector = {}
    for name, param in merged_model.named_parameters():
        if name in base_weights:
            delta = param.detach().clone() - base_weights[name]
            target_vector[name] = delta
    del merged_model
    torch.cuda.empty_cache()
    return target_vector


def aggregate_weight_magnitudes(expanded_weights_list):
    aggregated = {}
    for expanded_weights in expanded_weights_list:
        for name, weight in expanded_weights.items():
            weight_mag = torch.abs(weight)
            if name not in aggregated:
                aggregated[name] = weight_mag
            else:
                aggregated[name] = torch.maximum(aggregated[name], weight_mag)
    return aggregated


def create_binary_mask(aggregated_weights, target_params=100000):
    module_weights = defaultdict(list)
    lora_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

    for name, weight in aggregated_weights.items():
        if any(proj in name for proj in lora_modules):
            layer_match = name.split(".")
            layer_num = None
            for i, part in enumerate(layer_match):
                if part == "layers" and i + 1 < len(layer_match):
                    layer_num = layer_match[i + 1]
                    break

            if layer_num is not None:
                proj_type = None
                for proj in lora_modules:
                    if proj in name:
                        proj_type = proj
                        break

                if proj_type:
                    module_key = f"layer_{layer_num}_{proj_type}"
                    flat_weight = weight.flatten()
                    module_weights[module_key].append((name, flat_weight, weight.shape))

    total_modules = len(module_weights)
    assert total_modules != 0, "No valid modules found"

    k_per_module = target_params // total_modules
    print(
        f"Target {target_params} params across {total_modules} modules = ~{k_per_module} params per module"
    )

    binary_mask = {}
    total_selected = 0

    for _, weight_list in module_weights.items():
        if not weight_list:
            continue

        module_mask = {}
        combined_weights = []
        combined_info = []

        for name, flat_weight, orig_shape in weight_list:
            module_mask[name] = torch.zeros(orig_shape, dtype=torch.bool)
            combined_weights.append(flat_weight)
            combined_info.extend(
                [(name, i, orig_shape) for i in range(len(flat_weight))]
            )

        assert combined_weights
        assert combined_info

        all_weights = torch.cat(combined_weights)
        _, top_indices = torch.topk(all_weights, min(k_per_module, len(all_weights)))

        for idx in top_indices:
            name, orig_idx, orig_shape = combined_info[idx]
            assert orig_idx <= idx.item()
            existing_mask = module_mask[name].flatten()
            existing_mask[orig_idx] = True
            module_mask[name] = existing_mask.reshape(orig_shape)

        binary_mask.update(module_mask)
        total_selected += len(top_indices)

    print(f"Created binary mask with {total_selected} selected parameters")
    return binary_mask


def apply_binary_mask(weights, binary_mask):
    compressed_weights = []
    for name, mask in sorted(binary_mask.items()):
        if name in weights:
            weight = weights[name]
            compressed_weight = weight[mask]
            compressed_weights.append(compressed_weight.flatten())
    final_vector = torch.cat(compressed_weights, dim=0)
    return final_vector


def compress_task_vectors(
    target_params: int = 1000000,
    binary_mask_path: str = "models/unified_selection_mask.pt",
    dictionary_tasks_path: str = "results/dictionary_tasks.json",
    training_results_path: str = "results/training_results.json",
):
    with open(dictionary_tasks_path) as r:
        dictionary_tasks = json.load(r)["dictionary_tasks"]
    with open(training_results_path) as r:
        all_tasks = [task["task_name"] for task in json.load(r)]

    print(
        f"Loading {len(all_tasks)} task adapters, {len(dictionary_tasks)} in dictionary..."
    )

    expanded_weights_list = []
    task_vectors = {}

    base_weights = load_base_model_weights()
    for task in all_tasks:
        print(f"Processing {task}...")
        vector = extract_target_vector_from_merged(f"models/{task}/full", base_weights)
        task_vectors[task] = vector
        if task in dictionary_tasks:
            expanded_weights_list.append(vector)
        total_params = sum(w.numel() for w in vector.values())
        print(f"  Expanded {task}: {total_params} parameters")

    print(f"Loaded {len(expanded_weights_list)} dictionary task vectors")

    print("Aggregating weight magnitudes across dictionary adapters...")
    aggregated = aggregate_weight_magnitudes(expanded_weights_list)

    print("Creating binary selection mask...")
    binary_mask = create_binary_mask(aggregated, target_params=target_params)

    print("Saving binary mask...")
    torch.save(binary_mask, binary_mask_path)

    print("Applying mask to compress task vectors...")
    compressed_task_vectors = {}

    for task, weights in task_vectors.items():
        compressed = apply_binary_mask(weights, binary_mask)
        compressed_task_vectors[task] = compressed
        print(f"  {task}: {len(compressed)} parameters")

        torch.save(compressed, f"models/{task}/compressed_task_vector.pt")

    metadata = {
        "dictionary_tasks": dictionary_tasks,
        "tasks_processed": list(task_vectors.keys()),
        "target_compressed_params": target_params,
        "binary_mask_path": binary_mask_path,
        "compressed_task_vectors": {
            task: f"models/{task}/compressed_task_vector.pt"
            for task in task_vectors.keys()
        },
        "total_mask_entries": len(binary_mask),
        "mask_selection_method": "top-k per module (q_proj, k_proj, v_proj, o_proj)",
        "aggregation_method": "max magnitude across adapters",
    }

    with open("results/compress_task_vectors_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Compression completed successfully!")
    return metadata


if __name__ == "__main__":
    fire.Fire(compress_task_vectors)
