import json
import yaml
import glob
import numpy as np
from pathlib import Path


def process_models(dictionary_tasks, model_list, category):
    for model_name in model_list:
        config_file = f"results/merge_configs/{model_name}.yaml"

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        gt_vector = np.zeros(len(dictionary_tasks))
        for model_info in config["models"]:
            model_path = model_info["model"]["model"]
            task_name = model_path.split("/")[-2]

            if task_name in dictionary_tasks:
                task_idx = dictionary_tasks.index(task_name)
                weight = model_info["parameters"]["weight"]
                gt_vector[task_idx] = weight

        ground_truth[model_name] = {
            "category": category,
            "weights": gt_vector.tolist(),
            "tasks_used": [
                dictionary_tasks[i]
                for i in range(len(dictionary_tasks))
                if gt_vector[i] > 0
            ],
        }


with open("results/dictionary_tasks.json", "r") as f:
    dictionary_tasks = list(json.load(f)["dictionary_tasks"])
print(f"Dictionary tasks: {dictionary_tasks}")

config_files = glob.glob("results/merge_configs/*.yaml")
known_models = []
mixed_models = []
unknown_models = []

for config_file in config_files:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_name = Path(config_file).stem
    tasks_in_model = []
    for model_info in config["models"]:
        model_path = model_info["model"]["model"]
        task_name = model_path.split("/")[-2]
        tasks_in_model.append(task_name)
    tasks_in_model_set = set(tasks_in_model)
    dict_overlap = tasks_in_model_set.intersection(dictionary_tasks)
    non_dict_tasks = tasks_in_model_set - set(dictionary_tasks)
    if len(dict_overlap) == len(tasks_in_model_set) and len(non_dict_tasks) == 0:
        known_models.append(model_name)
    elif len(dict_overlap) > 0 and len(non_dict_tasks) > 0:
        mixed_models.append(model_name)
    elif len(dict_overlap) == 0 and len(non_dict_tasks) == len(tasks_in_model_set):
        unknown_models.append(model_name)
    if len(known_models) + len(mixed_models) + len(unknown_models) <= 5:
        print(
            f"{model_name}: tasks={tasks_in_model}, dict_overlap={len(dict_overlap)}, non_dict={len(non_dict_tasks)}"
        )

print("\nClassification results:")
print(f"Known models (all tasks in dictionary): {len(known_models)}")
print(f"Mixed models (some tasks in dictionary): {len(mixed_models)}")
print(f"Unknown models (no tasks in dictionary): {len(unknown_models)}")
print(f"Total: {len(known_models) + len(mixed_models) + len(unknown_models)}")

classification = {
    "known": known_models,
    "mixed": mixed_models,
    "unknown": unknown_models,
}

print(f'Known models: {len(classification["known"])}')
print(f'Mixed models: {len(classification["mixed"])}')

ground_truth = {}
process_models(dictionary_tasks, classification["known"], "known")
process_models(dictionary_tasks, classification["mixed"], "mixed")
for model_name in classification["unknown"]:
    ground_truth[model_name] = {
        "category": "unknown",
        "weights": [0.0] * len(dictionary_tasks),
        "tasks_used": [],
    }

print(f"\nCreated ground truth for {len(ground_truth)} models")

print("\nExample known model:")
known_example = classification["known"][0]
print(f"{known_example}: {ground_truth[known_example]}")

print("\nExample mixed model:")
mixed_example = classification["mixed"][0]
print(f"{mixed_example}: {ground_truth[mixed_example]}")

with open("results/ground_truth_compositions.json", "w") as f:
    json.dump(ground_truth, f, indent=2)

print("\nSaved ground truth compositions to results/ground_truth_compositions.json")
