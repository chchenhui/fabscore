import os
import json
import random
from pathlib import Path

import datasets


def load_processed_data(task_name, max_examples_per_task=125):
    data_dir = Path(f"data/processed/{task_name}_processed")
    examples = []

    if data_dir.exists():
        dataset = datasets.load_from_disk(str(data_dir))

        for item in dataset:
            examples.append(
                {
                    "messages": item["messages"],
                }
            )
    random.shuffle(examples)
    return examples[:max_examples_per_task]


def create_diverse_dataset():
    known_tasks = [
        "latin_translation",
        "codesearchnet_python",
        "python_instructions_alpaca",
        "alpaca_instructions",
        "ms_marco_qa",
        "xsum",
        "reason_math",
        "gsm8k_math",
    ]

    examples_per_task = 1000 // len(known_tasks)

    diverse_dataset = []
    task_counts = {}

    for task in known_tasks:
        print(f"Loading data for {task}...")
        task_examples = load_processed_data(task, examples_per_task)

        diverse_dataset.extend(task_examples)
        task_counts[task] = len(task_examples)
        print(f"  Loaded {len(task_examples)} examples")

    random.shuffle(diverse_dataset)

    print(f"\nTotal examples: {len(diverse_dataset)}")
    print("Examples per task:")
    for task, count in task_counts.items():
        print(f"  {task}: {count}")

    return diverse_dataset, task_counts


if __name__ == "__main__":
    random.seed(42)  # For reproducibility

    dataset, counts = create_diverse_dataset()

    os.makedirs("results", exist_ok=True)
    with open("results/diverse_dataset_1k.json", "w") as f:
        json.dump(dataset, f, indent=2)

    metadata = {
        "total_examples": len(dataset),
        "task_counts": counts,
        "target_size": 1000,
        "known_tasks": list(counts.keys()),
    }

    with open("results/diverse_dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDiverse dataset saved with {len(dataset)} examples")
