import os
import json
from abc import ABC, abstractmethod
from datasets import Dataset, load_dataset
from typing import List, Dict, Any, Optional, Tuple

from src.utils.common import save_dataset_sample, validate_chat_format


class BaseDatasetProcessor(ABC):
    def __init__(self, dataset_name: str, task_name: str, max_samples: int = 2000):
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.max_samples = max_samples
        self.system_prompt = self.get_system_prompt()

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        pass

    def load_raw_dataset(self) -> Dataset:
        print(f"Loading dataset: {self.dataset_name}")
        try:
            dataset = load_dataset(self.dataset_name, split="train")
            print(f"Loaded {len(dataset)} examples from {self.dataset_name}")
            return dataset
        except Exception as e:
            print(f"Failed to load dataset {self.dataset_name}: {e}")
            raise

    def process_dataset(self, output_dir: str) -> Tuple[Dataset, Dict[str, Any]]:
        raw_dataset = self.load_raw_dataset()

        if len(raw_dataset) > self.max_samples:
            raw_dataset = raw_dataset.shuffle(seed=42).select(range(self.max_samples))
            print(f"Limited dataset to {self.max_samples} samples")

        processed_examples = []
        failed_count = 0

        for i, example in enumerate(raw_dataset):
            try:
                messages = self.process_example(example)
                if messages and validate_chat_format(messages):
                    processed_examples.append(
                        {
                            "messages": messages,
                            "task": self.task_name,
                            "original_index": i,
                        }
                    )
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Failed to process example {i}: {e}")
                failed_count += 1

        print(f"Processed {len(processed_examples)} examples, {failed_count} failed")

        processed_dataset = Dataset.from_list(processed_examples)

        os.makedirs(output_dir, exist_ok=True)
        sample_path = os.path.join(output_dir, f"{self.task_name}_sample.txt")
        save_dataset_sample(processed_dataset, sample_path)

        metadata = {
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "total_examples": len(processed_examples),
            "failed_examples": failed_count,
            "system_prompt": self.system_prompt,
            "max_samples": self.max_samples,
        }

        metadata_path = os.path.join(output_dir, f"{self.task_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return processed_dataset, metadata

    def save_processed_dataset(self, dataset: Dataset, output_dir: str):
        output_path = os.path.join(output_dir, f"{self.task_name}_processed")
        dataset.save_to_disk(output_path)
        print(f"Saved processed dataset to {output_path}")
        return output_path
