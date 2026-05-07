import os
import json

import fire

from src.data_processing.processors import (
    MSMarcoProcessor,
    PythonInstructionsAlpacaProcessor,
    ArjunPythonQAProcessor,
    GSM8KProcessor,
    SQuADProcessor,
    AlpacaProcessor,
    CodeSearchNetPythonProcessor,
    StableCodePythonProcessor,
    ArxivSummarizationProcessor,
    LatinTranslationProcessor,
    StyleTransferProcessor,
    IMDBProcessor,
    OrcaMathProcessor,
    XSumProcessor,
    ReasonMathProcessor,
)

PROCESSORS = (
    ReasonMathProcessor(max_samples=1500),
    OrcaMathProcessor(max_samples=1500),
    XSumProcessor(max_samples=1500),
    IMDBProcessor(max_samples=1500),
    StyleTransferProcessor(max_samples=1500),
    MSMarcoProcessor(max_samples=1500),
    LatinTranslationProcessor(max_samples=1500),
    ArxivSummarizationProcessor(max_samples=1500),
    PythonInstructionsAlpacaProcessor(max_samples=1500),
    ArjunPythonQAProcessor(max_samples=1500),
    GSM8KProcessor(max_samples=1500),
    SQuADProcessor(max_samples=1500),
    AlpacaProcessor(max_samples=1500),
    CodeSearchNetPythonProcessor(max_samples=1500),
    StableCodePythonProcessor(max_samples=1500),
)


def process_all_datasets(output_base_dir: str = "data/processed"):
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    results = []
    processors = PROCESSORS
    for i, processor in enumerate(processors):
        print(f"Processing dataset {i+1}/{len(processors)}: {processor.task_name}")

        try:
            processed_dataset, metadata = processor.process_dataset(output_base_dir)
            dataset_path = processor.save_processed_dataset(
                processed_dataset, output_base_dir
            )
            result = {
                "task_name": processor.task_name,
                "dataset_name": processor.dataset_name,
                "total_examples": len(processed_dataset),
                "dataset_path": dataset_path,
                "status": "success",
            }
            results.append(result)
            print(
                f"Successfully processed {processor.task_name}: {len(processed_dataset)} examples"
            )

        except Exception as e:
            print(f"Failed to process {processor.task_name}: {e}")
            result = {
                "task_name": processor.task_name,
                "dataset_name": processor.dataset_name,
                "total_examples": 0,
                "dataset_path": None,
                "status": "failed",
                "error": str(e),
            }
            results.append(result)

    summary_path = "results/processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print("Processing complete!")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("Failed datasets:")
        for f in failed:
            print(f"  - {f['task_name']}: {f['error']}")

    return results


if __name__ == "__main__":
    fire.Fire(process_all_datasets)
