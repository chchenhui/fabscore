import os
import json

import torch
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def validate_adapter(adapter_path, task_name, test_examples):
    print(f"Validating adapter: {task_name}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )

    adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
    adapter_model.eval()

    results = []
    for i, example in enumerate(test_examples):
        try:
            if isinstance(example, dict) and "messages" in example:
                messages = example["messages"][:-1]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": str(example)},
                ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=512
            ).to(adapter_model.device)

            with torch.no_grad():
                outputs = adapter_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            result = {
                "example_id": i,
                "input": input_text,
                "generated_response": response.strip(),
                "status": "success",
            }
            results.append(result)

            print(f"Example {i+1}/{len(test_examples)} - Success")

        except Exception as e:
            print(f"Example {i+1} failed: {e}")
            results.append(
                {
                    "example_id": i,
                    "input": str(example),
                    "generated_response": None,
                    "status": "failed",
                    "error": str(e),
                }
            )

    del adapter_model, base_model
    torch.cuda.empty_cache()
    return results


def validate_adapters():
    test_cases = {
        "python_instructions_alpaca": [
            "Write a Python function to calculate the factorial of a number.",
            "Create a Python function that reverses a string.",
            "Write code to find the maximum element in a list.",
        ],
        "arjun_python_qa": [
            "What is the difference between a list and a tuple in Python?",
            "How do you handle exceptions in Python?",
            "Explain Python decorators with an example.",
        ],
        "gsm8k_math": [
            "Sarah has 15 apples. She gives 3 to her friend and eats 2. How many apples does she have left?",
            "A rectangle has length 8 and width 5. What is its area?",
            "If 3x + 7 = 22, what is the value of x?",
        ],
        "squad_qa": [
            "Context: The cat sat on the mat in the warm sunshine. Question: Where did the cat sit?",
            "Context: Python is a programming language created by Guido van Rossum. Question: Who created Python?",
            "Context: The library opens at 9 AM and closes at 5 PM. Question: What time does the library close?",
        ],
        "alpaca_instructions": [
            "Explain the concept of machine learning in simple terms.",
            "Write a brief summary of the benefits of renewable energy.",
            "Describe how to make a paper airplane.",
        ],
    }

    adapters = [
        ("python_instructions_alpaca", "models/python_instructions_alpaca/adapter"),
        ("arjun_python_qa", "models/arjun_python_qa/adapter"),
        ("gsm8k_math", "models/gsm8k_math/adapter"),
        ("squad_qa", "models/squad_qa/adapter"),
        ("alpaca_instructions", "models/alpaca_instructions/adapter"),
    ]

    all_results = {}

    for task_name, adapter_path in adapters:
        print(f"\n{'='*60}")
        print(f"Validating {task_name}")
        print(f"{'='*60}")

        test_examples = test_cases.get(task_name, ["Test example"])

        try:
            results = validate_adapter(adapter_path, task_name, test_examples)
            all_results[task_name] = {
                "adapter_path": adapter_path,
                "test_results": results,
                "success_rate": len([r for r in results if r["status"] == "success"])
                / len(results),
                "status": "completed",
            }

        except Exception as e:
            print(f"Validation failed for {task_name}: {e}")
            all_results[task_name] = {
                "adapter_path": adapter_path,
                "test_results": [],
                "success_rate": 0.0,
                "status": "failed",
                "error": str(e),
            }

    results_path = "results/validation_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    for task_name, result in all_results.items():
        if result["status"] == "completed":
            success_rate = result["success_rate"]
            print(f"✓ {task_name}: {success_rate:.1%} success rate")
        else:
            print(f"✗ {task_name}: Validation failed")

    return all_results


if __name__ == "__main__":
    fire.Fire(validate_adapters)
