import os
import json

import torch
import fire
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk


def train_production_adapter(
    model_name: str, task_name: str, dataset_path: str, examples_limit: int = 1500
):
    print(f"Training {task_name} with {examples_limit} examples")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    dataset = load_from_disk(dataset_path)

    if len(dataset) > examples_limit:
        dataset = dataset.shuffle(seed=42).select(range(examples_limit))

    def tokenize_function(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        tokenized = tokenizer(
            texts, truncation=True, padding=True, max_length=2048, return_tensors="pt"
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].clone(),
        }

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    train_size = int(0.9 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    output_dir = f"models/{task_name}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=25,
        logging_steps=25,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_total_limit=1,
        bf16=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print(f"Starting training for {task_name}...")
    train_result = trainer.train()

    adapter_path = os.path.join(output_dir, "adapter")
    lora_model.save_pretrained(adapter_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    print(f"Training completed for {task_name}!")
    print(f"Final train loss: {train_result.metrics.get('train_loss', 'N/A')}")

    # Clear GPU memory
    del model, lora_model, trainer
    torch.cuda.empty_cache()

    return {
        "task_name": task_name,
        "adapter_path": adapter_path,
        "metrics": train_result.metrics,
        "status": "success",
    }


def train_adapters(
    data_summary_path: str = "results/processing_summary.json",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    examples_limit: int = 1500,
):
    with open(data_summary_path) as r:
        data_config = json.load(r)
    datasets = [(config["task_name"], config["dataset_path"]) for config in data_config]

    results = []
    for i, (task_name, dataset_path) in enumerate(datasets):
        print(f"\n{'='*60}")
        print(f"Training {i+1}/{len(datasets)}: {task_name}")
        print(f"{'='*60}")

        try:
            result = train_production_adapter(
                task_name=task_name,
                dataset_path=dataset_path,
                model_name=model_name,
                examples_limit=examples_limit,
            )
            results.append(result)
        except Exception as e:
            print(f"Failed to train {task_name}: {e}")
            results.append(
                {
                    "task_name": task_name,
                    "adapter_path": None,
                    "metrics": None,
                    "status": "failed",
                    "error": str(e),
                }
            )

    results_path = "results/training_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    for r in successful:
        loss = r["metrics"].get("train_loss", "N/A")
        print(f"  ✓ {r['task_name']}: Final loss = {loss}")

    for r in failed:
        print(f"  ✗ {r['task_name']}: {r.get('error', 'Unknown error')}")

    return results


if __name__ == "__main__":
    fire.Fire(train_adapters)
