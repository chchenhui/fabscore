import unsloth  # noqa
import argparse
import datetime
import os
import sys

import torch
from accelerate import PartialState
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, unsloth_train
from peft import PeftModel
from utils.train_utils import EarlyStoppingCallback
import wandb

sys.path.append("..")

from typing import Any, Dict


def _tokenize_example(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    processed = tokenizer.apply_chat_template(
        example["messages"],
        return_dict=True,
        add_generation_prompt=False,
        add_special_tokens=False,
        enable_thinking=False,
    )

    input_ids: list[int] = processed["input_ids"]
    assistant_masks: list[int] | None = processed.get("assistant_masks")

    if assistant_masks is None or all(v == 0 for v in assistant_masks):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        assistant_masks = [0] * len(tokens)

        i = 0
        while i < len(tokens):
            # Look for assistant turn start pattern
            if (
                i < len(tokens) - 2
                and tokens[i] == "<|start_header_id|>"
                and i + 1 < len(tokens)
                and tokens[i + 1] == "assistant"
                and i + 2 < len(tokens)
                and tokens[i + 2] == "<|end_header_id|>"
            ):
                # Skip to after the <|end_header_id|> token
                j = i + 3

                # Skip any newline tokens immediately after <|end_header_id|>
                while j < len(tokens) and tokens[j] in ["\n", "Ċ", "ĊĊ"]:
                    j += 1

                # Mark all tokens until and including <|eot_id|> as assistant tokens
                while j < len(tokens) and tokens[j] != "<|eot_id|>":
                    assistant_masks[j] = 1
                    j += 1

                # Also mark the <|eot_id|> token itself as assistant content
                if j < len(tokens) and tokens[j] == "<|eot_id|>":
                    assistant_masks[j] = 1
                    j += 1

                i = j  # continue scanning after <|eot_id|>
            else:
                i += 1

    assert len(input_ids) == len(assistant_masks)
    return {
        "input_ids": input_ids,
        "assistant_masks": assistant_masks,
    }


def prepare_dataset(dataset, tokenizer):
    """Apply ``_tokenize_example`` over an entire 🤗 *datasets* object."""

    remove_cols = [c for c in dataset.column_names if c not in {"messages"}]

    return dataset.map(
        lambda ex: _tokenize_example(ex, tokenizer),
        remove_columns=remove_cols,
        desc="Tokenizing dataset with chat template",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    return parser.parse_args()


def load_environment(args):
    # Load environment variables
    if not os.path.exists(args.env):
        raise FileNotFoundError(f"Environment file not found: {args.env}")

    load_dotenv(args.env)

    # Check for required environment variables
    required_vars = ["HF_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    return {
        "hf_token": os.getenv("HF_TOKEN"),
        "wandb_api_key": os.getenv("WANDB_API_KEY"),
    }


def load_dataset_flexible(train_path: str, env_vars: dict):
    """Load training dataset from a local file or a HF Hub repo.

    1. If *train_path* exists on the local filesystem it is assumed to be a
       JSON/JSONL file and will be loaded with
       ``datasets.load_dataset("json", data_files=train_path, split="train")``.
    2. Otherwise *train_path* is treated as a Hugging Face dataset repository
       identifier (e.g. ``"username/my_dataset"``) and is loaded with
       ``datasets.load_dataset(train_path, split="train")``.

    A ``HF_TOKEN`` taken from *env_vars* is passed when fetching from the hub to
    allow access to private datasets.
    """
    # Local file path case
    if os.path.exists(train_path):
        return load_dataset("json", data_files=train_path, split="train")

    # Treat as a Hugging Face dataset repository identifier
    return load_dataset(train_path, split="train", token=env_vars.get("hf_token"))


def main():
    # Parse arguments
    args = parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Set random seed for reproducibility
    seed = cfg.seed

    # Initialize distributed state for multi-GPU training
    distributed_state = PartialState()

    # Load environment variables
    env_vars = load_environment(args)

    # Add timestamp to output_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_output_dir = cfg.training.output_dir
    cfg.training.output_dir = f"{orig_output_dir}_{timestamp}"

    # Model and tokenizer setup using Unsloth FastLanguageModel
    if distributed_state.is_main_process:
        print("Loading model and tokenizer with Unsloth FastLanguageModel...")

    # Model and tokenizer setup using standard transformers
    if distributed_state.is_main_process:
        print("Loading model and tokenizer with transformers + bitsandbytes...")
    device_map = "auto"
    # if multiple gpus, use device mapping
    if torch.cuda.device_count() > 1:
        device_map = {"": distributed_state.process_index}

    model, _ = FastLanguageModel.from_pretrained(
        model_name=cfg.model.model_id,
        max_seq_length=getattr(cfg.model, "max_seq_length", 2048),
        dtype=torch.bfloat16,
        load_in_4bit=cfg.model.quantization.load_in_4bit,
        load_in_8bit=cfg.model.quantization.load_in_8bit,
        full_finetuning=cfg.model.full_finetuning,
        token=env_vars["hf_token"],
        device_map=device_map,
        attn_implementation=cfg.model.attn_implementation,
        # trust_remote_code=True,
    )
    if distributed_state.is_main_process:
        print(f"Model loaded with Unsloth: {model}")

    # load tokenizer of original model
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_id, trust_remote_code=True, token=env_vars["hf_token"]
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare data
    # Load dataset (local JSON/JSONL file *or* HF Hub repo)
    dataset = load_dataset_flexible(cfg.data.train_path, env_vars)
    if distributed_state.is_main_process:
        print(f"Length of full dataset: {len(dataset)}")

    if cfg.data.validation_split > 0:
        # Simple random train/validation split
        split_dataset = dataset.train_test_split(
            test_size=cfg.data.validation_split, seed=cfg.seed
        )
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]

        if distributed_state.is_main_process:
            print("\nDataset Information (Random split):")
            print(f"Number of training examples: {len(train_dataset)}")
            print(f"Number of validation examples: {len(test_dataset)}")
            print(f"Validation split: {cfg.data.validation_split}")

    else:
        train_dataset = dataset
        test_dataset = None
        if distributed_state.is_main_process:
            print("\nDataset Information:")
            print(f"Number of training examples: {len(train_dataset)}")

    # Pre-tokenize datasets with chat template and enable_thinking=False
    if distributed_state.is_main_process:
        print("\nTokenizing datasets with chat template (enable_thinking=False)...")

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    if test_dataset is not None:
        test_dataset = prepare_dataset(test_dataset, tokenizer)

    # Add LoRA adapters using Unsloth's get_peft_model method
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    if cfg.model.lora_id:
        adapter = PeftModel.from_pretrained(model, cfg.model.lora_id)
        model = adapter.merge_and_unload()
        print(f"LoRA adapters merged and unloaded: {model}")

    # LoRA configuration is handled by Unsloth's get_peft_model method above
    if distributed_state.is_main_process:
        print("✓ LoRA adapters configured using Unsloth")

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        max_steps=cfg.training.max_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        learning_rate=cfg.training.learning_rate,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        save_strategy=cfg.training.save_strategy,
        max_grad_norm=cfg.training.max_grad_norm,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        eval_strategy=cfg.training.eval_strategy
        if cfg.data.validation_split > 0
        else "no",
        report_to="wandb",
        run_name=cfg.wandb.run_name,
        load_best_model_at_end=cfg.data.validation_split > 0,
        metric_for_best_model="eval_loss" if cfg.data.validation_split > 0 else None,
        greater_is_better=False,
        weight_decay=cfg.training.weight_decay,
        completion_only_loss=True,
        eos_token=cfg.model.eos_token,
        # Fix for PEFT model label_names warning
        label_names=["labels"],  # Explicitly set label_names for PeftModelForCausalLM
        push_to_hub=cfg.hub.push_to_hub,
        hub_model_id=f"{cfg.hub.account}/{cfg.wandb.run_name}",
        hub_token=env_vars["hf_token"],
        hub_private_repo=cfg.hub.private_repo,
        seed=seed,  # Set seed for training reproducibility
        warmup_ratio=cfg.training.warmup_ratio,
        dataloader_pin_memory=False,  # Helps with memory management
        remove_unused_columns=False,  # Keep all columns for SFT
        ddp_find_unused_parameters=False,  # Required for DDP performance
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_on_each_node=torch.cuda.device_count() > 1,
        packing=False,
        max_seq_length=cfg.model.max_seq_length,
    )

    # Initialize wandb if API key is available
    if env_vars["wandb_api_key"] and distributed_state.is_main_process:
        os.environ["WANDB_API_KEY"] = env_vars["wandb_api_key"]
        # Log only essential parameters
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            settings=wandb.Settings(
                start_method="thread"
            ),  # Use thread-based initialization
        )
        print("✓ Wandb initialized")

    # Initialize trainer
    # Note: SFTTrainer will automatically detect pre-tokenized datasets with 'input_ids' column
    # and will not perform additional tokenization or packing
    # Since we already applied PEFT using Unsloth, we don't need peft_config here
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        packing=False,
        dataset_text_field="text",
        dataset_num_proc=2,
    )

    if cfg.data.validation_split > 0 and cfg.training.early_stopping_patience > 0:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping_patience
            )
        )

    # Start training
    stats = unsloth_train(trainer)
    if distributed_state.is_main_process:
        print(stats)

    # Save the model
    final_model_path = f"{cfg.training.output_dir}-final"
    trainer.save_model(final_model_path)
    if distributed_state.is_main_process:
        print(f"✓ Model saved to {final_model_path}")

    # Finish wandb run if it was initialized
    if env_vars["wandb_api_key"] and distributed_state.is_main_process:
        wandb.finish()
        print("✓ Wandb run finished")


if __name__ == "__main__":
    main()
