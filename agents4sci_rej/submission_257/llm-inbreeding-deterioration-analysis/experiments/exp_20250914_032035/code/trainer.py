"""
Model Trainer for LLM Inbreeding Deterioration Analysis

This module implements the training pipeline for multi-generation experiments,
managing model training across different conditions and generations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiGenerationTrainer:
    """Manages training across multiple generations and conditions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config["device"]
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create directories
        self.checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path(config["paths"]["logs_dir"])
        self.logs_dir.mkdir(exist_ok=True)
        
        # Training history
        self.training_history = {}
        
        logger.info(f"MultiGenerationTrainer initialized on device: {self.device}")
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training by tokenizing texts."""
        
        def tokenize_function(examples):
            # Combine input and output for language modeling
            texts = []
            for inp, out in zip(examples['input'], examples['output']):
                # Format as question-answer pairs
                text = f"Question: {inp}\nAnswer: {out}{self.tokenizer.eos_token}"
                texts.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config["max_sequence_length"],
                return_tensors="pt"
            )
            
            # For language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train_generation(self, 
                        train_dataset: Dataset,
                        val_dataset: Dataset,
                        generation: int,
                        condition: str,
                        base_model_path: str = None) -> Tuple[str, Dict[str, Any]]:
        """Train a model for a specific generation and condition."""
        
        logger.info(f"Starting training - Generation: {generation}, Condition: {condition}")
        
        # Load base model
        if base_model_path and Path(base_model_path).exists():
            logger.info(f"Loading model from checkpoint: {base_model_path}")
            model = AutoModelForCausalLM.from_pretrained(base_model_path)
        else:
            logger.info(f"Loading base model: {self.config['base_model_name']}")
            model = AutoModelForCausalLM.from_pretrained(self.config["base_model_name"])
        
        # Prepare datasets
        train_dataset_tokenized = self.prepare_dataset(train_dataset)
        val_dataset_tokenized = self.prepare_dataset(val_dataset)
        
        # Setup training arguments
        output_dir = self.checkpoints_dir / f"generation_{generation}_{condition}"
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config["num_epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            warmup_steps=self.config["warmup_steps"],
            learning_rate=self.config["learning_rate"],
            logging_dir=str(self.logs_dir / f"generation_{generation}_{condition}"),
            logging_steps=self.config["logging"]["log_interval"],
            save_steps=self.config["logging"]["save_interval"],
            evaluation_strategy="steps",
            eval_steps=self.config["logging"]["save_interval"],
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb for this demo
            remove_unused_columns=False
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal language modeling, not masked
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_tokenized,
            eval_dataset=val_dataset_tokenized,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save the final model
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Extract training metrics
        training_metrics = {
            "final_train_loss": train_result.training_loss,
            "total_train_time": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "generation": generation,
            "condition": condition,
            "model_path": str(final_model_path)
        }
        
        # Store training history
        history_key = f"generation_{generation}_{condition}"
        self.training_history[history_key] = training_metrics
        
        logger.info(f"Training completed - Final loss: {training_metrics['final_train_loss']:.4f}")
        
        return str(final_model_path), training_metrics
    
    def generate_text(self, 
                     model_path: str, 
                     prompts: List[str], 
                     max_length: int = 200) -> List[str]:
        """Generate text using a trained model."""
        
        logger.info(f"Generating text with model: {model_path}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        model.to(self.device)
        
        generated_texts = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Format prompt
                formatted_prompt = f"Question: {prompt}\nAnswer:"
                
                # Tokenize
                inputs = self.tokenizer.encode(
                    formatted_prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config["max_sequence_length"] // 2
                )
                inputs = inputs.to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=min(max_length, self.config["max_sequence_length"]),
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True
                    )
                
                # Decode generated text
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract answer part (remove the prompt)
                if "Answer:" in generated:
                    answer = generated.split("Answer:", 1)[1].strip()
                else:
                    answer = generated.replace(formatted_prompt, "").strip()
                
                generated_texts.append(answer)
        
        logger.info(f"Generated {len(generated_texts)} text samples")
        return generated_texts
    
    def run_multi_generation_experiment(self, 
                                      base_dataset: Dict[str, Dataset],
                                      data_generator) -> Dict[str, Any]:
        """Run the complete multi-generation experiment."""
        
        logger.info("Starting multi-generation experiment")
        
        experiment_results = {
            "training_metrics": {},
            "generated_data": {},
            "model_paths": {}
        }
        
        conditions = list(self.config["conditions"].keys())
        num_generations = self.config["num_generations"]
        
        # Initialize data for each condition
        current_datasets = {}
        model_paths = {}
        
        for condition in conditions:
            current_datasets[condition] = base_dataset.copy()
            model_paths[condition] = None
        
        # Train across generations
        for generation in range(1, num_generations + 1):
            logger.info(f"\\n=== Starting Generation {generation} ===")
            
            generation_results = {}
            
            for condition in conditions:
                condition_config = self.config["conditions"][condition]
                
                # Prepare training data based on condition
                if generation == 1:
                    # First generation uses base human data
                    train_data = current_datasets[condition]["train"]
                    val_data = current_datasets[condition]["validation"]
                else:
                    # Subsequent generations use mixed data based on condition
                    train_data = self._prepare_mixed_dataset(
                        current_datasets[condition],
                        condition_config["data_split"],
                        generation,
                        condition
                    )
                    val_data = current_datasets[condition]["validation"]
                
                # Train model for this generation and condition
                model_path, training_metrics = self.train_generation(
                    train_dataset=train_data,
                    val_dataset=val_data,
                    generation=generation,
                    condition=condition,
                    base_model_path=model_paths[condition]
                )
                
                # Store results
                model_paths[condition] = model_path
                generation_results[condition] = {
                    "model_path": model_path,
                    "training_metrics": training_metrics
                }
                
                # Generate data for next generation (if not the last generation)
                if generation < num_generations:
                    test_prompts = [example["input"] for example in base_dataset["test"]]
                    generated_outputs = self.generate_text(model_path, test_prompts)
                    
                    # Create dataset for next generation
                    generated_data = []
                    for i, (prompt, output) in enumerate(zip(test_prompts, generated_outputs)):
                        generated_data.append({
                            "input": prompt,
                            "output": output,
                            "task_type": base_dataset["test"][i]["task_type"],
                            "generation": generation,
                            "condition": condition
                        })
                    
                    # Store generated data
                    experiment_results["generated_data"][f"gen_{generation}_{condition}"] = generated_data
                    
                    # Update dataset for next generation
                    current_datasets[condition]["train"] = Dataset.from_list(generated_data)
            
            # Store generation results
            experiment_results["training_metrics"][f"generation_{generation}"] = generation_results
            experiment_results["model_paths"][f"generation_{generation}"] = {cond: res["model_path"] for cond, res in generation_results.items()}
        
        # Save experiment results
        results_path = self.checkpoints_dir / "experiment_results.json"
        with open(results_path, 'w') as f:
            # Convert paths to strings for JSON serialization
            serializable_results = self._make_json_serializable(experiment_results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Multi-generation experiment completed. Results saved to {results_path}")
        return experiment_results
    
    def _prepare_mixed_dataset(self, 
                              datasets: Dict[str, Dataset], 
                              data_split: Dict[str, float],
                              generation: int,
                              condition: str) -> Dataset:
        """Prepare mixed dataset based on condition data split configuration."""
        
        # For simplicity, use the current training data
        # In a full implementation, this would mix human and predecessor data
        return datasets["train"]
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj
    
    def save_training_history(self, filename: str = "training_history.json"):
        """Save training history to file."""
        history_path = self.logs_dir / filename
        
        serializable_history = self._make_json_serializable(self.training_history)
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")

if __name__ == "__main__":
    from config import CONFIG
    from data_generator import MultiGenerationDataGenerator
    
    # Initialize components
    trainer = MultiGenerationTrainer(CONFIG)
    data_generator = MultiGenerationDataGenerator(CONFIG)
    
    # Generate base dataset
    logger.info("Generating base dataset...")
    base_dataset = data_generator.generate_base_human_data()
    
    # Test single generation training
    logger.info("Testing single generation training...")
    
    model_path, metrics = trainer.train_generation(
        train_dataset=base_dataset["train"].select(range(10)),  # Small subset for testing
        val_dataset=base_dataset["validation"].select(range(5)),
        generation=1,
        condition="control"
    )
    
    print(f"Training completed. Model saved to: {model_path}")
    print(f"Training metrics: {metrics}")
    
    # Test text generation
    test_prompts = ["What is the capital of France?", "How does photosynthesis work?"]
    generated_texts = trainer.generate_text(model_path, test_prompts)
    
    print("\nGenerated texts:")
    for prompt, generated in zip(test_prompts, generated_texts):
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print()