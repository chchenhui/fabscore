"""Fine-tuning trainer for arithmetic reasoning models."""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import wandb
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class TrainingConfig:
    """Configuration for arithmetic model training."""
    
    model_name: str = "Qwen/Qwen3-0.6B"
    max_length: int = 512
    learning_rate: float = 2e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Output directories
    output_dir: str = "results/models"
    logging_dir: str = "results/logs"
    
    # Wandb configuration
    use_wandb: bool = False
    wandb_project: str = "llm-arithmetic"
    wandb_run_name: Optional[str] = None


class ArithmeticDataset(Dataset):
    """Dataset for arithmetic training examples."""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Create input-output format for causal LM
        input_text = example["input_text"]
        target_text = example["target_text"]
        
        # Format as instruction-following
        full_text = f"### Instruction:\n{input_text}\n\n### Response:\n{target_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        # Mask the instruction part (only learn from the response)
        instruction_part = f"### Instruction:\n{input_text}\n\n### Response:\n"
        instruction_encoding = self.tokenizer(
            instruction_part,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        instruction_length = instruction_encoding["input_ids"].shape[1]
        labels[:instruction_length] = -100  # Ignore instruction tokens in loss
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class ArithmeticTrainer:
    """Trainer for arithmetic reasoning models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        self._setup_model()
    
    def _setup_model(self):
        """Initialize model and tokenizer."""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Apply LoRA if configured
        if self.config.use_lora:
            self.logger.info("Applying LoRA configuration")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
    
    def load_training_data(self, data_file: str) -> List[Dict[str, Any]]:
        """Load training data from JSON file."""
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded {len(data)} training examples from {data_file}")
        return data
    
    def create_datasets(self, train_data: List[Dict[str, Any]], 
                       eval_data: Optional[List[Dict[str, Any]]] = None) -> Tuple[ArithmeticDataset, Optional[ArithmeticDataset]]:
        """Create training and evaluation datasets."""
        train_dataset = ArithmeticDataset(train_data, self.tokenizer, self.config.max_length)
        
        eval_dataset = None
        if eval_data:
            eval_dataset = ArithmeticDataset(eval_data, self.tokenizer, self.config.max_length)
        
        return train_dataset, eval_dataset
    
    def train_single_stage(self, train_dataset: ArithmeticDataset, 
                          eval_dataset: Optional[ArithmeticDataset] = None,
                          stage_name: str = "single_stage") -> str:
        """Train a single stage of the curriculum."""
        self.logger.info(f"Starting training stage: {stage_name}")
        
        # Setup output directory for this stage
        stage_output_dir = Path(self.config.output_dir) / stage_name
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if configured
        if self.config.use_wandb:
            run_name = f"{stage_name}_{self.config.wandb_run_name}" if self.config.wandb_run_name else stage_name
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config=self.config.__dict__
            )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(stage_output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
            report_to="wandb" if self.config.use_wandb else None,
            logging_dir=self.config.logging_dir,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        self.logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save model
        final_model_path = stage_output_dir / "final_model"
        self.trainer.save_model(str(final_model_path))
        
        # Save training metrics
        metrics_file = stage_output_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        self.logger.info(f"Training completed. Model saved to {final_model_path}")
        
        if self.config.use_wandb:
            wandb.finish()
        
        return str(final_model_path)
    
    def train_curriculum(self, curriculum_data: Dict[str, str], 
                        validation_data: Optional[str] = None) -> List[str]:
        """Train using curriculum learning."""
        self.logger.info("Starting curriculum learning training")
        
        model_paths = []
        eval_data = None
        
        if validation_data:
            eval_data = self.load_training_data(validation_data)
        
        for stage_name, data_file in curriculum_data.items():
            self.logger.info(f"Training curriculum stage: {stage_name}")
            
            # Load data for this stage
            train_data = self.load_training_data(data_file)
            
            # Create datasets
            train_dataset, eval_dataset = self.create_datasets(train_data, eval_data)
            
            # Train this stage
            model_path = self.train_single_stage(train_dataset, eval_dataset, stage_name)
            model_paths.append(model_path)
            
            self.logger.info(f"Completed stage {stage_name}, model saved to {model_path}")
        
        return model_paths
    
    def evaluate_model(self, model_path: str, eval_data: str) -> Dict[str, float]:
        """Evaluate a trained model."""
        self.logger.info(f"Evaluating model: {model_path}")
        
        # Load evaluation data
        eval_examples = self.load_training_data(eval_data)
        _, eval_dataset = self.create_datasets([], eval_examples)
        
        # Load the trained model
        if self.config.use_lora:
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Setup trainer for evaluation
        training_args = TrainingArguments(
            output_dir="temp_eval",
            per_device_eval_batch_size=self.config.batch_size,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        self.logger.info(f"Evaluation results: {eval_result}")
        return eval_result