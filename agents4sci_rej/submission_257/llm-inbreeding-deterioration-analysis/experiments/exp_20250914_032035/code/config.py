"""
Configuration file for LLM Inbreeding Deterioration Analysis Experiment

This file contains all experimental parameters, model configurations,
and evaluation settings for the multi-generation training study.
"""

import torch
from pathlib import Path

# Base configuration
CONFIG = {
    # Experimental setup
    "experiment_name": "multi_generation_degradation",
    "num_generations": 5,
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Model configuration
    "base_model_name": "distilbert-base-uncased",  # Using smaller model for feasibility
    "max_sequence_length": 512,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    
    # Training conditions
    "conditions": {
        "exclusive": {
            "name": "exclusive_predecessor",
            "description": "Trained only on previous generation outputs",
            "data_split": {"predecessor": 1.0, "human": 0.0}
        },
        "mixed": {
            "name": "mixed_training", 
            "description": "Trained on mix of human and predecessor data",
            "data_split": {"predecessor": 0.5, "human": 0.5}
        },
        "control": {
            "name": "human_only_control",
            "description": "Trained only on human data",
            "data_split": {"predecessor": 0.0, "human": 1.0}
        }
    },
    
    # Dataset configuration
    "dataset_config": {
        "train_size": 10000,  # Size per condition per generation
        "val_size": 2000,
        "test_size": 2000,
        "text_tasks": [
            "question_answering",
            "summarization", 
            "creative_writing",
            "factual_completion",
            "reasoning"
        ]
    },
    
    # Evaluation metrics
    "evaluation_metrics": {
        "language_quality": ["perplexity", "fluency_score"],
        "factual_accuracy": ["exact_match", "f1_score"],
        "diversity": ["distinct_ngrams", "entropy"],
        "coherence": ["coherence_score", "semantic_similarity"],
        "reasoning": ["logical_consistency", "problem_solving_accuracy"],
        "creativity": ["novelty_score", "semantic_diversity"]
    },
    
    # Output paths
    "paths": {
        "data_dir": Path("../data"),
        "results_dir": Path("../results"), 
        "checkpoints_dir": Path("../checkpoints"),
        "logs_dir": Path("../logs")
    },
    
    # Statistical analysis
    "statistics": {
        "significance_threshold": 0.05,
        "confidence_interval": 0.95,
        "bootstrap_iterations": 1000,
        "multiple_testing_correction": "bonferroni"
    },
    
    # Logging and monitoring
    "logging": {
        "log_level": "INFO",
        "log_interval": 100,
        "save_interval": 1000,
        "use_wandb": False  # Set to True if wandb account available
    }
}

# Validation function
def validate_config():
    """Validate configuration parameters"""
    assert CONFIG["num_generations"] >= 2, "Need at least 2 generations"
    assert CONFIG["batch_size"] > 0, "Batch size must be positive"
    assert 0 < CONFIG["learning_rate"] < 1, "Learning rate must be between 0 and 1"
    
    # Ensure data splits sum to 1.0 for each condition
    for condition, params in CONFIG["conditions"].items():
        total = sum(params["data_split"].values())
        assert abs(total - 1.0) < 1e-6, f"Data split for {condition} must sum to 1.0"
    
    print("Configuration validation passed ✅")

if __name__ == "__main__":
    validate_config()
    print("Configuration loaded successfully!")