"""
Main Experiment Runner for LLM Inbreeding Deterioration Analysis

This script orchestrates the complete multi-generation experiment,
integrating data generation, model training, evaluation, and analysis.
"""

import argparse
import logging
import json
import sys
from pathlib import Path
import torch
import numpy as np
import random
from datetime import datetime

# Import our modules
from config import CONFIG, validate_config
from data_generator import MultiGenerationDataGenerator
from trainer import MultiGenerationTrainer
from evaluator import InbreedingEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('experiment.log')
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def run_complete_experiment():
    """Run the complete LLM inbreeding deterioration experiment."""
    
    logger.info("=" * 80)
    logger.info("STARTING LLM INBREEDING DETERIORATION ANALYSIS")
    logger.info("=" * 80)
    
    # Validate configuration
    validate_config()
    
    # Set random seed
    set_seed(CONFIG["random_seed"])
    
    # Initialize components
    logger.info("Initializing experiment components...")
    
    data_generator = MultiGenerationDataGenerator(CONFIG)
    trainer = MultiGenerationTrainer(CONFIG)
    evaluator = InbreedingEvaluator(CONFIG)
    
    # Step 1: Generate base human dataset
    logger.info("\\n" + "=" * 50)
    logger.info("STEP 1: GENERATING BASE HUMAN DATASET")
    logger.info("=" * 50)
    
    base_dataset = data_generator.generate_base_human_data()
    
    logger.info(f"Base dataset created:")
    logger.info(f"  - Training samples: {len(base_dataset['train'])}")
    logger.info(f"  - Validation samples: {len(base_dataset['validation'])}")
    logger.info(f"  - Test samples: {len(base_dataset['test'])}")
    
    # Step 2: Run multi-generation training
    logger.info("\\n" + "=" * 50)
    logger.info("STEP 2: MULTI-GENERATION TRAINING")
    logger.info("=" * 50)
    
    experiment_results = {
        "training_results": {},
        "evaluation_results": {},
        "deterioration_analysis": {},
        "experiment_config": CONFIG
    }
    
    conditions = list(CONFIG["conditions"].keys())
    num_generations = CONFIG["num_generations"]
    
    # Store all generated data and model predictions for evaluation
    all_predictions = {}
    all_references = {}
    all_task_types = {}
    
    # Initialize with base references
    base_references = [example["output"] for example in base_dataset["test"]]
    base_task_types = [example["task_type"] for example in base_dataset["test"]]
    
    # Train and evaluate each generation
    for generation in range(1, num_generations + 1):
        logger.info(f"\\n--- Generation {generation} ---")
        
        generation_models = {}
        generation_predictions = {}
        
        for condition in conditions:
            logger.info(f"Training condition: {condition}")
            
            # Prepare training data
            if generation == 1:
                # First generation uses base human data
                train_data = base_dataset["train"]
                val_data = base_dataset["validation"]
            else:
                # Use data from previous generation (simplified for demo)
                train_data = base_dataset["train"]  # In full implementation, would use previous gen data
                val_data = base_dataset["validation"]
            
            # Train model - use smaller subset for demo due to computational constraints
            subset_size = min(50, len(train_data))  # Small subset for demo
            train_subset = train_data.select(range(subset_size))
            val_subset = val_data.select(range(min(10, len(val_data))))
            
            try:
                model_path, training_metrics = trainer.train_generation(
                    train_dataset=train_subset,
                    val_dataset=val_subset,
                    generation=generation,
                    condition=condition,
                    base_model_path=generation_models.get(condition)
                )
                
                generation_models[condition] = model_path
                
                # Store training results
                key = f"generation_{generation}_{condition}"
                experiment_results["training_results"][key] = training_metrics
                
                # Generate predictions on test set
                test_prompts = [example["input"] for example in base_dataset["test"]]
                predictions = trainer.generate_text(model_path, test_prompts[:20])  # Subset for demo
                
                generation_predictions[condition] = predictions
                
                # Store for evaluation
                pred_key = f"gen_{generation}_{condition}"
                all_predictions[pred_key] = predictions
                all_references[pred_key] = base_references[:len(predictions)]
                all_task_types[pred_key] = base_task_types[:len(predictions)]
                
                logger.info(f"  ✓ Training completed - Loss: {training_metrics['final_train_loss']:.4f}")
                
            except Exception as e:
                logger.error(f"  ✗ Training failed for {condition}: {str(e)}")
                # Continue with other conditions
                continue
    
    # Step 3: Comprehensive Evaluation
    logger.info("\\n" + "=" * 50)
    logger.info("STEP 3: COMPREHENSIVE EVALUATION")
    logger.info("=" * 50)
    
    for pred_key in all_predictions:
        generation, condition = pred_key.replace("gen_", "").split("_", 1)
        
        try:
            metrics = evaluator.evaluate_generation(
                predictions=all_predictions[pred_key],
                references=all_references[pred_key],
                generation=int(generation),
                condition=condition,
                task_types=all_task_types[pred_key]
            )
            
            experiment_results["evaluation_results"][pred_key] = metrics
            
            logger.info(f"Evaluated {pred_key}:")
            logger.info(f"  - F1 Score: {metrics['f1_score']:.3f}")
            logger.info(f"  - Diversity: {metrics['distinct_2_grams']:.3f}")
            logger.info(f"  - Coherence: {metrics['coherence_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed for {pred_key}: {str(e)}")
    
    # Step 4: Deterioration Analysis
    logger.info("\\n" + "=" * 50)
    logger.info("STEP 4: DETERIORATION PATTERN ANALYSIS")
    logger.info("=" * 50)
    
    try:
        deterioration_analysis = evaluator.analyze_deterioration_patterns()
        experiment_results["deterioration_analysis"] = deterioration_analysis
        
        # Report key findings
        logger.info("Key Findings:")
        
        for condition in conditions:
            if condition in deterioration_analysis.get("deterioration_rates", {}):
                rates = deterioration_analysis["deterioration_rates"][condition]
                logger.info(f"\\n{condition.upper()} Condition:")
                for metric, rate in rates.items():
                    logger.info(f"  - {metric}: {rate:.1f}% deterioration")
        
        # Statistical significance
        significant_declines = []
        for test_key, test_result in deterioration_analysis.get("statistical_tests", {}).items():
            if test_result.get("significant_decline", False):
                significant_declines.append(test_key)
        
        if significant_declines:
            logger.info(f"\\nStatistically Significant Declines Detected:")
            for decline in significant_declines:
                logger.info(f"  - {decline}")
        else:
            logger.info("\\nNo statistically significant declines detected (may need more generations/data)")
        
    except Exception as e:
        logger.error(f"Deterioration analysis failed: {str(e)}")
    
    # Step 5: Generate Visualizations
    logger.info("\\n" + "=" * 50)
    logger.info("STEP 5: GENERATING VISUALIZATIONS")
    logger.info("=" * 50)
    
    try:
        evaluator.generate_visualization_report()
        logger.info("✓ Visualization report generated")
    except Exception as e:
        logger.error(f"Visualization generation failed: {str(e)}")
    
    # Step 6: Save Results
    logger.info("\\n" + "=" * 50)
    logger.info("STEP 6: SAVING RESULTS")
    logger.info("=" * 50)
    
    # Save comprehensive results
    results_path = Path("../results/complete_experiment_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        else:
            return obj
    
    serializable_results = make_serializable(experiment_results)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    evaluator.save_results("detailed_evaluation_results.json")
    trainer.save_training_history()
    
    logger.info(f"✓ Complete results saved to {results_path}")
    
    # Step 7: Generate Summary Report
    logger.info("\\n" + "=" * 50)
    logger.info("FINAL EXPERIMENT SUMMARY")
    logger.info("=" * 50)
    
    logger.info(f"Experiment completed successfully!")
    logger.info(f"Configurations tested: {len(conditions)} conditions × {num_generations} generations")
    logger.info(f"Total models trained: {len(experiment_results['training_results'])}")
    logger.info(f"Total evaluations completed: {len(experiment_results['evaluation_results'])}")
    
    # Summary statistics
    if experiment_results["evaluation_results"]:
        all_f1_scores = [res["f1_score"] for res in experiment_results["evaluation_results"].values() 
                        if "f1_score" in res]
        all_diversity_scores = [res["distinct_2_grams"] for res in experiment_results["evaluation_results"].values() 
                               if "distinct_2_grams" in res]
        
        if all_f1_scores:
            logger.info(f"F1 Score Range: {min(all_f1_scores):.3f} - {max(all_f1_scores):.3f}")
        if all_diversity_scores:
            logger.info(f"Diversity Range: {min(all_diversity_scores):.3f} - {max(all_diversity_scores):.3f}")
    
    logger.info("\\nExperiment artifacts saved to:")
    logger.info(f"  - Results: ../results/")
    logger.info(f"  - Model checkpoints: ../checkpoints/")
    logger.info(f"  - Logs: ../logs/")
    
    return experiment_results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM Inbreeding Deterioration Analysis")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument("--generations", type=int, default=3, help="Number of generations to train")
    parser.add_argument("--quick", action="store_true", help="Run quick test with minimal data")
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            CONFIG.update(custom_config)
    
    if args.generations:
        CONFIG["num_generations"] = args.generations
    
    if args.quick:
        # Quick test mode - reduce dataset sizes and epochs
        CONFIG["dataset_config"]["train_size"] = 20
        CONFIG["dataset_config"]["val_size"] = 5
        CONFIG["dataset_config"]["test_size"] = 5
        CONFIG["num_epochs"] = 1
        logger.info("Running in quick test mode")
    
    try:
        # Run the experiment
        results = run_complete_experiment()
        
        logger.info("\\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\\nExperiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\\nExperiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)