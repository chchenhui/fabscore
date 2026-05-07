#!/usr/bin/env python3
"""Phase 1: Multi-model evaluation experiment."""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llm_arithmetic.evaluation.model_evaluator import create_evaluator
from llm_arithmetic.evaluation.benchmark_generator import BenchmarkGenerator
from llm_arithmetic.evaluation.base_evaluator import EvaluationResult
from llm_arithmetic.evaluation.prompt_configs import PromptType, get_available_prompt_types


def setup_directories(task: str = "basic"):
    """Create necessary directories for results."""
    base_results_dir = Path(__file__).parent.parent.parent / "results" / "phase1"
    results_dir = base_results_dir / task
    results_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "benchmarks"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir, data_dir


def generate_benchmark(data_dir: Path, force_regenerate: bool = False, task: str = "basic"):
    """Generate or load the benchmark based on task."""
    if task == "math401":
        # Use the actual MATH 401 dataset
        math401_file = Path(__file__).parent.parent.parent / "data" / "math401" / "math401_generated.json"
        
        if math401_file.exists():
            print(f"Loading MATH 401 dataset from {math401_file}")
            with open(math401_file, 'r') as f:
                data = json.load(f)
            
            from llm_arithmetic.evaluation.base_evaluator import ArithmeticProblem
            problems = []
            for item in data:
                problems.append(ArithmeticProblem(
                    problem=item["problem"],
                    answer=item["answer"],
                    operation=item["operation"],
                    operands=item["operands"],
                    difficulty=item["difficulty"],
                    metadata=item.get("metadata")
                ))
            return problems
        else:
            print(f"MATH 401 dataset not found at {math401_file}")
            print("Falling back to basic benchmark...")
    
    # Basic benchmark logic (default)
    benchmark_file = data_dir / "math401_plus.json"
    
    if benchmark_file.exists() and not force_regenerate:
        print(f"Loading existing benchmark from {benchmark_file}")
        with open(benchmark_file, 'r') as f:
            data = json.load(f)
        
        from llm_arithmetic.evaluation.base_evaluator import ArithmeticProblem
        problems = []
        for item in data:
            problems.append(ArithmeticProblem(
                problem=item["problem"],
                answer=item["answer"],
                operation=item["operation"],
                operands=item["operands"],
                difficulty=item["difficulty"],
                metadata=item.get("metadata")
            ))
        return problems
    
    print("Generating new MATH 401+ benchmark...")
    generator = BenchmarkGenerator(seed=42)
    problems = generator.generate_math401_benchmark()
    generator.save_benchmark(problems, str(benchmark_file))
    
    return problems


def run_single_evaluation(provider: str, model_name: str, problems, results_dir: Path, 
                         verbose: bool = True, base_generation_kwargs: dict = None):
    """Run evaluation for a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating {provider}/{model_name}")
    print(f"{'='*60}")
    
    # Customize generation kwargs for different providers
    if base_generation_kwargs is None:
        base_generation_kwargs = {}
    
    generation_kwargs = base_generation_kwargs.copy()
    
    # Provider-specific parameter handling
    if provider.lower() in ["huggingface", "hf"]:
        # HuggingFace models use max_new_tokens instead of max_tokens
        if 'max_tokens' in generation_kwargs:
            generation_kwargs['max_new_tokens'] = generation_kwargs.pop('max_tokens')
        # Keep do_sample and top_p for HuggingFace models
    else:
        # API providers (OpenAI, Anthropic, Together) don't support sampling parameters
        generation_kwargs.pop('do_sample', None)
        generation_kwargs.pop('top_p', None)
        
        # Also ensure max_new_tokens is converted back to max_tokens for API providers
        if 'max_new_tokens' in generation_kwargs:
            generation_kwargs['max_tokens'] = generation_kwargs.pop('max_new_tokens')
    
    try:
        evaluator = create_evaluator(provider, model_name, verbose=verbose, **generation_kwargs)
        results = evaluator.evaluate_problems(problems)
        
        # Create filename-safe model name and include prompt type
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        prompt_type_str = generation_kwargs.get('prompt_type', 'step_by_step_boxed')
        if hasattr(prompt_type_str, 'value'):
            prompt_type_str = prompt_type_str.value
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{provider}_{safe_model_name}_{prompt_type_str}_{timestamp}.json"
        
        evaluator.save_results(results, str(results_file))
        
        print(f"\nResults for {provider}/{model_name}:")
        print(f"  Total problems: {results.total_problems}")
        print(f"  Correct answers: {results.correct_answers}")
        print(f"  Accuracy: {results.accuracy:.3f}")
        print(f"  Avg response time: {results.avg_response_time:.2f}s")
        
        print("\nAccuracy by operation:")
        for op, stats in results.results_by_operation.items():
            print(f"  {op}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
        
        print("\nAccuracy by difficulty:")
        for diff, stats in results.results_by_difficulty.items():
            print(f"  {diff}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
        
        return results
    
    except Exception as e:
        print(f"Error evaluating {provider}/{model_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 multi-model evaluation")
    parser.add_argument("--models", nargs="+", 
                       default=["openai/gpt-4", "anthropic/claude-3-5-sonnet-20241022"],
                       help="Models to evaluate in format provider/model_name")
    parser.add_argument("--benchmark-size", type=int, default=211,
                       help="Number of problems to evaluate (subset of full benchmark)")
    parser.add_argument("--regenerate-benchmark", action="store_true",
                       help="Force regeneration of benchmark")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose output")
    parser.add_argument("--task", choices=["basic", "math401"], default="basic",
                       help="Task/dataset to evaluate on")
    parser.add_argument("--max-tokens", type=int, default=4000,
                       help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for generation (lower = more deterministic)")
    parser.add_argument("--prompt-type", choices=get_available_prompt_types(), 
                       default="step_by_step_boxed",
                       help="Type of prompt to use for evaluation")
    
    args = parser.parse_args()
    
    # Setup
    results_dir, data_dir = setup_directories(args.task)
    
    # Generate benchmark
    all_problems = generate_benchmark(data_dir, args.regenerate_benchmark, args.task)
    
    # Use subset if requested
    if args.benchmark_size < len(all_problems):
        import random
        random.seed(42)
        problems = random.sample(all_problems, args.benchmark_size)
        print(f"Using {len(problems)} problems (subset of {len(all_problems)})")
    else:
        problems = all_problems
        print(f"Using all {len(problems)} problems")
    
    # Set up generation kwargs based on command line args
    base_generation_kwargs = {}
    if hasattr(args, 'max_tokens') and args.max_tokens:
        base_generation_kwargs['max_tokens'] = args.max_tokens
    if hasattr(args, 'temperature') and args.temperature is not None:
        base_generation_kwargs['temperature'] = args.temperature
        if args.temperature > 0:  # Only add sampling params if temperature > 0
            base_generation_kwargs['do_sample'] = True
            base_generation_kwargs['top_p'] = 0.9
    
    # Add prompt type
    base_generation_kwargs['prompt_type'] = PromptType(args.prompt_type)
    
    # Extract prompt type string for filenames
    prompt_type_str = args.prompt_type
    
    # Run evaluations
    all_results = {}
    
    for model_spec in args.models:
        try:
            provider, model_name = model_spec.split("/", 1)
        except ValueError:
            print(f"Invalid model specification: {model_spec}. Use format provider/model_name")
            continue
        
        result = run_single_evaluation(provider, model_name, problems, results_dir, args.verbose, base_generation_kwargs)
        if result:
            all_results[f"{provider}/{model_name}"] = result
    
    # Create summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    if all_results:
        print(f"\n{'Model':<40} {'Accuracy':<10} {'Avg Time':<12}")
        print("-" * 65)
        
        for model_name, result in all_results.items():
            print(f"{model_name:<40} {result.accuracy:<10.3f} {result.avg_response_time:<12.2f}s")
        
        # Save summary
        summary_file = results_dir / f"evaluation_summary_{args.prompt_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_size": len(problems),
            "prompt_type": args.prompt_type,
            "models_evaluated": list(all_results.keys()),
            "results_summary": {
                model_name: {
                    "accuracy": result.accuracy,
                    "avg_response_time": result.avg_response_time,
                    "results_by_operation": result.results_by_operation,
                    "results_by_difficulty": result.results_by_difficulty,
                    "metadata": result.metadata
                }
                for model_name, result in all_results.items()
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
    
    print(f"\nIndividual results saved in: {results_dir}")


if __name__ == "__main__":
    main()