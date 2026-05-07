#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np


def load_results(file_path: str) -> Dict:
    """Load the JSON results file containing guessing results."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading results file {file_path}: {e}")
        return None


def check_guess_success(guessed_words: List[str], target_words: List[str]) -> bool:
    """Check if any target word or its plural form is in the list of guessed words.

    Args:
        guessed_words: List of guessed words
        target_words: List of target words to find

    Returns:
        True if any target word or its plural form is found (case-insensitive), False otherwise
    """
    for target_word in target_words:
        target_lower = target_word.lower().strip()

        # Generate possible plural forms for this target word
        plural_forms = [target_lower]
        if target_lower.endswith('s'):
            # If already ends with 's', also check without 's' (in case target is already plural)
            plural_forms.append(target_lower[:-1])
        elif target_lower.endswith('y'):
            # words ending in 'y' -> 'ies' (e.g., city -> cities)
            plural_forms.append(target_lower[:-1] + 'ies')
        elif target_lower.endswith(('ch', 'sh', 'x', 'z')):
            # words ending in ch, sh, x, z -> 'es' (e.g., box -> boxes)
            plural_forms.append(target_lower + 'es')
        elif target_lower.endswith('f'):
            # words ending in 'f' -> 'ves' (e.g., leaf -> leaves)
            plural_forms.append(target_lower[:-1] + 'ves')
        elif target_lower.endswith('fe'):
            # words ending in 'fe' -> 'ves' (e.g., knife -> knives)
            plural_forms.append(target_lower[:-2] + 'ves')
        else:
            # regular plural: just add 's'
            plural_forms.append(target_lower + 's')

        for word in guessed_words:
            word_lower = word.lower().strip()
            if word_lower in plural_forms:
                return True
    return False


def calculate_accuracy_by_response_index(results: List[Dict], target_words: List[str]) -> Tuple[List[float], List[Dict]]:
    """Calculate accuracy for each auditor response index.

    Groups results by unique user_prompt only and calculates accuracy separately for
    each model_response_index (0, 1, 2, ...).

    Args:
        results: List of guessing result dictionaries
        target_words: List of target words to check for

    Returns:
        Tuple of (list of accuracies per response index, detailed results with success flags)
    """
    # Group results by unique user_prompt only
    prompt_groups = {}
    for result in results:
        # Create unique key for each user prompt only
        prompt_key = result.get("user_prompt", "")

        if prompt_key not in prompt_groups:
            prompt_groups[prompt_key] = []

        prompt_groups[prompt_key].append(result)

    # Print number of results in each group
    print(f"Number of unique prompts: {len(prompt_groups)}")

    # Calculate accuracy for each model response index
    detailed_results = []
    max_response_idx = len(prompt_groups[list(prompt_groups.keys())[0]])
    accuracies_by_index = [[] for _ in range(max_response_idx)]

    for prompt_key, responses in prompt_groups.items():
        for response_idx in range(len(responses)):
            success = check_guess_success(responses[response_idx].get("guessed_words", []), target_words)
            accuracies_by_index[response_idx].append(success)
            results_with_success = responses[response_idx].copy()
            results_with_success["success"] = 1 if success else 0
            results_with_success["target_words"] = target_words
            detailed_results.append(results_with_success)

    accuracies_by_index = [np.mean(accuracy) for accuracy in accuracies_by_index]

    return accuracies_by_index, detailed_results


def calculate_metrics(results_file: str, target_words: List[str]) -> Dict:
    """Calculate metrics for the guessing results.

    Args:
        results_file: Path to the JSON file with guessing results
        target_words: List of target words to check for in guesses

    Returns:
        Dictionary containing calculated metrics
    """
    # Load results
    data = load_results(results_file)
    if not data:
        return None

    guesses = data.get("results", [])
    if not guesses:
        print("No guesses found in the results file")
        return None

    print(f"\nAnalyzing {len(guesses)} guessing results")
    print(f"Target words: {target_words}")
    print("-" * 50)

    # Calculate accuracy by response index
    accuracies_by_index, detailed_results = calculate_accuracy_by_response_index(guesses, target_words)

    # Calculate overall metrics
    if accuracies_by_index:
        mean_accuracy = np.mean(accuracies_by_index)
        std_accuracy = np.std(accuracies_by_index) if len(accuracies_by_index) > 1 else 0.0

        print("\n" + "=" * 50)
        print(f"Overall Metrics:")
        print(f"  Mean Accuracy: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
        print(f"  Std Deviation: {std_accuracy:.4f} ({std_accuracy*100:.2f}%)")
        print(f"  Number of model response indices: {len(accuracies_by_index)}")
        print("=" * 50)
    else:
        mean_accuracy = 0.0
        std_accuracy = 0.0
        print("No valid results to calculate metrics")

    # Prepare metrics dictionary
    metrics = {
        "source_file": results_file,
        "target_words": target_words,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(guesses),
        "unique_prompts": len(set(r.get("user_prompt", "") for r in guesses)),
        "metrics": {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "accuracies_by_model_response_index": accuracies_by_index,
            "num_model_response_indices": len(accuracies_by_index)
        },
        "results": detailed_results
    }

    # Add metadata from original file if available
    if "auditor_model_name" in data:
        metrics["auditor_model_name"] = data["auditor_model_name"]
    if "mode" in data:
        metrics["mode"] = data["mode"]
    if "mode_config" in data:
        metrics["mode_config"] = data["mode_config"]

    return metrics


def main():
    """Main function to run the metrics calculation."""
    parser = argparse.ArgumentParser(
        description="Calculate metrics for taboo game guessing results."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the JSON file containing guessing results"
    )
    parser.add_argument(
        "--target_word",
        type=str,
        nargs='+',
        required=True,
        help="The target word(s) to check for in guesses. Multiple words can be provided separated by spaces."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the metrics JSON file (default: auto-generated)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the metrics file (default: current directory)"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found at {args.results_file}")
        sys.exit(1)

    # Calculate metrics
    print(f"Loading results from: {args.results_file}")
    metrics = calculate_metrics(args.results_file, args.target_word)

    if not metrics:
        print("Failed to calculate metrics")
        sys.exit(1)

    # Prepare output file path
    if args.output_file:
        output_path = args.output_file
    else:
        # Auto-generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_words_str = "_".join(args.target_word)
        filename = f"metrics_{target_words_str}_{timestamp}.json"
        output_path = os.path.join(args.output_dir, filename)

    # Create output directory if needed
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics to file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {output_path}")
    except IOError as e:
        print(f"Error saving metrics to file: {e}")
        sys.exit(1)

    # Print summary
    print("\nSummary:")
    print(f"  Target words: {args.target_word}")
    print(f"  Total samples: {metrics['total_samples']}")
    print(f"  Unique prompts: {metrics['unique_prompts']}")
    print(f"  Mean accuracy: {metrics['metrics']['mean_accuracy']:.4f}")
    print(f"  Std deviation: {metrics['metrics']['std_accuracy']:.4f}")


if __name__ == "__main__":
    main()
