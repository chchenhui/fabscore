#!/usr/bin/env python3
"""
Script to calculate hybrid accuracy from two JSON result files.
For each entry, success is 1 if either method has success=1 for that position.
"""

import json
import sys
from typing import Any, Dict, List


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and return JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def group_by_user_prompt(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by unique user_prompt only."""
    prompt_groups = {}
    for result in results:
        prompt_key = result.get("user_prompt", "")
        if prompt_key not in prompt_groups:
            prompt_groups[prompt_key] = []
        prompt_groups[prompt_key].append(result)
    return prompt_groups

def calculate_hybrid_accuracy_mismatched(file1_path: str, file2_path: str) -> None:
    """Calculate hybrid accuracy with special handling for mismatched response counts.

    When one file has single responses per prompt and another has multiple responses,
    calculates hybrid accuracy by response position with mean and std dev.
    """

    # Load both files
    data1 = load_json_file(file1_path)
    data2 = load_json_file(file2_path)

    # Extract detailed results
    results1 = data1.get("results", [])
    results2 = data2.get("results", [])

    # Group results by user_prompt for both methods
    groups1 = group_by_user_prompt(results1)
    groups2 = group_by_user_prompt(results2)

    # Find common prompts
    common_prompts = set(groups1.keys()) & set(groups2.keys())
    print(f"Common prompts: {len(common_prompts)}")

    # Analyze response count patterns
    response_counts1 = [len(groups1[prompt]) for prompt in common_prompts]
    response_counts2 = [len(groups2[prompt]) for prompt in common_prompts]

    print(f"File 1 response counts: min={min(response_counts1)}, max={max(response_counts1)}")
    print(f"File 2 response counts: min={min(response_counts2)}, max={max(response_counts2)}")

    # Determine which file has single vs multiple responses
    if (
        min(response_counts1) == max(response_counts1) == 1
        and max(response_counts2) > 1
    ):
        # File 1 has single responses, File 2 has multiple
        single_groups = groups1
        multi_groups = groups2
        single_file_num = 1
        multi_file_num = 2
        max_responses = max(response_counts2)
    elif (
        min(response_counts2) == max(response_counts2) == 1
        and max(response_counts1) > 1
    ):
        # File 2 has single responses, File 1 has multiple
        single_groups = groups2
        multi_groups = groups1
        single_file_num = 2
        multi_file_num = 1
        max_responses = max(response_counts1)
    else:
        # Both have same pattern, fall back to simple matching
        hybrid_successes = []
        individual_successes1 = []
        individual_successes2 = []

        for prompt in common_prompts:
            responses1 = groups1[prompt]
            responses2 = groups2[prompt]
            min_len = min(len(responses1), len(responses2))

            for i in range(min_len):
                success1 = responses1[i].get("success", 0)
                success2 = responses2[i].get("success", 0)
                hybrid_success = 1 if (success1 == 1 or success2 == 1) else 0

                hybrid_successes.append(hybrid_success)
                individual_successes1.append(success1)
                individual_successes2.append(success2)

        import numpy as np

        hybrid_accuracy = np.mean(hybrid_successes) if hybrid_successes else 0
        individual_accuracy1 = (
            np.mean(individual_successes1) if individual_successes1 else 0
        )
        individual_accuracy2 = (
            np.mean(individual_successes2) if individual_successes2 else 0
        )

        print("\n=== Hybrid Accuracy Results (Mismatched Mode - Equal Patterns) ===")
        print(f"File 1: {file1_path}")
        print(f"File 2: {file2_path}")
        print(f"Total comparisons: {len(hybrid_successes)}")

        print("\n=== Summary Statistics ===")
        print(
            f"Method 1 - Accuracy: {individual_accuracy1:.4f} ({individual_accuracy1 * 100:.2f}%)"
        )
        print(
            f"Method 2 - Accuracy: {individual_accuracy2:.4f} ({individual_accuracy2 * 100:.2f}%)"
        )
        print(
            f"Hybrid   - Accuracy: {hybrid_accuracy:.4f} ({hybrid_accuracy * 100:.2f}%)"
        )
        print(
            f"Improvement: +{(hybrid_accuracy - max(individual_accuracy1, individual_accuracy2)) * 100:.2f} percentage points"
        )

        print(f"\nHybrid (mean, std): ({hybrid_accuracy * 100:.4f}, 0.0000)")
        return

    # Calculate hybrid accuracies by response position
    # For each response position (0-9), we take all prompts and for each prompt:
    # - Use the single response from single_groups[prompt][0]
    # - Use the response at that position from multi_groups[prompt][position]
    # - Calculate hybrid success and collect all results for that position

    hybrid_accuracies_by_position = []
    individual_accuracies_single_by_position = []
    individual_accuracies_multi_by_position = []

    for response_pos in range(max_responses):
        hybrid_successes_pos = []
        single_successes_pos = []
        multi_successes_pos = []

        for prompt in common_prompts:
            single_responses = single_groups[prompt]
            multi_responses = multi_groups[prompt]

            # For each prompt, match the single response with the response at this position
            if len(single_responses) >= 1 and response_pos < len(multi_responses):
                single_success = single_responses[0].get(
                    "success", 0
                )  # Always the same single response for this prompt
                multi_success = multi_responses[response_pos].get(
                    "success", 0
                )  # Response at current position for this prompt
                hybrid_success = 1 if (single_success == 1 or multi_success == 1) else 0

                hybrid_successes_pos.append(hybrid_success)
                single_successes_pos.append(single_success)
                multi_successes_pos.append(multi_success)

        # Calculate accuracies for this response position across all prompts
        if hybrid_successes_pos:
            hybrid_acc = sum(hybrid_successes_pos) / len(hybrid_successes_pos)
            single_acc = sum(single_successes_pos) / len(single_successes_pos)
            multi_acc = sum(multi_successes_pos) / len(multi_successes_pos)

            hybrid_accuracies_by_position.append(hybrid_acc)
            individual_accuracies_single_by_position.append(single_acc)
            individual_accuracies_multi_by_position.append(multi_acc)

    # Calculate mean and std
    import numpy as np

    hybrid_mean = (
        np.mean(hybrid_accuracies_by_position) if hybrid_accuracies_by_position else 0
    )
    hybrid_std = (
        np.std(hybrid_accuracies_by_position)
        if len(hybrid_accuracies_by_position) > 1
        else 0
    )

    single_mean = np.mean(individual_accuracies_single_by_position) if individual_accuracies_single_by_position else 0
    single_std = np.std(individual_accuracies_single_by_position) if len(individual_accuracies_single_by_position) > 1 else 0

    multi_mean = (
        np.mean(individual_accuracies_multi_by_position)
        if individual_accuracies_multi_by_position
        else 0
    )
    multi_std = (
        np.std(individual_accuracies_multi_by_position)
        if len(individual_accuracies_multi_by_position) > 1
        else 0
    )

    # Print results
    print("\n=== Hybrid Accuracy Results (Mismatched Mode - By Position) ===")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print(f"Single response file: File {single_file_num}")
    print(f"Multiple response file: File {multi_file_num}")
    print(f"Response positions: {len(hybrid_accuracies_by_position)}")

    print("\nAccuracies by response position:")
    for i, (h_acc, s_acc, m_acc) in enumerate(
        zip(
            hybrid_accuracies_by_position,
            individual_accuracies_single_by_position,
            individual_accuracies_multi_by_position,
        )
    ):
        print(
            f"  Position {i}: Single={s_acc:.4f}, Multi={m_acc:.4f}, Hybrid={h_acc:.4f}"
        )

    print("\n=== Summary Statistics ===")
    print(f"Single response method - Mean: {single_mean:.4f} ({single_mean*100:.2f}%), Std: {single_std:.4f}")
    print(f"Multi response method  - Mean: {multi_mean:.4f} ({multi_mean*100:.2f}%), Std: {multi_std:.4f}")
    print(f"Hybrid                 - Mean: {hybrid_mean:.4f} ({hybrid_mean*100:.2f}%), Std: {hybrid_std:.4f}")
    print(f"Improvement: +{(hybrid_mean - max(single_mean, multi_mean))*100:.2f} percentage points")

    print(f"\nHybrid (mean, std): ({hybrid_mean * 100:.4f}, {hybrid_std * 100:.4f})")


def calculate_hybrid_accuracy(
    file1_path: str, file2_path: str, mismatched_mode: bool = False
) -> None:
    """Calculate hybrid accuracy from two JSON result files, grouped by response index."""

    if mismatched_mode:
        return calculate_hybrid_accuracy_mismatched(file1_path, file2_path)

    # Load both files
    data1 = load_json_file(file1_path)
    data2 = load_json_file(file2_path)

    # Extract detailed results
    results1 = data1.get("results", [])
    results2 = data2.get("results", [])

    # Check if both files have the same number of entries
    if len(results1) != len(results2):
        print(f"Warning: Files have different number of entries ({len(results1)} vs {len(results2)})")
        min_len = min(len(results1), len(results2))
        results1 = results1[:min_len]
        results2 = results2[:min_len]

    # Group results by user_prompt for both methods
    groups1 = group_by_user_prompt(results1)
    groups2 = group_by_user_prompt(results2)

    # Ensure both have the same prompts
    common_prompts = set(groups1.keys()) & set(groups2.keys())
    if len(common_prompts) != len(groups1) or len(common_prompts) != len(groups2):
        print(f"Warning: Different prompts between files. Using {len(common_prompts)} common prompts.")

    # Check if we should auto-enable mismatched mode
    response_counts1 = [len(groups1[prompt]) for prompt in common_prompts]
    response_counts2 = [len(groups2[prompt]) for prompt in common_prompts]

    if (min(response_counts1) == max(response_counts1) == 1 and max(response_counts2) > 1) or \
       (min(response_counts2) == max(response_counts2) == 1 and max(response_counts1) > 1):
        print("Detected mismatched response patterns. Auto-enabling mismatched mode.")
        return calculate_hybrid_accuracy_mismatched(file1_path, file2_path)

    # Determine max response index
    max_response_idx = 0
    for prompt in common_prompts:
        max_response_idx = max(
            max_response_idx, len(groups1[prompt]), len(groups2[prompt])
        )

    # Calculate accuracy by response index
    hybrid_accuracies_by_index = []
    individual_accuracies1_by_index = []
    individual_accuracies2_by_index = []

    for response_idx in range(max_response_idx):
        hybrid_successes = []
        individual_successes1 = []
        individual_successes2 = []

        for prompt in common_prompts:
            responses1 = groups1[prompt]
            responses2 = groups2[prompt]

            if response_idx < len(responses1) and response_idx < len(responses2):
                success1 = responses1[response_idx].get("success", 0)
                success2 = responses2[response_idx].get("success", 0)
                hybrid_success = 1 if (success1 == 1 or success2 == 1) else 0

                hybrid_successes.append(hybrid_success)
                individual_successes1.append(success1)
                individual_successes2.append(success2)

        # Calculate accuracies for this response index
        if hybrid_successes:
            hybrid_acc = sum(hybrid_successes) / len(hybrid_successes)
            individual_acc1 = sum(individual_successes1) / len(individual_successes1)
            individual_acc2 = sum(individual_successes2) / len(individual_successes2)

            hybrid_accuracies_by_index.append(hybrid_acc)
            individual_accuracies1_by_index.append(individual_acc1)
            individual_accuracies2_by_index.append(individual_acc2)

    # Calculate mean and std
    import numpy as np

    hybrid_mean = (
        np.mean(hybrid_accuracies_by_index) if hybrid_accuracies_by_index else 0
    )
    hybrid_std = (
        np.std(hybrid_accuracies_by_index) if len(hybrid_accuracies_by_index) > 1 else 0
    )

    individual_mean1 = np.mean(individual_accuracies1_by_index) if individual_accuracies1_by_index else 0
    individual_std1 = np.std(individual_accuracies1_by_index) if len(individual_accuracies1_by_index) > 1 else 0

    individual_mean2 = (
        np.mean(individual_accuracies2_by_index)
        if individual_accuracies2_by_index
        else 0
    )
    individual_std2 = (
        np.std(individual_accuracies2_by_index)
        if len(individual_accuracies2_by_index) > 1
        else 0
    )

    # Print results
    print("\n=== Hybrid Accuracy Results (by Response Index) ===")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print(f"Common prompts: {len(common_prompts)}")
    print(f"Response indices: {len(hybrid_accuracies_by_index)}")

    print("\nAccuracies by response index:")
    for i, (h_acc, acc1, acc2) in enumerate(
        zip(
            hybrid_accuracies_by_index,
            individual_accuracies1_by_index,
            individual_accuracies2_by_index,
        )
    ):
        print(
            f"  Index {i}: Method1={acc1:.4f}, Method2={acc2:.4f}, Hybrid={h_acc:.4f}"
        )

    print("\n=== Summary Statistics ===")
    print(f"Method 1 - Mean: {individual_mean1:.4f} ({individual_mean1*100:.2f}%), Std: {individual_std1:.4f}")
    print(f"Method 2 - Mean: {individual_mean2:.4f} ({individual_mean2*100:.2f}%), Std: {individual_std2:.4f}")
    print(f"Hybrid   - Mean: {hybrid_mean:.4f} ({hybrid_mean*100:.2f}%), Std: {hybrid_std:.4f}")
    print(f"Improvement: +{(hybrid_mean - max(individual_mean1, individual_mean2))*100:.2f} percentage points")

    print(f"\nHybrid (mean, std): ({hybrid_mean * 100:.4f}, {hybrid_std * 100:.4f})")


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(
            "Usage: python calculate_hybrid_accuracy.py <file1.json> <file2.json> [--mismatched]"
        )
        print(
            "  --mismatched: Force mismatched mode for handling different response counts"
        )
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    mismatched_mode = len(sys.argv) == 4 and sys.argv[3] == "--mismatched"

    try:
        calculate_hybrid_accuracy(file1_path, file2_path, mismatched_mode)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
