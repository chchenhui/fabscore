import json
import numpy as np


class DecompositionAnalysis:
    def __init__(self):
        self.dictionary_tasks = [
            "latin_translation",
            "codesearchnet_python",
            "python_instructions_alpaca",
            "alpaca_instructions",
            "ms_marco_qa",
            "xsum",
            "reason_math",
            "gsm8k_math",
        ]

        # Load data with correct paths
        with open("results/decomposition_results.json", "r") as f:
            self.raw_results = json.load(f)

        with open("results/ground_truth_compositions.json", "r") as f:
            self.ground_truth = json.load(f)

        with open("results/dictionary_tasks.json", "r") as f:
            dict_data = json.load(f)
            self.dictionary_tasks = dict_data["dictionary_tasks"]

        # Map merging method abbreviations to full names
        self.merging_method_map = {
            "task": "task_arithmetic",
            "dare": "dare_linear",
            "linear": "linear",
            "ties": "ties",
        }

        print(f"Loaded {len(self.raw_results)} raw results")
        print(f"Loaded {len(self.ground_truth)} ground truth compositions")

    def filter_best_hyperparameters(self):
        """Filter results to keep only best hyperparameters for each model-algorithm pair"""
        best_hyperparams = {}

        # Find best hyperparameters for each (model_name, algorithm) pair
        for result in self.raw_results:
            key = (result["model_name"], result["algorithm"])
            reconstruction_error = result["metrics"]["reconstruction_error"]

            if (
                key not in best_hyperparams
                or reconstruction_error < best_hyperparams[key]["error"]
            ):
                best_hyperparams[key] = {
                    "hyperparams": result["hyperparams"],
                    "error": reconstruction_error,
                }

        # Filter results to keep only best hyperparameters and map merging methods
        filtered_results = []
        for result in self.raw_results:
            key = (result["model_name"], result["algorithm"])
            if (
                key in best_hyperparams
                and result["hyperparams"] == best_hyperparams[key]["hyperparams"]
            ):
                # Map merging method name
                result["merging_method_full"] = self.merging_method_map.get(
                    result["merging_method"], result["merging_method"]
                )
                filtered_results.append(result)

        self.filtered_results = filtered_results
        print(
            f"Filtered to {len(self.filtered_results)} results with best hyperparameters"
        )

        return self.filtered_results

    def compute_summary_statistics(self, data, metric):
        """Compute summary statistics for a metric"""
        values = []
        for item in data:
            if (
                isinstance(item, dict)
                and "metrics" in item
                and metric in item["metrics"]
            ):
                val = item["metrics"][metric]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    values.append(val)
            elif isinstance(item, dict) and metric in item:
                val = item[metric]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    values.append(val)

        if not values:
            return {"count": 0}

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "count": len(values),
        }

    def generate_pivot_tables(self):
        """Generate comprehensive pivot tables"""

        # Metrics to analyze
        all_metrics = [
            "reconstruction_error",
            "sparsity",
            "component_precision",
            "component_recall",
            "perfect_match",
            "execution_time",
        ]

        self.pivot_tables = {}

        # 1. By Algorithm
        by_algorithm = {}
        algorithms = list(set(result["algorithm"] for result in self.filtered_results))

        for algorithm in algorithms:
            algo_data = [
                r for r in self.filtered_results if r["algorithm"] == algorithm
            ]
            by_algorithm[algorithm] = {}
            for metric in all_metrics:
                by_algorithm[algorithm][metric] = self.compute_summary_statistics(
                    algo_data, metric
                )

        self.pivot_tables["by_algorithm"] = by_algorithm

        # 2. By Category
        by_category = {}
        categories = list(set(result["category"] for result in self.filtered_results))

        for category in categories:
            cat_data = [r for r in self.filtered_results if r["category"] == category]
            by_category[category] = {}
            for metric in all_metrics:
                by_category[category][metric] = self.compute_summary_statistics(
                    cat_data, metric
                )

        self.pivot_tables["by_category"] = by_category

        # 3. By Merging Method (use full names)
        by_merging = {}
        merging_methods = list(
            set(result["merging_method_full"] for result in self.filtered_results)
        )

        for method in merging_methods:
            method_data = [
                r for r in self.filtered_results if r["merging_method_full"] == method
            ]
            by_merging[method] = {}
            for metric in all_metrics:
                by_merging[method][metric] = self.compute_summary_statistics(
                    method_data, metric
                )

        self.pivot_tables["by_merging_method"] = by_merging

        # 4. Algorithm x Category
        by_algo_category = {}
        for algorithm in algorithms:
            by_algo_category[algorithm] = {}
            for category in categories:
                subset = [
                    r
                    for r in self.filtered_results
                    if r["algorithm"] == algorithm and r["category"] == category
                ]
                if len(subset) > 0:
                    by_algo_category[algorithm][category] = {}
                    for metric in all_metrics:
                        by_algo_category[algorithm][category][metric] = (
                            self.compute_summary_statistics(subset, metric)
                        )

        self.pivot_tables["by_algorithm_by_category"] = by_algo_category

        # 5. Algorithm x Merging Method
        by_algo_merging = {}
        for algorithm in algorithms:
            by_algo_merging[algorithm] = {}
            for method in merging_methods:
                subset = [
                    r
                    for r in self.filtered_results
                    if r["algorithm"] == algorithm
                    and r["merging_method_full"] == method
                ]
                if len(subset) > 0:
                    by_algo_merging[algorithm][method] = {}
                    for metric in all_metrics:
                        by_algo_merging[algorithm][method][metric] = (
                            self.compute_summary_statistics(subset, metric)
                        )

        self.pivot_tables["by_algorithm_by_merging_method"] = by_algo_merging

        # 6. Known x Algorithm x Merging Method
        by_algo_merging = {}
        for algorithm in algorithms:
            by_algo_merging[algorithm] = {}
            for method in merging_methods:
                subset = [
                    r
                    for r in self.filtered_results
                    if r["algorithm"] == algorithm
                    and r["merging_method_full"] == method
                    and r["category"] == "known"
                ]
                if len(subset) > 0:
                    by_algo_merging[algorithm][method] = {}
                    for metric in all_metrics:
                        by_algo_merging[algorithm][method][metric] = (
                            self.compute_summary_statistics(subset, metric)
                        )

        self.pivot_tables["known_by_algorithm_by_merging_method"] = by_algo_merging

        # 7. Mixed x Algorithm x Merging Method
        by_algo_merging = {}
        for algorithm in algorithms:
            by_algo_merging[algorithm] = {}
            for method in merging_methods:
                subset = [
                    r
                    for r in self.filtered_results
                    if r["algorithm"] == algorithm
                    and r["merging_method_full"] == method
                    and r["category"] == "mixed"
                ]
                if len(subset) > 0:
                    by_algo_merging[algorithm][method] = {}
                    for metric in all_metrics:
                        by_algo_merging[algorithm][method][metric] = (
                            self.compute_summary_statistics(subset, metric)
                        )

        self.pivot_tables["mixed_by_algorithm_by_merging_method"] = by_algo_merging
        return self.pivot_tables


# Run the complete analysis
print("Starting Phase 4 Analysis...")
analyzer = DecompositionAnalysis()
filtered_results = analyzer.filter_best_hyperparameters()
pivot_tables = analyzer.generate_pivot_tables()

print("\n=== Analysis Complete ===")
print("Pivot table categories:", list(pivot_tables.keys()))

# Save the complete analysis
analysis_output = {
    "pivot_tables": pivot_tables,
    "summary": {
        "total_filtered_results": len(filtered_results),
        "algorithms": sorted(list(set(r["algorithm"] for r in filtered_results))),
        "categories": sorted(list(set(r["category"] for r in filtered_results))),
        "merging_methods": sorted(
            list(set(r["merging_method_full"] for r in filtered_results))
        ),
    },
}

with open("results/pivot_decomposition_analysis.json", "w") as f:
    json.dump(analysis_output, f, indent=2)

print("\nComplete analysis saved to results/pivot_decomposition_analysis.json")
