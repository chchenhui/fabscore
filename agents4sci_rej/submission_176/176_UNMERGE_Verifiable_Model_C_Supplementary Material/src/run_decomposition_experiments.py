import itertools
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np
from src.decomposition_algorithms import DecompositionAlgorithms


class DecompositionExperiments:
    def __init__(self, dictionary_tasks: List[str], results_dir: str = "results"):
        self.dictionary_tasks = dictionary_tasks
        self.results_dir = Path(results_dir)
        self.dictionary_matrix = None
        self.ground_truth_compositions = None
        self.target_models = []

    def load_dictionary(self, models_dir: str = "models"):
        dictionary_vectors = []

        for task in self.dictionary_tasks:
            vector_path = Path(models_dir) / task / "compressed_task_vector.pt"
            vector = torch.load(vector_path, map_location="cpu")
            dictionary_vectors.append(vector)

        self.dictionary_matrix = torch.stack(dictionary_vectors)
        print(f"Loaded dictionary matrix: {self.dictionary_matrix.shape}")

    def load_ground_truth_compositions(self):
        gt_path = self.results_dir / "ground_truth_compositions.json"
        with open(gt_path, "r") as f:
            self.ground_truth_compositions = json.load(f)
        print(f"Loaded ground truth for {len(self.ground_truth_compositions)} models")

    def load_target_models(self, models_dir: str = "models/merged"):
        models_path = Path(models_dir)

        for model_dir in models_path.iterdir():
            if model_dir.is_dir():
                vector_path = model_dir / "compressed_vector.pt"
                if vector_path.exists():
                    vector = torch.load(vector_path, map_location="cpu")
                    self.target_models.append(
                        {"name": model_dir.name, "vector": vector}
                    )

        print(f"Loaded {len(self.target_models)} target models")

    def categorize_models(self):
        known_models = []
        mixed_models = []
        unknown_models = []

        for model in self.target_models:
            model_name = model["name"]
            if model_name in self.ground_truth_compositions:
                category = self.ground_truth_compositions[model_name]["category"]
                if category == "known":
                    known_models.append(model)
                elif category == "mixed":
                    mixed_models.append(model)
                elif category == "unknown":
                    unknown_models.append(model)

        print(
            f"Model categories: Known={len(known_models)}, Mixed={len(mixed_models)}, Unknown={len(unknown_models)}"
        )
        return known_models, mixed_models, unknown_models

    def get_hyperparameter_grids(self):
        return {
            "lasso": {"alpha": [1e-8, 1e-7, 1e-6], "max_iter": [5000]},
            "ridge": {"alpha": [1e-8, 1e-6, 1e-4, 1e-2]},
            "elastic_net": {
                "alpha": [1e-8, 1e-6, 1e-4],
                "l1_ratio": [0.1, 0.5, 0.9],
                "max_iter": [5000],
            },
            "omp": {
                "n_nonzero_coefs": [1, 2, 3, 4, 5, 6, 7, 8],
                "tol": [1e-8, 1e-6, 1e-4],
            },
        }

    def compute_metrics(
        self,
        coefficients: torch.Tensor,
        reconstruction: torch.Tensor,
        target_vector: torch.Tensor,
        ground_truth_weights: List[float] = None,
    ):
        metrics = {}

        cosine_sim = torch.cosine_similarity(
            target_vector.unsqueeze(0), reconstruction.unsqueeze(0)
        ).item()
        metrics["reconstruction_error"] = 1 - max(0, cosine_sim) ** 2
        metrics["sparsity"] = torch.sum(torch.abs(coefficients) > 1e-6).item()

        if ground_truth_weights is not None:
            gt_weights = torch.tensor(ground_truth_weights)
            gt_nonzero = torch.abs(gt_weights) > 1e-6
            pred_nonzero = torch.abs(coefficients) > 1e-6

            true_positives = torch.sum(gt_nonzero & pred_nonzero).item()
            false_positives = torch.sum((~gt_nonzero) & pred_nonzero).item()
            false_negatives = torch.sum(gt_nonzero & (~pred_nonzero)).item()

            metrics["component_precision"] = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0.0
            )
            metrics["component_recall"] = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0.0
            )

            perfect_match = torch.allclose(gt_nonzero, pred_nonzero)
            metrics["perfect_match"] = 1.0 if perfect_match else 0.0

        return metrics

    def run_single_experiment(
        self,
        algorithm: str,
        model: Dict,
        hyperparams: Dict,
        decomposer: DecompositionAlgorithms,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        target_vector = model["vector"]
        model_name = model["name"]

        start_time = time.time()

        try:
            if algorithm == "dot_product":
                coefficients, reconstruction = decomposer.dot_product_similarity(
                    target_vector
                )
            elif algorithm == "lasso":
                coefficients, reconstruction = decomposer.lasso_regression(
                    target_vector, **hyperparams
                )
            elif algorithm == "ridge":
                coefficients, reconstruction = decomposer.ridge_regression(
                    target_vector, **hyperparams
                )
            elif algorithm == "elastic_net":
                coefficients, reconstruction = decomposer.elastic_net_regression(
                    target_vector, **hyperparams
                )
            elif algorithm == "omp":
                coefficients, reconstruction = decomposer.orthogonal_matching_pursuit(
                    target_vector, **hyperparams
                )
            elif algorithm == "nnls":
                coefficients, reconstruction = decomposer.non_negative_least_squares(
                    target_vector
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        except Exception as e:
            print(f"Error in {algorithm} for {model_name}: {e}")
            coefficients = torch.zeros(len(self.dictionary_tasks))
            reconstruction = torch.zeros_like(target_vector)

        execution_time = time.time() - start_time

        ground_truth_weights = None
        if model_name in self.ground_truth_compositions:
            ground_truth_weights = self.ground_truth_compositions[model_name]["weights"]

        metrics = self.compute_metrics(
            coefficients, reconstruction, target_vector, ground_truth_weights
        )
        metrics["execution_time"] = execution_time

        return {
            "model_name": model_name,
            "algorithm": algorithm,
            "hyperparams": hyperparams,
            "seed": seed,
            "coefficients": coefficients.tolist(),
            "metrics": metrics,
        }

    def run_experiments(
        self,
        algorithms: List[str] = None,
        seeds: List[int] = None,
        max_models_per_category: Optional[int] = None,
    ):
        if algorithms is None:
            algorithms = [
                "omp",
                "nnls",
                "dot_product",
                "lasso",
                "ridge",
                "elastic_net",
            ]

        if seeds is None:
            seeds = [42, 123, 456]

        self.load_dictionary()
        self.load_ground_truth_compositions()
        self.load_target_models()

        decomposer = DecompositionAlgorithms(self.dictionary_matrix)
        known_models, mixed_models, unknown_models = self.categorize_models()

        if max_models_per_category:
            known_models = known_models[:max_models_per_category]
            mixed_models = mixed_models[:max_models_per_category]
            unknown_models = unknown_models[:max_models_per_category]

        hyperparameter_grids = self.get_hyperparameter_grids()

        all_results = []
        total_experiments = 0

        for algorithm in algorithms:
            if algorithm in hyperparameter_grids:
                grid = hyperparameter_grids[algorithm]
                param_combinations = self._generate_param_combinations(grid)
            else:
                param_combinations = [{}]

            for models, category in [
                (known_models, "known"),
                (mixed_models, "mixed"),
                (unknown_models, "unknown"),
            ]:
                for model in models:
                    for params in param_combinations:
                        for seed in seeds:
                            total_experiments += 1

        print(f"Running {total_experiments} total experiments...")

        experiment_count = 0
        for algorithm in algorithms:
            print(f"Running algorithm: {algorithm}")

            if algorithm in hyperparameter_grids:
                grid = hyperparameter_grids[algorithm]
                param_combinations = self._generate_param_combinations(grid)
            else:
                param_combinations = [{}]

            for models, category in [
                (known_models, "known"),
                (mixed_models, "mixed"),
                (unknown_models, "unknown"),
            ]:
                print(f"  Category: {category} ({len(models)} models)")

                for model in models:
                    merging_method = model["name"].split("_")[1]
                    for params in param_combinations:
                        for seed in seeds:
                            experiment_count += 1
                            if experiment_count % 100 == 0:
                                print(
                                    f"    Progress: {experiment_count}/{total_experiments} ({100*experiment_count/total_experiments:.1f}%)"
                                )

                            result = self.run_single_experiment(
                                algorithm, model, params, decomposer, seed
                            )
                            result["category"] = category
                            result["merging_method"] = merging_method
                            all_results.append(result)

        return all_results

    def _generate_param_combinations(self, param_grid: Dict):
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)

        return combinations

    def save_results(
        self, results: List[Dict], filename: str = "decomposition_results.json"
    ):
        output_path = self.results_dir / filename

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved {len(results)} results to {output_path}")

    def analyze_results(self, results: List[Dict]):
        analysis = {
            "by_algorithm": defaultdict(list),
            "by_category": defaultdict(list),
            "by_merging_method": defaultdict(list),
            "by_algorithm_by_category": defaultdict(list),
            "by_algorithm_by_category_by_merging_method": defaultdict(list),
        }

        best_hyperparams = {}
        for result in results:
            key = (result["model_name"], result["algorithm"])
            reconstruction_error = result["metrics"]["reconstruction_error"]
            if key not in best_hyperparams or reconstruction_error < best_hyperparams[key]["error"]:
                best_hyperparams[key] = {
                    "hyperparams": result["hyperparams"],
                    "error": reconstruction_error
                }
        filtered_results = []
        for result in results:
            key = (result["model_name"], result["algorithm"])
            if key in best_hyperparams and result["hyperparams"] == best_hyperparams[key]["hyperparams"]:
                filtered_results.append(result)

        for result in filtered_results:
            algorithm = result["algorithm"]
            category = result["category"]
            metrics = result["metrics"]
            merging_method = result["merging_method"]

            analysis["by_algorithm"][algorithm].append(metrics)
            analysis["by_category"][category].append(metrics)
            analysis["by_merging_method"][merging_method].append(metrics)
            analysis["by_algorithm_by_category"][f"{algorithm}_{category}"].append(
                metrics
            )
            analysis["by_algorithm_by_category_by_merging_method"][
                f"{algorithm}_{category}_{merging_method}"
            ].append(metrics)

        for algorithm, metric_list in analysis["by_algorithm"].items():
            analysis["by_algorithm"][algorithm] = self._compute_summary_stats(
                metric_list
            )
        for category, metric_list in analysis["by_category"].items():
            analysis["by_category"][category] = self._compute_summary_stats(metric_list)
        for category, metric_list in analysis["by_merging_method"].items():
            analysis["by_merging_method"][category] = self._compute_summary_stats(
                metric_list
            )
        for item, metric_list in analysis["by_algorithm_by_category"].items():
            analysis["by_algorithm_by_category"][item] = self._compute_summary_stats(
                metric_list
            )
        for item, metric_list in analysis[
            "by_algorithm_by_category_by_merging_method"
        ].items():
            analysis["by_algorithm_by_category_by_merging_method"][item] = (
                self._compute_summary_stats(metric_list)
            )
        return analysis

    def _compute_summary_stats(self, metric_list: List[Dict]):
        if not metric_list:
            return {}

        summary = {}

        for key in metric_list[0].keys():
            values = [m[key] for m in metric_list if key in m]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }

        return summary


def main():
    print("=" * 60)
    print("UNMERGE Phase 4: Full-Scale Decomposition Experiments")
    print("=" * 60)

    with open("results/dictionary_tasks.json", "r") as f:
        dictionary_data = json.load(f)
    dictionary_tasks = dictionary_data["dictionary_tasks"]

    print(f"Dictionary tasks: {dictionary_tasks}")
    print(f"Number of dictionary tasks: {len(dictionary_tasks)}")

    experiment_runner = DecompositionExperiments(dictionary_tasks)
    results = experiment_runner.run_experiments()

    print(f"\nCompleted {len(results)} experiments")

    experiment_runner.save_results(results, "decomposition_results.json")

    print("\nAnalyzing results...")
    analysis = experiment_runner.analyze_results(results)

    analysis_path = "results/decomposition_results_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {analysis_path}")

    print("\n" + "=" * 60)
    print("PHASE 4 FULL EXPERIMENTS COMPLETED")
    print("=" * 60)

    print("\nSummary by Algorithm:")
    for algorithm, stats in analysis["by_algorithm"].items():
        if "reconstruction_error" in stats:
            mean_error = stats["reconstruction_error"]["mean"]
            std_error = stats["reconstruction_error"]["std"]
            mean_sparsity = stats["sparsity"]["mean"]
            mean_time = stats["execution_time"]["mean"]
            print(
                f"  {algorithm:15s}: Error={mean_error:.4f}±{std_error:.4f}, Sparsity={mean_sparsity:.1f}, Time={mean_time:.3f}s"
            )

    print("\nSummary by Category:")
    for category, stats in analysis["by_category"].items():
        if "reconstruction_error" in stats:
            mean_error = stats["reconstruction_error"]["mean"]
            std_error = stats["reconstruction_error"]["std"]
            mean_sparsity = stats["sparsity"]["mean"]
            print(
                f"  {category:15s}: Error={mean_error:.4f}±{std_error:.4f}, Sparsity={mean_sparsity:.1f}"
            )

    # Calculate best algorithms
    print("\nAlgorithm Rankings by Reconstruction Error:")
    algo_errors = [
        (algo, stats["reconstruction_error"]["mean"])
        for algo, stats in analysis["by_algorithm"].items()
        if "reconstruction_error" in stats
    ]
    algo_errors.sort(key=lambda x: x[1])

    for i, (algo, error) in enumerate(algo_errors, 1):
        print(f"  {i}. {algo}: {error:.4f}")


if __name__ == "__main__":
    main()
