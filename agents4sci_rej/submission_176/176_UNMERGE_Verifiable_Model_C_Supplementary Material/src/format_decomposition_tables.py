import json

# Load the analysis results
with open("results/pivot_decomposition_analysis.json", "r") as f:
    analysis = json.load(f)


def format_stats(stats_dict, metric_name):
    """Format statistics dictionary into readable string"""
    if stats_dict.get("count", 0) == 0:
        return "N/A"

    mean = stats_dict.get("mean", 0)
    std = stats_dict.get("std", 0)
    count = stats_dict.get("count", 0)

    if metric_name in [
        "reconstruction_error",
        "component_precision",
        "component_recall",
        "perfect_match",
    ]:
        return f"{mean:.4f} ± {std:.4f} (n={count})"
    elif metric_name == "sparsity":
        return f"{mean:.2f} ± {std:.2f} (n={count})"
    elif metric_name == "execution_time":
        return f"{mean:.4f}s ± {std:.4f}s (n={count})"
    else:
        return f"{mean:.4f} ± {std:.4f} (n={count})"


def create_pivot_table_markdown(data, title, row_key, col_key=None):
    """Create markdown table from pivot data"""
    markdown = f"\n### {title}\n\n"

    if col_key is None:
        # Simple table: row_key vs metrics
        metrics = [
            "reconstruction_error",
            "sparsity",
            "component_precision",
            "component_recall",
            "perfect_match",
        ]

        # Header
        markdown += f"| {row_key.replace('_', ' ').title()} | Reconstruction Error | Sparsity | Component Precision | Component Recall | Perfect Match |\n"
        markdown += "|" + "---|" * 6 + "\n"

        # Rows
        for key, values in data.items():
            row = f"| {key} |"
            for metric in metrics:
                if metric in values:
                    row += f" {format_stats(values[metric], metric)} |"
                else:
                    row += " N/A |"
            markdown += row + "\n"

    else:
        # Complex table: row_key x col_key
        # Get all unique column keys
        all_cols = set()
        for row_data in data.values():
            all_cols.update(row_data.keys())
        all_cols = sorted(all_cols)

        # Create table for reconstruction error
        markdown += "\n#### Reconstruction Error\n\n"
        markdown += (
            f"| {row_key.replace('_', ' ').title()} |" + "|".join(all_cols) + "|\n"
        )
        markdown += "|" + "---|" * (len(all_cols) + 1) + "\n"

        for row_key_val, row_data in data.items():
            row = f"| {row_key_val} |"
            for col in all_cols:
                if col in row_data and "reconstruction_error" in row_data[col]:
                    stats = row_data[col]["reconstruction_error"]
                    row += f" {stats.get('mean', 0):.4f} |"
                else:
                    row += " N/A |"
            markdown += row + "\n"

        # Create table for sparsity
        markdown += "\n#### Sparsity\n\n"
        markdown += (
            f"| {row_key.replace('_', ' ').title()} |" + "|".join(all_cols) + "|\n"
        )
        markdown += "|" + "---|" * (len(all_cols) + 1) + "\n"

        for row_key_val, row_data in data.items():
            row = f"| {row_key_val} |"
            for col in all_cols:
                if col in row_data and "sparsity" in row_data[col]:
                    stats = row_data[col]["sparsity"]
                    row += f" {stats.get('mean', 0):.2f} |"
                else:
                    row += " N/A |"
            markdown += row + "\n"

    return markdown


# Generate all pivot tables
pivot_tables = analysis["pivot_tables"]

print("=== PIVOT TABLES FOR PHASE 4 REPORT ===\n")

# 1. By Algorithm
algo_table = create_pivot_table_markdown(
    pivot_tables["by_algorithm"], "Results by Decomposition Algorithm", "algorithm"
)
print(algo_table)

# 2. By Category
category_table = create_pivot_table_markdown(
    pivot_tables["by_category"], "Results by Model Category", "category"
)
print(category_table)

# 3. By Merging Method
merging_table = create_pivot_table_markdown(
    pivot_tables["by_merging_method"], "Results by Merging Method", "merging_method"
)
print(merging_table)

# 4. Algorithm x Category (Reconstruction Error focus)
print("\n### Algorithm x Category - Reconstruction Error")
print("| Algorithm | Known | Mixed | Unknown |")
print("|---|---|---|---|")

for algo, categories in pivot_tables["by_algorithm_by_category"].items():
    row = f"| {algo} |"
    for cat in ["known", "mixed", "unknown"]:
        if cat in categories and "reconstruction_error" in categories[cat]:
            mean_re = categories[cat]["reconstruction_error"].get("mean", 0)
            row += f" {mean_re:.4f} |"
        else:
            row += " N/A |"
    print(row)

# 4. Algorithm x Category (Reconstruction Error focus)
print("\n### Algorithm x Category - Reconstruction Error")
print("| Algorithm | Known | Mixed | Unknown |")
print("|---|---|---|---|")

for algo, categories in pivot_tables["by_algorithm_by_category"].items():
    row = f"| {algo} |"
    for cat in ["known", "mixed", "unknown"]:
        if cat in categories and "reconstruction_error" in categories[cat]:
            mean_re = categories[cat]["reconstruction_error"].get("mean", 0)
            row += f" {mean_re:.4f} |"
        else:
            row += " N/A |"
    print(row)

# 5. Algorithm x Merging Method (Reconstruction Error focus)
print("\n### Algorithm x Merging method- Reconstruction Error")
print("| Algorithm | DARE | TIES | Linear | Task Arithmetic |")
print("|---|---|---|---|---|")

for algo, methods in pivot_tables["by_algorithm_by_merging_method"].items():
    row = f"| {algo} |"
    for method in ["dare_linear", "ties", "linear", "task_arithmetic"]:
        if method in methods and "reconstruction_error" in methods[method]:
            mean_re = methods[method]["reconstruction_error"].get("mean", 0)
            row += f" {mean_re:.4f} |"
        else:
            row += " N/A |"
    print(row)


# 6. Known x Algorithm x Merging Method
print("\n### Algorithm x Merging method (known category only) - Reconstruction Error")
print("| Algorithm | DARE | TIES | Linear | Task Arithmetic |")
print("|---|---|---|---|---|")

for algo, methods in pivot_tables["known_by_algorithm_by_merging_method"].items():
    row = f"| {algo} |"
    for method in ["dare_linear", "ties", "linear", "task_arithmetic"]:
        if method in methods and "reconstruction_error" in methods[method]:
            mean_re = methods[method]["reconstruction_error"].get("mean", 0)
            row += f" {mean_re:.4f} |"
        else:
            row += " N/A |"
    print(row)

print("\n### Algorithm x Merging method (known category only) - Precision / Recall")
print("| Algorithm | DARE | TIES | Linear | Task Arithmetic |")
print("|---|---|---|---|---|")

for algo, methods in pivot_tables["known_by_algorithm_by_merging_method"].items():
    row = f"| {algo} |"
    for method in ["dare_linear", "ties", "linear", "task_arithmetic"]:
        if method in methods and "reconstruction_error" in methods[method]:
            mean_precision = methods[method]["component_precision"].get("mean", 0)
            mean_recall = methods[method]["component_recall"].get("mean", 0)
            row += f" {mean_precision:.2f} / {mean_recall:.2f} |"
        else:
            row += " N/A |"
    print(row)


print("\n### Algorithm x Merging method (known category only) - Sparsity")
print("| Algorithm | DARE | TIES | Linear | Task Arithmetic |")
print("|---|---|---|---|---|")

for algo, methods in pivot_tables["known_by_algorithm_by_merging_method"].items():
    row = f"| {algo} |"
    for method in ["dare_linear", "ties", "linear", "task_arithmetic"]:
        if method in methods and "reconstruction_error" in methods[method]:
            mean_sparsity = methods[method]["sparsity"].get("mean", 0)
            row += f" {mean_sparsity:.2f} |"
        else:
            row += " N/A |"
    print(row)

print("\n### Algorithm x Merging method (mixed category only) - Precision / Recall")
print("| Algorithm | DARE | TIES | Linear | Task Arithmetic |")
print("|---|---|---|---|---|")

for algo, methods in pivot_tables["mixed_by_algorithm_by_merging_method"].items():
    row = f"| {algo} |"
    for method in ["dare_linear", "ties", "linear", "task_arithmetic"]:
        if method in methods and "reconstruction_error" in methods[method]:
            mean_precision = methods[method]["component_precision"].get("mean", 0)
            mean_recall = methods[method]["component_recall"].get("mean", 0)
            row += f" {mean_precision:.2f} / {mean_recall:.2f} |"
        else:
            row += " N/A |"
    print(row)


print("\n=== ANALYSIS SUMMARY ===")
summary = analysis["summary"]
print(f"- Total filtered results: {summary['total_filtered_results']}")
print(f"- Algorithms: {', '.join(summary['algorithms'])}")
print(f"- Categories: {', '.join(summary['categories'])}")
print(f"- Merging methods: {', '.join(summary['merging_methods'])}")
