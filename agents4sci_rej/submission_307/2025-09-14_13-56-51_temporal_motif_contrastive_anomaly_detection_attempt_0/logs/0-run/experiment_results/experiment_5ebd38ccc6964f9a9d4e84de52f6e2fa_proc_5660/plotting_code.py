import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Extract data for plotting
    node_counts = experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "node_count_settings"
    ]
    losses = experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "losses"
    ]
    metrics = experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "metrics"
    ]
    tmc = experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "temporal_motif_coverage"
    ]

    # Plot losses over epochs for each node count
    for idx, num_nodes in enumerate(node_counts):
        if idx >= 5:  # Limit to at most 5 plots
            break
        plt.figure()
        plt.plot(losses[idx]["train"], label="Training Loss")
        plt.plot(losses[idx]["val"], label="Validation Loss")
        plt.title(f"Loss Over Epochs (Node Count: {num_nodes})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_plot_nodes_{num_nodes}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plots: {e}")

try:
    # Plot validation metrics over epochs for each node count
    for idx, num_nodes in enumerate(node_counts):
        if idx >= 5:  # Limit to at most 5 plots
            break
        plt.figure()
        plt.plot(metrics[idx]["val"], label="Validation F1 Score")
        plt.title(f"Validation F1 Score Over Epochs (Node Count: {num_nodes})")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"metrics_plot_nodes_{num_nodes}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating metrics plots: {e}")

try:
    # Plot Temporal Motif Coverage for each node count setting
    for idx, num_nodes in enumerate(node_counts):
        plt.figure()
        plt.plot(tmc[idx], label="TMC")
        plt.title(f"Temporal Motif Coverage (Node Count: {num_nodes})")
        plt.xlabel("Epochs")
        plt.ylabel("TMC")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"tmc_plot_nodes_{num_nodes}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating TMC plots: {e}")
