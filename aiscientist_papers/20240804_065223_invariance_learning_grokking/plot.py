import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp

# LOAD FINAL RESULTS:
datasets = ["x_div_y", "x_minus_y", "x_plus_y", "permutation"]
folders = os.listdir("./")
final_results = {}
results_info = {}
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        results_dict = np.load(
            osp.join(folder, "all_results.npy"), allow_pickle=True
        ).item()
        print(results_dict.keys())
        run_info = {}
        for dataset in datasets:
            run_info[dataset] = {}
            val_losses = []
            train_losses = []
            val_accs = []
            train_accs = []
            invariance_scores = []
            steps = None
            for k in results_dict.keys():
                if dataset in k and "val_info" in k:
                    steps = [info["step"] for info in results_dict[k]]
                    val_losses.append([info["val_loss"] for info in results_dict[k]])
                    val_accs.append([info["val_accuracy"] for info in results_dict[k]])
                if dataset in k and "train_info" in k:
                    train_losses.append(
                        [info["train_loss"] for info in results_dict[k]]
                    )
                    train_accs.append(
                        [info["train_accuracy"] for info in results_dict[k]]
                    )
                if dataset in k and "invariance_scores" in k:
                    invariance_scores.append([info["score"] for info in results_dict[k]])
            
            if steps is not None:
                run_info[dataset]["step"] = steps
                mean_val_losses = np.mean(val_losses, axis=0) if val_losses else []
                mean_train_losses = np.mean(train_losses, axis=0) if train_losses else []
                mean_val_accs = np.mean(val_accs, axis=0) if val_accs else []
                mean_train_accs = np.mean(train_accs, axis=0) if train_accs else []
                mean_invariance_scores = np.mean(invariance_scores, axis=0) if invariance_scores else []
                
                run_info[dataset]["val_loss"] = mean_val_losses
                run_info[dataset]["train_loss"] = mean_train_losses
                run_info[dataset]["val_acc"] = mean_val_accs
                run_info[dataset]["train_acc"] = mean_train_accs
                run_info[dataset]["invariance_score"] = mean_invariance_scores
                
                if len(val_losses) > 0:
                    run_info[dataset]["val_loss_sterr"] = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))
                    run_info[dataset]["train_loss_sterr"] = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
                    run_info[dataset]["val_acc_sterr"] = np.std(val_accs, axis=0) / np.sqrt(len(val_accs))
                    run_info[dataset]["train_acc_sterr"] = np.std(train_accs, axis=0) / np.sqrt(len(train_accs))
                    run_info[dataset]["invariance_score_sterr"] = np.std(invariance_scores, axis=0) / np.sqrt(len(invariance_scores))
                else:
                    run_info[dataset]["val_loss_sterr"] = np.zeros_like(mean_val_losses)
                    run_info[dataset]["train_loss_sterr"] = np.zeros_like(mean_train_losses)
                    run_info[dataset]["val_acc_sterr"] = np.zeros_like(mean_val_accs)
                    run_info[dataset]["train_acc_sterr"] = np.zeros_like(mean_train_accs)
                    run_info[dataset]["invariance_score_sterr"] = np.zeros_like(mean_invariance_scores)
        
        results_info[folder] = run_info

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baseline",
    "run_1": "Invariance Score Implementation",
    "run_2": "Visualization of Metrics",
    "run_4": "Detailed Analysis of Permutation Task",
    "run_5": "Increased Model Capacity",
}


# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap("tab20")
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs))

# Combined plot: training loss, validation accuracy, and invariance score
for dataset in datasets:
    plt.figure(figsize=(12, 8))
    for i, run in enumerate(runs):
        iters = results_info[run][dataset]["step"]
        
        # Training loss
        mean_loss = results_info[run][dataset]["train_loss"]
        sterr_loss = results_info[run][dataset]["train_loss_sterr"]
        plt.plot(iters, mean_loss, label=f'{labels[run]} - Train Loss', color=colors[i], linestyle='-')
        plt.fill_between(iters, mean_loss - sterr_loss, mean_loss + sterr_loss, color=colors[i], alpha=0.1)
        
        # Validation accuracy
        mean_acc = results_info[run][dataset]["val_acc"]
        sterr_acc = results_info[run][dataset]["val_acc_sterr"]
        plt.plot(iters, mean_acc, label=f'{labels[run]} - Val Acc', color=colors[i], linestyle='--')
        plt.fill_between(iters, mean_acc - sterr_acc, mean_acc + sterr_acc, color=colors[i], alpha=0.1)
        
        # Invariance score (only for run_1)
        if run == "run_1" and "invariance_score" in results_info[run][dataset]:
            mean_inv = results_info[run][dataset]["invariance_score"]
            sterr_inv = results_info[run][dataset]["invariance_score_sterr"]
            if len(mean_inv) == len(iters):
                plt.plot(iters, mean_inv, label=f'{labels[run]} - Inv Score', color=colors[i], linestyle=':')
                plt.fill_between(iters, mean_inv - sterr_inv, mean_inv + sterr_inv, color=colors[i], alpha=0.1)
            else:
                print(f"Warning: Invariance score data for {dataset} in {run} has incorrect dimensions. Skipping this plot.")

    plt.title(f"Training Loss, Validation Accuracy, and Invariance Score for {dataset} Dataset")
    plt.xlabel("Update Steps")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"combined_metrics_{dataset}.png")
    plt.close()

def plot_invariance_vs_accuracy(results_info, datasets, runs, colors, out_dir):
    for dataset in datasets:
        plt.figure(figsize=(12, 8))
        for i, run in enumerate(runs):
            if run in results_info and dataset in results_info[run]:
                steps = results_info[run][dataset]["step"]
                val_acc = results_info[run][dataset]["val_acc"]
                inv_score = results_info[run][dataset].get("invariance_score", [])
                
                plt.plot(steps, val_acc, label=f'{labels[run]} - Val Acc', color=colors[i], linestyle='-')
                
                if len(inv_score) == len(steps):
                    plt.plot(steps, inv_score, label=f'{labels[run]} - Inv Score', color=colors[i], linestyle='--')
                else:
                    print(f"Warning: Invariance score data for {dataset} in {run} has incorrect length. Skipping this plot.")
        
        plt.title(f"Validation Accuracy vs Invariance Score for {dataset} Dataset")
        plt.xlabel("Update Steps")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"inv_vs_acc_{dataset}.png"))
        plt.close()

def plot_grokking_points(results_info, datasets, runs, colors, out_dir):
    grokking_points = {dataset: [] for dataset in datasets}
    invariance_points = {dataset: [] for dataset in datasets}
    
    for run in runs:
        if run in results_info:
            for dataset in datasets:
                if dataset in results_info[run]:
                    grokking_points[dataset].append(results_info[run][dataset].get("grokking_point", 7500))
                    invariance_points[dataset].append(results_info[run][dataset].get("invariance_point", 7500))
    
    # Filter out any None values
    grokking_points = {k: [v for v in vals if v is not None] for k, vals in grokking_points.items()}
    invariance_points = {k: [v for v in vals if v is not None] for k, vals in invariance_points.items()}
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, [np.mean(grokking_points[d]) for d in datasets], width, label='Grokking Point', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width/2, [np.mean(invariance_points[d]) for d in datasets], width, label='Invariance Point', color='red', alpha=0.7)
    
    ax.set_ylabel('Steps')
    ax.set_title('Grokking Point vs Invariance Point')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "grokking_vs_invariance_points.png"))
    plt.close()

# Call the new plotting functions
plot_invariance_vs_accuracy(results_info, datasets, runs, colors, ".")
plot_grokking_points(results_info, datasets, runs, colors, ".")

# Original plots (keeping these for reference)
# ... [rest of the original plotting code]
