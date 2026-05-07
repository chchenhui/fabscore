import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

ablation_types = ["vision_dropout", "vision_no_dropout"]
datasets = ["mnist", "fashion_mnist", "svhn"]

for ablation in ablation_types:
    for ds in datasets:
        try:
            epochs = experiment_data[ablation][ds]["epochs"]
            tloss = experiment_data[ablation][ds]["losses"]["train"]
            vloss = experiment_data[ablation][ds]["losses"]["val"]
            plt.figure()
            plt.plot(epochs, tloss, label="Train Loss")
            plt.plot(epochs, vloss, label="Val Loss")
            plt.title(f"{ds.upper()} Loss Curves ({ablation})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            fname = f"{ablation}_{ds}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve ({ablation}, {ds}): {e}")
            plt.close()

        try:
            tacc = experiment_data[ablation][ds]["metrics"]["train"]
            vacc = experiment_data[ablation][ds]["metrics"]["val"]
            plt.figure()
            plt.plot(epochs, tacc, label="Train Accuracy")
            plt.plot(epochs, vacc, label="Val Accuracy")
            plt.title(f"{ds.upper()} Accuracy Curves ({ablation})")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            fname = f"{ablation}_{ds}_acc_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy curve ({ablation}, {ds}): {e}")
            plt.close()

        try:
            train_logic = experiment_data[ablation][ds]["metrics"]["train_logic"]
            val_logic = experiment_data[ablation][ds]["metrics"]["val_logic"]
            plt.figure()
            plt.plot(epochs, train_logic, label="Train Logic Consistency")
            plt.plot(epochs, val_logic, label="Val Logic Consistency")
            plt.title(f"{ds.upper()} Logical Consistency Accuracy ({ablation})")
            plt.xlabel("Epoch")
            plt.ylabel("Logic Consistency Accuracy")
            plt.legend()
            plt.tight_layout()
            fname = f"{ablation}_{ds}_logic_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating logic consistency plot ({ablation}, {ds}): {e}")
            plt.close()

        try:
            vacc = experiment_data[ablation][ds]["metrics"]["val"]
            val_logic = experiment_data[ablation][ds]["metrics"]["val_logic"]
            plt.figure()
            plt.plot(epochs, vacc, label="Validation Accuracy")
            plt.plot(epochs, val_logic, label="Validation Logic Consistency")
            plt.title(f"{ds.upper()} Val Acc vs. Logic Acc ({ablation})")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            fname = f"{ablation}_{ds}_valacc_vs_logic.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating val acc vs logic plot ({ablation}, {ds}): {e}")
            plt.close()

# Across ablations: plot comparison for validation accuracy and logical consistency
for ds in datasets:
    try:
        plt.figure()
        for ablation in ablation_types:
            try:
                epochs = experiment_data[ablation][ds]["epochs"]
                vacc = experiment_data[ablation][ds]["metrics"]["val"]
                plt.plot(epochs, vacc, label=f"Val Acc - {ablation}")
            except Exception as e:
                print(f"Err in val acc comparison ({ablation}, {ds}): {e}")
        plt.title(f"{ds.upper()} Validation Accuracy by Ablation")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        fname = f"{ds}_valacc_ablation_compare.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating ablation valacc compare ({ds}): {e}")
        plt.close()
    try:
        plt.figure()
        for ablation in ablation_types:
            try:
                epochs = experiment_data[ablation][ds]["epochs"]
                vlogic = experiment_data[ablation][ds]["metrics"]["val_logic"]
                plt.plot(epochs, vlogic, label=f"Logic Acc - {ablation}")
            except Exception as e:
                print(f"Err in logic acc comparison ({ablation}, {ds}): {e}")
        plt.title(f"{ds.upper()} Logical Consistency by Ablation")
        plt.xlabel("Epoch")
        plt.ylabel("Logical Consistency Accuracy")
        plt.legend()
        plt.tight_layout()
        fname = f"{ds}_logicacc_ablation_compare.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating ablation logicacc compare ({ds}): {e}")
        plt.close()
