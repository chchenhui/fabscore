import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/chenhui/fabscore/aiscientist_papers/mdl_grokking_correlation")
INPUT_PATH = ROOT / "fabscore_codex/workspace/claim_25_x_div_y/all_results.npy"
OUT_DIR = ROOT / "fabscore_codex/workspace/claim_28_x_div_y"
SUMMARY_PATH = OUT_DIR / "summary.json"
PLOT_PATH = OUT_DIR / "mdl_gen_gap_x_div_y_reproduced.png"


def load_results():
    return np.load(INPUT_PATH, allow_pickle=True).item()


def build_seed_summary(results):
    seed_ids = sorted(
        {
            key.removeprefix("x_div_y_").removesuffix("_mdl_info")
            for key in results
            if key.startswith("x_div_y_") and key.endswith("_mdl_info")
        }
    )
    per_seed = []
    for seed in seed_ids:
        train_info = results[f"x_div_y_{seed}_train_info"]
        val_info = results[f"x_div_y_{seed}_val_info"]
        mdl_info = results[f"x_div_y_{seed}_mdl_info"]

        step = np.array([item["step"] for item in val_info], dtype=float)
        val_acc = np.array([item["val_accuracy"] for item in val_info], dtype=float)
        train_acc = np.array(
            [item["train_accuracy"] for item in train_info], dtype=float
        )
        mdl_step = np.array([item["step"] for item in mdl_info], dtype=float)
        mdl = np.array([item["mdl"] for item in mdl_info], dtype=float)
        val_acc_interp = np.interp(mdl_step, step, val_acc)
        train_acc_interp = np.interp(mdl_step, step, train_acc)
        gen_gap = train_acc_interp - val_acc_interp

        per_seed.append(
            {
                "seed": int(seed),
                "step": step.tolist(),
                "mdl_step": mdl_step.astype(int).tolist(),
                "mdl": mdl.astype(int).tolist(),
                "val_acc_interp": val_acc_interp.tolist(),
                "train_acc_interp": train_acc_interp.tolist(),
                "gen_gap": gen_gap.tolist(),
                "mdl_first": float(mdl[0]),
                "mdl_last": float(mdl[-1]),
                "mdl_min": float(np.min(mdl)),
                "mdl_max": float(np.max(mdl)),
                "gen_gap_first": float(gen_gap[0]),
                "gen_gap_last": float(gen_gap[-1]),
                "gen_gap_min": float(np.min(gen_gap)),
                "gen_gap_max": float(np.max(gen_gap)),
            }
        )
    return per_seed


def aggregate(seed_rows):
    mdl_steps = np.array(seed_rows[0]["mdl_step"], dtype=int)
    mdl_matrix = np.array([row["mdl"] for row in seed_rows], dtype=float)
    gap_matrix = np.array([row["gen_gap"] for row in seed_rows], dtype=float)
    mean_mdl = mdl_matrix.mean(axis=0)
    mean_gap = gap_matrix.mean(axis=0)
    return {
        "reused_artifact": str(INPUT_PATH),
        "mdl_steps": mdl_steps.tolist(),
        "mean_mdl": mean_mdl.tolist(),
        "mean_gen_gap": mean_gap.tolist(),
        "mean_mdl_first": float(mean_mdl[0]),
        "mean_mdl_last": float(mean_mdl[-1]),
        "mean_mdl_drop": float(mean_mdl[0] - mean_mdl[-1]),
        "mean_gen_gap_first": float(mean_gap[0]),
        "mean_gen_gap_last": float(mean_gap[-1]),
        "mean_gen_gap_drop": float(mean_gap[0] - mean_gap[-1]),
        "mean_gen_gap_min": float(np.min(mean_gap)),
        "mean_gen_gap_max": float(np.max(mean_gap)),
    }


def save_plot(seed_rows, aggregate_summary):
    mdl_steps = np.array(aggregate_summary["mdl_steps"], dtype=float)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for row in seed_rows:
        plt.plot(row["mdl_step"], row["mdl"], label=f"seed_{row['seed']} - MDL")
    plt.plot(mdl_steps, aggregate_summary["mean_mdl"], color="black", linewidth=2, label="mean MDL")
    plt.title("MDL Evolution and Generalization Gap - x_div_y")
    plt.ylabel("MDL")
    plt.legend()

    plt.subplot(2, 1, 2)
    for row in seed_rows:
        plt.plot(row["mdl_step"], row["gen_gap"], label=f"seed_{row['seed']} - Gen Gap")
    plt.plot(
        mdl_steps,
        aggregate_summary["mean_gen_gap"],
        color="black",
        linewidth=2,
        label="mean Gen Gap",
    )
    plt.xlabel("Steps")
    plt.ylabel("Generalization Gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    seed_rows = build_seed_summary(results)
    aggregate_summary = aggregate(seed_rows)
    save_plot(seed_rows, aggregate_summary)
    payload = {
        "seed_summaries": seed_rows,
        "aggregate": aggregate_summary,
        "plot_code_alignment": (
            "Matches plot.py Figure-4 logic for x_div_y: "
            "gen_gap = interp(train_acc, mdl_step) - interp(val_acc, mdl_step), "
            "with raw mdl/mdL_step drawn from x_div_y_*_mdl_info."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
