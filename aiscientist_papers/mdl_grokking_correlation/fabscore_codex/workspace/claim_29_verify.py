import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/chenhui/fabscore/aiscientist_papers/mdl_grokking_correlation")
WORKSPACE = ROOT / "fabscore_codex" / "workspace"
OUT_DIR = WORKSPACE / "claim_29_run_1"
SUMMARY_PATH = WORKSPACE / "claim_29_summary.json"
COMMAND_LOG_PATH = WORKSPACE / "claim_29_command_output.txt"


def append_command_output(command, proc):
    with COMMAND_LOG_PATH.open("a") as f:
        f.write("===== COMMAND =====\n")
        f.write(command + "\n")
        f.write("----- OUTPUT -----\n")
        if proc.stdout:
            f.write(proc.stdout)
            if not proc.stdout.endswith("\n"):
                f.write("\n")
        if proc.stderr:
            f.write(proc.stderr)
            if not proc.stderr.endswith("\n"):
                f.write("\n")


def run_experiment():
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(WORKSPACE / "mplconfig")
    env.setdefault("PYTHONUNBUFFERED", "1")
    command = [
        sys.executable,
        "experiment.py",
        "--out_dir",
        str(OUT_DIR.relative_to(ROOT)),
    ]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    append_command_output(" ".join(command), proc)
    proc.check_returncode()


def load_run_results():
    return np.load(OUT_DIR / "all_results.npy", allow_pickle=True).item()


def summarize_dataset(results_dict, dataset):
    val_losses = []
    train_losses = []
    val_accs = []
    train_accs = []
    mdl_data = []
    steps = None

    for key, info in results_dict.items():
        if dataset in key and "val_info" in key:
            steps = np.array([item["step"] for item in info], dtype=int)
            val_losses.append([item["val_loss"] for item in info])
            val_accs.append([item["val_accuracy"] for item in info])
        if dataset in key and "train_info" in key:
            train_losses.append([item["train_loss"] for item in info])
            train_accs.append([item["train_accuracy"] for item in info])
        if dataset in key and "mdl_info" in key:
            mdl_data.append(info)

    val_acc = np.mean(np.array(val_accs, dtype=float), axis=0)
    train_acc = np.mean(np.array(train_accs, dtype=float), axis=0)
    mdl_step = np.array([item["step"] for item in mdl_data[0]], dtype=int)
    mdl = np.array([item["mdl"] for item in mdl_data[0]], dtype=float)
    mdl_diff = np.diff(mdl)
    mdl_transition_idx = int(np.argmin(mdl_diff))
    mdl_transition_point = int(mdl_step[mdl_transition_idx])
    grokking_point = next(
        (int(step) for step, acc in zip(steps, val_acc) if acc >= 0.95),
        None,
    )
    mdl_transition_rate = float(np.min(np.gradient(mdl, mdl_step)))
    grokking_speed = None
    if (
        grokking_point is not None
        and mdl_transition_point is not None
        and grokking_point != mdl_transition_point
    ):
        grokking_speed = float(1 / (grokking_point - mdl_transition_point))

    return {
        "val_step_count": int(len(steps)),
        "mdl_step_count": int(len(mdl_step)),
        "mean_final_val_acc": float(val_acc[-1]),
        "max_mean_val_acc": float(np.max(val_acc)),
        "mdl_transition_point": mdl_transition_point,
        "grokking_point": grokking_point,
        "mdl_transition_rate": mdl_transition_rate,
        "grokking_speed": grokking_speed,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMMAND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_experiment()
    results = load_run_results()
    summary = {
        dataset: summarize_dataset(results, dataset)
        for dataset in ["x_div_y", "x_minus_y", "x_plus_y", "permutation"]
    }
    payload = {
        "regenerated_run_dir": str(OUT_DIR),
        "artifacts_present": sorted(path.name for path in OUT_DIR.iterdir()),
        "figure5_exact_blocker": (
            "plot.py iterates over run_1 through run_4, but only one fresh run-like "
            "artifact was regenerated in this session."
        ),
        "single_run_summary": summary,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
