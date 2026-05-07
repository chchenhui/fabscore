"""Visualize p_no_speech distribution across UrbanSound8K clips.
Separates hallucinating vs non-hallucinating clips (from Condition A),
overlays tau thresholds at 0.5 and 0.6.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from schm.evaluation.results_io import load_results_json

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def plot_distribution():
    cond_a = load_results_json(RESULTS_DIR / "condition_a_urbansound8k.json")
    p_nospeech_data = load_results_json(RESULTS_DIR / "p_nospeech_urbansound8k.json")

    p_map = {r["clip_id"]: r["p_no_speech"] for r in p_nospeech_data}

    hall_p = []
    non_hall_p = []

    for clip in cond_a:
        cid = clip["clip_id"]
        if cid not in p_map:
            continue
        p_val = p_map[cid]
        is_hall = len(clip["transcription"].strip()) > 0
        if is_hall:
            hall_p.append(p_val)
        else:
            non_hall_p.append(p_val)

    print(f"Hallucinating clips: {len(hall_p)}", flush=True)
    print(f"Non-hallucinating clips: {len(non_hall_p)}", flush=True)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 51)

    if len(hall_p) > 0:
        ax.hist(hall_p, bins=bins, alpha=0.6, color="red", label=f"Hallucinating (n={len(hall_p)})", edgecolor="darkred", linewidth=0.5)
    if len(non_hall_p) > 0:
        ax.hist(non_hall_p, bins=bins, alpha=0.6, color="blue", label=f"Non-hallucinating (n={len(non_hall_p)})", edgecolor="darkblue", linewidth=0.5)

    ax.axvline(x=0.5, color="orange", linestyle="--", linewidth=2, label="tau = 0.5")
    ax.axvline(x=0.6, color="green", linestyle="-", linewidth=2, label="tau = 0.6 (default)")

    ax.set_xlabel("p_no_speech", fontsize=13)
    ax.set_ylabel("Number of clips", fontsize=13)
    ax.set_title("Distribution of p_no_speech on UrbanSound8K\n(Condition A: Default Whisper-large-v3)", fontsize=14)
    ax.legend(fontsize=11, loc="upper center")
    ax.set_xlim(0, 1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "p_nospeech_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}", flush=True)


if __name__ == "__main__":
    plot_distribution()
