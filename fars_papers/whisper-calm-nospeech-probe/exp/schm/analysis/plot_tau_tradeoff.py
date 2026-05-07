"""Plot tau tradeoff: UrbanSound8K hallucination rate vs LibriSpeech WER.
X-axis = US8K hallucination rate (%), Y-axis = LS WER (%).
Shows Condition A, Condition B, and SCHM-suppress at tau={0.5, 0.6, 0.7}.
SCHM points connected to show the tradeoff frontier.
Saves to schm/figures/tau_tradeoff.png.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "tau_robustness.json")) as f:
        data = json.load(f)

    rows = data["rows"]

    cond_a = [r for r in rows if r["tau"] == "inf"][0]
    cond_b = [r for r in rows if r["tau"] == 0][0]
    schm_rows = sorted([r for r in rows if r["mode"] == "suppress"], key=lambda r: r["tau"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, wer_key, split_label in [
        (axes[0], "ls_clean_wer_pct", "LibriSpeech test-clean"),
        (axes[1], "ls_other_wer_pct", "LibriSpeech test-other"),
    ]:
        schm_x = [r["us8k_halluc_rate_pct"] for r in schm_rows]
        schm_y = [r[wer_key] for r in schm_rows]

        ax.plot(schm_x, schm_y, "o-", color="#2196F3", markersize=10, linewidth=2,
                label="SCHM suppress", zorder=5)

        for r in schm_rows:
            tau_val = r["tau"]
            ax.annotate(
                f"  $\\tau$={tau_val}",
                (r["us8k_halluc_rate_pct"], r[wer_key]),
                fontsize=10, ha="left", va="bottom",
            )

        ax.scatter(
            [cond_a["us8k_halluc_rate_pct"]], [cond_a[wer_key]],
            marker="s", s=120, color="#4CAF50", zorder=6, label="Cond A (default)"
        )
        ax.annotate(
            "  A (default)", (cond_a["us8k_halluc_rate_pct"], cond_a[wer_key]),
            fontsize=10, ha="left", va="top",
        )

        ax.scatter(
            [cond_b["us8k_halluc_rate_pct"]], [cond_b[wer_key]],
            marker="D", s=120, color="#F44336", zorder=6, label="Cond B (always mask)"
        )
        ax.annotate(
            "  B (always mask)", (cond_b["us8k_halluc_rate_pct"], cond_b[wer_key]),
            fontsize=10, ha="left", va="bottom",
        )

        ax.set_xlabel("UrbanSound8K Hallucination Rate (%)", fontsize=12)
        ax.set_ylabel(f"{split_label} WER (%)", fontsize=12)
        ax.set_title(f"Hallucination vs WER Tradeoff ({split_label})", fontsize=13)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)

        all_x = schm_x + [cond_a["us8k_halluc_rate_pct"], cond_b["us8k_halluc_rate_pct"]]
        all_y = schm_y + [cond_a[wer_key], cond_b[wer_key]]
        x_margin = (max(all_x) - min(all_x)) * 0.15 + 1
        y_margin = (max(all_y) - min(all_y)) * 0.2 + 0.1
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "tau_tradeoff.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    avg_wer = [(r["ls_clean_wer_pct"] + r["ls_other_wer_pct"]) / 2 for r in schm_rows]
    schm_x = [r["us8k_halluc_rate_pct"] for r in schm_rows]

    ax2.plot(schm_x, avg_wer, "o-", color="#2196F3", markersize=10, linewidth=2,
             label="SCHM suppress", zorder=5)
    for r, aw in zip(schm_rows, avg_wer):
        ax2.annotate(f"  $\\tau$={r['tau']}", (r["us8k_halluc_rate_pct"], aw),
                     fontsize=10, ha="left", va="bottom")

    a_avg = (cond_a["ls_clean_wer_pct"] + cond_a["ls_other_wer_pct"]) / 2
    b_avg = (cond_b["ls_clean_wer_pct"] + cond_b["ls_other_wer_pct"]) / 2

    ax2.scatter([cond_a["us8k_halluc_rate_pct"]], [a_avg],
                marker="s", s=120, color="#4CAF50", zorder=6, label="Cond A (default)")
    ax2.annotate("  A (default)", (cond_a["us8k_halluc_rate_pct"], a_avg),
                 fontsize=10, ha="left", va="top")

    ax2.scatter([cond_b["us8k_halluc_rate_pct"]], [b_avg],
                marker="D", s=120, color="#F44336", zorder=6, label="Cond B (always mask)")
    ax2.annotate("  B (always mask)", (cond_b["us8k_halluc_rate_pct"], b_avg),
                 fontsize=10, ha="left", va="bottom")

    ax2.set_xlabel("UrbanSound8K Hallucination Rate (%)", fontsize=12)
    ax2.set_ylabel("LibriSpeech Avg WER (clean+other) (%)", fontsize=12)
    ax2.set_title("SCHM Tau Sensitivity: Hallucination vs WER Tradeoff", fontsize=13)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path2 = os.path.join(FIGURES_DIR, "tau_tradeoff_avg.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path2}")


if __name__ == "__main__":
    main()
