# Visualize flip rate analysis: grouped bar chart of C->W and W->C rates per method,
# plus a per-category breakdown table for MMStar. Reads flip_rate_analysis.json.
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


METHOD_LABELS = {
    "vanilla": "Vanilla",
    "visual_replay": "Visual Replay",
    "adaptive_mi": "Adaptive MI",
}
METHOD_ORDER = ["vanilla", "visual_replay", "adaptive_mi"]
COLORS_CW = "#d62728"
COLORS_WC = "#2ca02c"


def plot_grouped_bars(ax, data, benchmark="mmstar"):
    methods = [m for m in METHOD_ORDER if m in data["methods"] and benchmark in data["methods"][m]]
    if not methods:
        ax.text(0.5, 0.5, f"No data for {benchmark}", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(methods))
    width = 0.35

    cw_rates = [data["methods"][m][benchmark]["cw_rate"] for m in methods]
    wc_rates = [data["methods"][m][benchmark]["wc_rate"] for m in methods]

    bars_cw = ax.bar(x - width / 2, cw_rates, width, label="Correct\u2192Wrong", color=COLORS_CW, alpha=0.85, edgecolor="black", linewidth=0.5)
    bars_wc = ax.bar(x + width / 2, wc_rates, width, label="Wrong\u2192Correct", color=COLORS_WC, alpha=0.85, edgecolor="black", linewidth=0.5)

    for bar in bars_cw:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_wc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=11)
    ax.set_ylabel("Flip Rate (%)", fontsize=11)
    ax.set_title(f"Flip Rates on {benchmark.upper()} (128\u2192512 tokens)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, max(max(cw_rates), max(wc_rates)) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_category_table(ax, data):
    cat_data = data.get("mmstar_category_breakdown", {})
    methods = [m for m in METHOD_ORDER if m in cat_data]
    if not methods:
        ax.text(0.5, 0.5, "No category data available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    all_cats = set()
    for m in methods:
        all_cats.update(cat_data[m].keys())
    categories = sorted(all_cats)

    col_labels = ["Category"]
    for m in methods:
        col_labels.extend([f"{METHOD_LABELS.get(m, m)}\nC\u2192W%", f"{METHOD_LABELS.get(m, m)}\nW\u2192C%", f"{METHOD_LABELS.get(m, m)}\nNet%"])

    table_data = []
    for cat in categories:
        row = [cat]
        for m in methods:
            c = cat_data[m].get(cat, {})
            row.append(f"{c.get('cw_rate', 0):.1f}")
            row.append(f"{c.get('wc_rate', 0):.1f}")
            row.append(f"{c.get('net_flip_rate', 0):.1f}")
        table_data.append(row)

    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D9E2F3")
        cell.set_edgecolor("#888888")

    ax.set_title("MMStar Flip Rates by Category (128\u2192512 tokens)", fontsize=12, fontweight="bold", pad=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.input_file is None:
        args.input_file = os.path.join(project_root, "mi_decoding", "results", "flip_rate_analysis.json")
    if args.output_file is None:
        args.output_file = os.path.join(project_root, "mi_decoding", "results", "flip_rate_analysis.pdf")

    with open(args.input_file) as f:
        data = json.load(f)

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_grouped_bars(ax1, data, "mmstar")

    ax2 = fig.add_subplot(gs[0, 1])
    plot_grouped_bars(ax2, data, "hallusionbench")

    ax3 = fig.add_subplot(gs[1, :])
    make_category_table(ax3, data)

    fig.suptitle("Budget Flip Rate Analysis: Short (128) vs Long (512) Tokens", fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(args.output_file, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {args.output_file}")

    png_path = args.output_file.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"PNG saved to {png_path}")

    plt.close()


if __name__ == "__main__":
    main()
