import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -----------------------------
# Data: Claim-level (Stacked)
# -----------------------------
VERDICT_COUNTS = {
    "aiscientist": {
        "Verified": 1048,
        "Insufficient Evidence": 35,
        "No Code Files": 11,
        "Data Fabrication": 0,
        "Experiment Fabrication": 108,
        "Result Fabrication": 69,
    },
    "mlragent": {
        "Verified": 221,
        "Insufficient Evidence": 27,
        "No Code Files": 12,
        "Data Fabrication": 72,
        "Experiment Fabrication": 183,
        "Result Fabrication": 15,
    },
    "agents4sci_acc": {
        "Verified": 482,
        "Insufficient Evidence": 920,
        "No Code Files": 751,
        "Data Fabrication": 4,
        "Experiment Fabrication": 222,
        "Result Fabrication": 59,
    },
    "agents4sci_rej": {
        "Verified": 416,
        "Insufficient Evidence": 89,
        "No Code Files": 140,
        "Data Fabrication": 23,
        "Experiment Fabrication": 661,
        "Result Fabrication": 58,
    },
    "fars": {
        "Verified": 1324,
        "Insufficient Evidence": 15,
        "No Code Files": 8,
        "Data Fabrication": 0,
        "Experiment Fabrication": 0,
        "Result Fabrication": 5,
    },
}

source_keys = ["aiscientist", "mlragent", "agents4sci_acc", "agents4sci_rej", "fars"]
sources = ["AI Scientist", "MLR-Agent", "Agents4Sci (Acc.)", "Agents4Sci (Rej.)", "FARS"]
categories = ["Verified", "Unverifiable", "Data Fabrication", "Experiment Fabrication", "Result Fabrication"]

claim_totals = []
claim_data_pct = {cat: [] for cat in categories}

for sk in source_keys:
    counts = VERDICT_COUNTS[sk]
    total = sum(counts.values())
    claim_totals.append(total)
    claim_data_pct["Verified"].append(counts.get("Verified", 0) / total)
    claim_data_pct["Unverifiable"].append((counts.get("Insufficient Evidence", 0) + counts.get("No Code Files", 0)) / total)
    claim_data_pct["Data Fabrication"].append(counts.get("Data Fabrication", 0) / total)
    claim_data_pct["Experiment Fabrication"].append(counts.get("Experiment Fabrication", 0) / total)
    claim_data_pct["Result Fabrication"].append(counts.get("Result Fabrication", 0) / total)

# -----------------------------
# Data: Paper-level (Simple)
# -----------------------------
PAPER_COUNTS = {
    "aiscientist":  {"total": 30, "with_fab": 13},
    "mlragent":     {"total": 30, "with_fab": 15},
    "agents4sci_acc": {"total": 27, "with_fab": 16},
    "agents4sci_rej": {"total": 27, "with_fab": 22},
    "fars":         {"total": 30, "with_fab":  3},
}

paper_totals  = [PAPER_COUNTS[k]["total"] for k in source_keys]
paper_fab_pct = [PAPER_COUNTS[k]["with_fab"] / PAPER_COUNTS[k]["total"] for k in source_keys]

# -----------------------------
# Plot Settings: COLOSSAL VISIBILITY
# -----------------------------
plt.rcParams.update({
    "axes.unicode_minus": False,
    "font.family": "sans-serif",
    "axes.labelsize": 36, 
    "xtick.labelsize": 30, 
    "ytick.labelsize": 34, 
    "legend.fontsize": 34, 
})

colors = {
    "Verified": "#9ecae1",
    "Unverifiable": "#cccccc", 
    "Data Fabrication": "#ffcc80",
    "Experiment Fabrication": "#ff7f0e",
    "Result Fabrication": "#d62728",
    "Paper Fab": "#1f77b4"
}

# -----------------------------
# Combined Figure: Symmetrical Layout
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(36, 16), dpi=200, gridspec_kw={'width_ratios': [1.1, 1]})

y_pos = np.arange(len(sources))
bar_height = 0.8
pct_font_size = 34 
title_size = 44
label_size = 36


TITLE_Y = 1.05
LABEL_Y = -0.15

# --- Left Plot: Claim Verdicts ---
left = np.zeros(len(sources))
stagger_counts = np.zeros(len(sources))

for cat in categories:
    widths = np.array(claim_data_pct[cat])
    ax1.barh(y_pos, widths, left=left, height=bar_height, color=colors[cat], edgecolor="white", linewidth=2.0, label=cat, zorder=3)
    
    for i, w in enumerate(widths):
        if w <= 0: continue
        label_text = f"{w*100:.1f}"
        if w > 0.08: 
            text_color = "white" if cat == "Result Fabrication" else "#000000"
            ax1.text(left[i] + w/2, y_pos[i], label_text, ha='center', va='center', 
                     fontsize=pct_font_size, color=text_color, fontweight='bold', zorder=5)
        else:
            count = int(stagger_counts[i])
            is_top = (count % 2 == 0)
            dy = -0.28 if is_top else -0.02
            x_loc = 1.02 + (count // 2) * 0.22 
            ax1.text(x_loc, y_pos[i] + dy, "■", ha='left', va='center', 
                     fontsize=30, color=colors[cat], fontweight='bold', zorder=5)
            ax1.text(x_loc + 0.04, y_pos[i] + dy, label_text, ha='left', va='center', 
                     fontsize=pct_font_size, color="#000000", fontweight='bold', zorder=5)
            stagger_counts[i] += 1
            
    left += widths

unified_yticklabels = [f"{name}\n({c_n} claims, {t_n} papers)" for name, c_n, t_n in zip(sources, claim_totals, paper_totals)]
ax1.set_yticks(y_pos)
ax1.set_yticklabels(unified_yticklabels, fontsize=34, fontweight='normal', color="#000000")

for label in ax1.get_yticklabels():
    label.set_linespacing(1.5)

ax1.text(0.5, TITLE_Y, "Claim-level Verdict Distribution", transform=ax1.get_xaxis_transform(),
         ha='center', va='bottom', fontsize=title_size, fontweight='bold', color="#000000")
ax1.text(0.5, LABEL_Y, "Percentage of Claims (%)", transform=ax1.get_xaxis_transform(),
         ha='center', va='top', fontsize=label_size, fontweight='bold', color="#000000")

ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_xlim(0, 1.35) 
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%' if x <= 1.0 else ""))
ax1.spines['bottom'].set_bounds(0, 1.0)
ax1.invert_yaxis()

# --- Right Plot: Paper Fabrication Rate ---
ax2.barh(y_pos, paper_fab_pct, height=bar_height, color=colors["Paper Fab"], edgecolor="white", linewidth=2.0, zorder=3)

for i, p in enumerate(paper_fab_pct):
    ax2.text(p + 0.02, y_pos[i], f"{p*100:.1f}", ha='left', va='center', 
             fontsize=pct_font_size, fontweight='bold', color="#000000", zorder=5)

ax2.text(0.5, TITLE_Y, "Paper-level Fabrication Frequency", transform=ax2.get_xaxis_transform(),
         ha='center', va='bottom', fontsize=title_size, fontweight='bold', color="#000000")
ax2.text(0.5, LABEL_Y, "Percentage of Papers (%)", transform=ax2.get_xaxis_transform(),
         ha='center', va='top', fontsize=label_size, fontweight='bold', color="#000000")

ax2.set_yticks(y_pos)
ax2.set_yticklabels([]) 
ax2.set_xlim(0, 1.25) 
ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%' if x <= 1.0 else ""))
ax2.spines['bottom'].set_bounds(0, 1.0)
ax2.invert_yaxis()

# Style
for ax in [ax1, ax2]:
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0)
    ax.tick_params(axis='x', colors='#000000')
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)

# Legend
handles1, labels1 = ax1.get_legend_handles_labels()
paper_fab_handle = Patch(facecolor=colors["Paper Fab"], edgecolor="white", linewidth=2.0, label="Papers with Fabrication")
handles = handles1 + [paper_fab_handle]
labels = labels1 + ["Papers with Fabrication"]

leg = fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.09), 
                 frameon=False, fontsize=34, borderpad=1.2, handlelength=1.4)

plt.tight_layout()
# Further increased wspace from 0.06 to 0.10 for more distinct separation
plt.subplots_adjust(bottom=0.2, wspace=0.10)

_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combined_verdict_plots.pdf")
plt.savefig(_out, bbox_inches="tight")
print(f"Saved optimized wider-spaced combined plot to {_out}")