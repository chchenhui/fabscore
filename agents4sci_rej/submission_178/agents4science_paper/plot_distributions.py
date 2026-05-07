# save as plot_distributions.py
import os
import csv
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ---------- Helper: safe load CSV ----------
def try_load_numeric_series(csv_path, colname):
    vals = []
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                v = float(row[colname])
                if np.isfinite(v):
                    vals.append(v)
            except:
                pass
    return vals

def try_load_categorical_series(csv_path, colname):
    vals = []
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            v = row.get(colname, "")
            if v is not None and v != "":
                vals.append(v)
    return vals

# ---------- Numeric: floor area hist ----------
# Expected CSV schema example:
#   sim_numeric.csv with column: floor_area_m2
#   real_numeric.csv with column: floor_area_m2
sim_num = try_load_numeric_series("sim_numeric.csv", "floor_area_m2")
real_num = try_load_numeric_series("real_numeric.csv", "floor_area_m2")

if sim_num is None or real_num is None:
    # fallback demo data
    rng = np.random.default_rng(7)
    sim_num = list(rng.normal(2000, 500, 400))
    real_num = list(rng.normal(3200, 900, 120))

# Determine common bin edges
all_vals = np.array(sim_num + real_num)
bins = np.histogram_bin_edges(all_vals, bins='auto')

plt.figure()  # 1 plot per figure
plt.hist(sim_num, bins=bins, alpha=0.5, label="Simulation", density=True)
plt.hist(real_num, bins=bins, alpha=0.5, label="Real", density=True)
plt.xlabel("Floor area (m²)")
plt.ylabel("Density")
plt.title("Distribution: Simulation vs Real (Floor Area)")
plt.legend()
plt.tight_layout()
plt.savefig("dist_numeric_floor_area.pdf")
plt.close()

# ---------- Categorical: building type bar ----------
# Expected CSV schema example:
#   sim_cat.csv with column: building_type
#   real_cat.csv with column: building_type
sim_cat = try_load_categorical_series("sim_cat.csv", "building_type")
real_cat = try_load_categorical_series("real_cat.csv", "building_type")

if sim_cat is None or real_cat is None:
    # fallback demo categories
    sim_cat = ["SFH", "MF_2_4", "MF_5p", "Office", "School"]
    real_cat = ["SFH", "MF_5p", "Office"]
    # generate random counts
    rng = np.random.default_rng(3)
    sim_cat = list(rng.choice(sim_cat, 400, replace=True, p=[0.5,0.15,0.15,0.15,0.05]))
    real_cat = list(rng.choice(real_cat, 120, replace=True, p=[0.6,0.25,0.15]))

cnt_sim = Counter(sim_cat)
cnt_real = Counter(real_cat)
all_keys = sorted(set(cnt_sim.keys()) | set(cnt_real.keys()))

sim_vals = [cnt_sim.get(k, 0) for k in all_keys]
real_vals = [cnt_real.get(k, 0) for k in all_keys]

x = np.arange(len(all_keys))
width = 0.4

plt.figure()  # separate figure
plt.bar(x - width/2, sim_vals, width, label="Simulation")
plt.bar(x + width/2, real_vals, width, label="Real")
plt.xticks(x, all_keys, rotation=20)
plt.ylabel("Count")
plt.title("Categorical Distribution: Simulation vs Real (Building Type)")
plt.legend()
plt.tight_layout()
plt.savefig("dist_categorical_building_type.pdf")
plt.close()

print("Saved: dist_numeric_floor_area.pdf, dist_categorical_building_type.pdf")
