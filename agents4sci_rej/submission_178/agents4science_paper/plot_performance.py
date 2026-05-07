# save as plot_performance.py
import numpy as np
import matplotlib.pyplot as plt

# Numbers from the paper
metrics = ["MAE (kWh/mo)", "RMSE (kWh/mo)", "R²"]
baseline = np.array([127.95, 151.31, -2.44], dtype=float)
hybrid   = np.array([ 58.25,  76.97,  0.10], dtype=float)

# ---------- Absolute comparison ----------
x = np.arange(len(metrics))
width = 0.35

plt.figure()
plt.bar(x - width/2, baseline, width, label="Baseline GBM")
plt.bar(x + width/2, hybrid,   width, label="Hybrid")
plt.xticks(x, metrics, rotation=0)
plt.ylabel("Value")
plt.title("Performance Comparison (Absolute)")
plt.legend()
plt.tight_layout()
plt.savefig("perf_absolute.pdf")
plt.close()

# ---------- Relative improvement ----------
# Define "improvement" as (baseline - hybrid)/baseline for error metrics (MAE/RMSE)
# For R², define improvement as (hybrid - baseline) (absolute delta), since larger is better.
improve = np.zeros_like(baseline)
# MAE (index 0) and RMSE (index 1): percentage reduction
for idx in [0,1]:
    if baseline[idx] != 0:
        improve[idx] = (baseline[idx] - hybrid[idx]) / baseline[idx] * 100.0
    else:
        improve[idx] = np.nan

# R²: absolute delta (not percent)
improve[2] = hybrid[2] - baseline[2]

# Plot: put MAE/RMSE as %, R² as absolute delta; annotate axis labels accordingly
labels = ["MAE reduction (%)", "RMSE reduction (%)", "R² absolute Δ"]

plt.figure()
plt.bar(np.arange(len(labels)), improve)
plt.xticks(np.arange(len(labels)), labels, rotation=10)
plt.ylabel("Value")
plt.title("Hybrid Gains over Baseline")
plt.tight_layout()
plt.savefig("perf_improvement.pdf")
plt.close()

print("Saved: perf_absolute.pdf, perf_improvement.pdf")
