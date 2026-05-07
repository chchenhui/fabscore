import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1. Generate synthetic dataset
np.random.seed(42)
x = np.linspace(40, 90, 20)  # vaccination coverage (%)
y_true = 15 - 0.12 * x       # baseline trend
y = y_true + np.random.normal(0, 0.5, len(x))  # add Gaussian noise

# 2. Fit linear regression (OLS via polyfit)
coeffs = np.polyfit(x, y, 1)
y_fit = np.polyval(coeffs, x)

# 3. Evaluation metrics
mse = np.mean((y - y_fit)**2)
r, _ = pearsonr(x, y)
print(f"Correlation coefficient r = {r:.2f}")
print(f"Mean Squared Error MSE = {mse:.2f}")

# 4. Visualization
plt.figure(figsize=(7,5))
plt.scatter(x, y, color="steelblue", edgecolor="black", s=70, label="Observed Data")
plt.plot(x, y_fit, color="darkred", linestyle="--", linewidth=2.5, label="Fitted Regression Line")
plt.xlabel("Vaccination Coverage (%)", fontsize=12)
plt.ylabel("Influenza Hospitalization Rate\n(per 100,000 population)", fontsize=12)
plt.title("Relationship between Vaccination Coverage and Hospitalization", fontsize=13, weight="bold")
plt.legend(frameon=True, fontsize=11, loc="upper right", fancybox=True, shadow=True)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("public_health_case.pdf", dpi=300)
plt.show()
