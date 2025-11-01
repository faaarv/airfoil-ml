# check range, outliers, skewness, and correlation
import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns

# ===== LOAD DATA =====
y_profile = np.load("y_profile.npy")       
conditions = np.load("conditions.npy")   # (Re, Alpha, Ncrit)
coefficients = np.load("coefficients.npy")  #(Cl, Cd, Cdp, Cm)
pressures = np.load("pressures.npy")
airfoil_names = np.load("airfoil_names.npy")

print("Profiles shape:", y_profile.shape)
print("Pressures shape:", pressures.shape)
print("conditions shape:", conditions.shape)
print("coefficients shape:", coefficients.shape)


def summary_stats(name, arr):
    print(f"\n{name} statistics:")
    print(f"  min:  {np.min(arr):.6f}")
    print(f"  max:  {np.max(arr):.6f}")
    print(f"  mean: {np.mean(arr):.6f}")
    print(f"  std:  {np.std(arr):.6f}")
    print(f"  skew: {skew(arr.flatten()):.6f}")
    
    mean = np.mean(arr)
    std = np.std(arr)
    outliers = np.sum((arr < mean - 3*std) | (arr > mean + 3*std))
    print(f"  outliers (>|3σ|): {outliers} ({100*outliers/arr.size:.3f}%)")

summary_stats("Profiles", y_profile)
summary_stats("Pressures", pressures)
summary_stats("Re", conditions[:,0])
summary_stats("Alpha", conditions[:,1])
summary_stats("Ncrit", conditions[:,2])
summary_stats("CL", coefficients[:,0])
summary_stats("Cd", coefficients[:,1])
summary_stats("Cdp", coefficients[:,2])
summary_stats("Cm", coefficients[:,3])


assert len(y_profile) == len(conditions) == len(coefficients) == len(pressures) == len(airfoil_names)
print(y_profile.dtype, conditions.dtype, coefficients.dtype, pressures.dtype)

features = np.hstack([y_profile, conditions])
labels = np.hstack([coefficients, pressures])
print("\nFeatures shape:", features.shape)
print("Labels shape:", labels.shape)

# ===== CORRELATION CHECK =====
columns = ['Re', 'Alpha', 'Ncrit', 'CL', 'Cd', 'Cdp', 'Cm']
data_for_corr = np.hstack([conditions, coefficients])
df = pd.DataFrame(data_for_corr, columns=columns)

corr = df.corr()
print("\nCorrelation matrix:\n", corr)

# ===== visualize correlation heatmap =====
plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap: Conditions & Coefficients")
plt.tight_layout()
plt.show()