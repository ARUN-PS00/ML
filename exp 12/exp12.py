import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("Loading Digits dataset...")

digits = load_digits()
X = digits.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = [2, 4, 6, 8, 10, 12]

inertia_values = []
silhouette_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    inertia_values.append(inertia)
    silhouette_values.append(silhouette)
    print(f"K = {k}")
    print(f"Inertia: {inertia:.2f}")
    print(f"Silhouette Score: {silhouette:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o', linewidth=2)
plt.title('K vs Inertia')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_values, marker='o', linewidth=2)
plt.title('K vs Silhouette Score')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

print("\n=======COMPARISON=========")
print("K\tInertia\t\tSilhouette")

for k, inertia, silhouette in zip(k_values, inertia_values, silhouette_values):
    print(f"K = {k}\t{inertia:.2f}\t\t{silhouette:.4f}")

