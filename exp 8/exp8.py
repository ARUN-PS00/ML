import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Downloading Fashion MNIST dataset (this may take a moment)...")
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='auto')
X = X / 255.0
X_subset, _, y_subset, _ = train_test_split(X, y, train_size=15000, stratify=y, random_state=45)

X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("-" * 40)

k_values = [1, 3, 5, 7, 9, 15]
accuracies = []
execution_times = []

print("Starting KNN evaluation...")

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    start_time = time.time()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    end_time = time.time()
    acc = accuracy_score(y_test, y_pred)
    total_time = end_time - start_time
    accuracies.append(acc)
    execution_times.append(total_time)
    print(f"K = {k:2d} | Accuracy: {acc*100:.2f}% | Time taken: {total_time:.2f} seconds")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, accuracies, marker='o', color='b')
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.7)
plt.subplot(1, 2, 2)
plt.plot(k_values, execution_times, marker='o', color='r')
plt.title('Execution Time vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Time (seconds)')
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

