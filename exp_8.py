import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = fetch_openml(
    'Fashion-MNIST', version=1,
    return_X_y=True, as_frame=False
)

X = X / 255.0

X, _, y, _ = train_test_split(
    X, y, train_size=15000,
    stratify=y, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

k_values = [1, 3, 5, 7, 9, 15]

accuracies = []
times = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    start = time.time()
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    end = time.time()
    
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    times.append(end - start)
    
    print("K:", k)
    print("Accuracy:", acc)
    print("Time:", end - start)

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy")
plt.show()

plt.plot(k_values, times, marker='o')
plt.xlabel("K Value")
plt.ylabel("Execution Time")
plt.title("KNN Execution Time")
plt.show()
