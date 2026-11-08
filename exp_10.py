import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Loading Iris dataset...")

iris = datasets.load_iris()

X = iris.data[:, (2, 3)]
y = iris.target

y_binary = (y == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Linear SVM...")

svm_clf = SVC(kernel="linear", C=1.0, random_state=42)
svm_clf.fit(X_train_scaled, y_train)

def plot_svc_decision_boundary(svm_model, xmin, xmax):
    w = svm_model.coef_[0]
    b = svm_model.intercept_[0]

    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = (-w[0] / w[1] * x0 - b / w[1])

    margin_up = decision_boundary + 1 / w[1]
    margin_down = decision_boundary - 1 / w[1]

    plt.plot(x0, decision_boundary, "k-", linewidth=2, label="Decision Boundary")
    plt.plot(x0, margin_up, "k--", linewidth=1.5, label="Margin")
    plt.plot(x0, margin_down, "k--", linewidth=1.5)

    svs = svm_model.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors="#FFAAAA", edgecolors="r", label="Support Vectors")

plt.figure(figsize=(10, 6))
plt.plot(X_train_scaled[y_train == 1, 0], X_train_scaled[y_train == 1, 1], "bo", label="Setosa")
plt.plot(X_train_scaled[y_train == 0, 0], X_train_scaled[y_train == 0, 1], "gs", label="Non-Setosa")
plot_svc_decision_boundary(svm_clf, X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5)
plt.xlabel("Petal Length (Scaled)")
plt.ylabel("Petal Width (Scaled)")
plt.title("Linear SVM: Setosa vs. Non-Setosa")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.axis([
    X_train_scaled[:, 0].min() - 0.5,
    X_train_scaled[:, 0].max() + 0.5,
    X_train_scaled[:, 1].min() - 0.5,
    X_train_scaled[:, 1].max() + 0.5
])
plt.tight_layout()
plt.show()

