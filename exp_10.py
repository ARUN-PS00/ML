import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

w = svm.coef_[0]
b = svm.intercept_[0]

x = np.linspace(-2, 2, 100)
boundary = -(w[0] / w[1]) * x - b / w[1]

upper = boundary + 1 / w[1]
lower = boundary - 1 / w[1]

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

plt.plot(x, boundary, 'k-')
plt.plot(x, upper, 'k--')
plt.plot(x, lower, 'k--')

sv = svm.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], s=150, facecolors='none', edgecolors='red')

plt.title("Linear SVM Decision Boundary")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
