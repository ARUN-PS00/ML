import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

X = df[['AveRooms']].values
y = df['Price'].values

mask = X[:, 0] < 15
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]

theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

theta_gd = np.random.randn(2)
m = len(X_train)
eta = 0.05

for i in range(1000):
    gradient = 2/m * X_train.T @ (X_train @ theta_gd - y_train)
    theta_gd = theta_gd - eta * gradient

y_pred = X_test @ theta_gd

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

plt.scatter(X_test[:, 1], y_test)
plt.plot(X_test[:, 1], y_pred, color='red')
plt.show()

