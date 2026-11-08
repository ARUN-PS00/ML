import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = sns.load_dataset('mpg')

df = df.dropna(subset=['mpg', 'displacement'])

X = df[['displacement']].values
y = df['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for degree in [1, 2, 3, 4]:
    if degree == 1:
        Xtr = X_train
        Xte = X_test
    else:
        poly = PolynomialFeatures(degree=degree)
        Xtr = poly.fit_transform(X_train)
        Xte = poly.transform(X_test)

    model = LinearRegression()
    model.fit(Xtr, y_train)

    y_pred = model.predict(Xte)

    print("Degree:", degree)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

